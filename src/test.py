import argparse
import torch
import json
import sys
import difflib
import time
from multiprocessing import Pool
from functools import partial
from nltk.metrics.distance import edit_distance
from evaluate import load
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import os
import logging
from vllm import LLM, SamplingParams
import vllm

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CustomDataset

NUM_PPRIME_TOKENS = 6
BATCH_SIZE = 1
NUM_DIFF_CHARS = 500

def setup_logging(logging_folder, greedy = False):
    # Create logging folder if it doesn't exist
    os.makedirs(logging_folder, exist_ok=True)
    
    # Configure logging
    if greedy:
        log_file = os.path.join(logging_folder, 'test_greedy.log')
    else:
        log_file = os.path.join(logging_folder, 'test.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)  # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

def setup_args():
    parser = argparse.ArgumentParser(description='Evaluate model generation')
    parser.add_argument('--model', type=str, required=True,
                      help='Name of the model to evaluate')
    parser.add_argument('--instruct', action="store_true", default=False, help='Instruct model')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Path to the dataset')
    parser.add_argument('--logging_folder', type=str, required=True,
                      help='Path to the logging folder')
    parser.add_argument('--sample_count', type=int, default=-1,
                      help='Number of samples to check (-1 for all)')
    parser.add_argument("--greedy", action="store_true", default=False,
                        help="Use greedy decoding")
    return parser.parse_args()

def find_longest_common_substring(s1, s2):
   if not s1 or not s2:
       return 0, ""
       
   m, n = len(s1), len(s2)
   dp = [[0] * (n + 1) for _ in range(m + 1)]
   max_length = 0
   end_pos = 0
   
   for i in range(1, m + 1):
       for j in range(1, n + 1):
           if s1[i-1] == s2[j-1]:
               dp[i][j] = dp[i-1][j-1] + 1
               if dp[i][j] > max_length:
                   max_length = dp[i][j]
                   end_pos = i
                   
   longest_substring = s1[end_pos - max_length:end_pos]
   return max_length, longest_substring

def save_metrics_to_json(metrics_dict, file_path):
    import json
    with open(file_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

def compute_and_log_metrics(text1, text2, logger, logging_folder, sample_id, offset):
    """Compute and log various metrics between two texts with empty text handling"""
   
    # Check for empty texts
    if not text1 or not text2:
        logger.info("Warning: Empty text detected. Returning zero scores.")
        metrics = {
            'sample_id': sample_id,
            'edit_distance': 0 if not text1 and not text2 else len(text1 or text2),
            'rouge_scores': {
                'rouge1': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rouge2': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rougeL': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rougeLsum': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
            },
            'bleu_scores': {'bleu': 0.0},
            'longest_common_substring': 0
        }
        
        # Save metrics files
        rouge_file = os.path.join(logging_folder, f'rouge_sample_{sample_id}.json')
        bleu_file = os.path.join(logging_folder, f'bleu_sample_{sample_id}.json')
        metrics_file = os.path.join(logging_folder, f'metrics_sample_{sample_id}.json')
        
        save_metrics_to_json(metrics['rouge_scores'], rouge_file)
        save_metrics_to_json(metrics['bleu_scores'], bleu_file)
        save_metrics_to_json(metrics, metrics_file)
        
        return metrics
    
    # Calculate edit distance
    split1 = text1.split()
    split2 = text2.split()
    edit_dist = edit_distance(split1, split2)
    logger.info("Edit distance: %d", edit_dist)
    
    # Calculate ROUGE scores
    rouge = load("rouge")
    rouge_scores = rouge.compute(predictions=[text2], references=[text1])
    logger.info("Rouge score: %s", rouge_scores)
    
    # Calculate BLEU scores with safeguard against empty input
    bleu = load("bleu")
    try:
        bleu_scores = bleu.compute(predictions=[text2], references=[[text1]])
    except (ZeroDivisionError, ValueError):
        logger.info("Warning: BLEU score computation failed. Using zero score.")
        bleu_scores = {'bleu': 0.0}
    logger.info("Bleu score: %s", bleu_scores)
    
    # Calculate longest common substring
    lcs_length, lcs_text = find_longest_common_substring(text1, text2)
    logger.info("Longest common substring length: %d", lcs_length)
    logger.info("Longest common substring: %s", lcs_text)
    
    # Prepare metrics dictionary
    metrics = {
        'sample_id': sample_id,
        'edit_distance': edit_dist,
        'rouge_scores': rouge_scores,
        'bleu_scores': bleu_scores,
        'longest_common_substring': lcs_length
    }
    
    # Save metrics files directly in logging directory
    rouge_file = os.path.join(logging_folder, f'rouge_sample_{sample_id}.json')
    bleu_file = os.path.join(logging_folder, f'bleu_sample_{sample_id}.json')
    metrics_file = os.path.join(logging_folder, f'metrics_sample_{sample_id}.json')
    
    save_metrics_to_json(rouge_scores, rouge_file)
    save_metrics_to_json(bleu_scores, bleu_file)
    save_metrics_to_json(metrics, metrics_file)
    
    return metrics

def print_differences(text1, text2, logger, logging_folder, sample_id, offset):   
    logger.info("")
    logger.info("*** Label:\n %s", text1)
    logger.info("*** Generated:\n %s", text2)
    
    compute_and_log_metrics(text1, text2, logger, logging_folder, sample_id, offset)

def compare_first_chars(str1, str2):
    str1, str2 = str1.strip(), str2.strip()
    if not str1 or not str2:
        return False
    return str1[0] == str2[0]
    
def generate_and_compare_batch(model, instruct, tokenizer, batch, batch_sample_indices, 
                               starting_offset=0, prime_length=NUM_PPRIME_TOKENS, askQuestions=False,
                               greedy=False):     
    device = None
    batch_prime_ids = []
    batch_article_ids = []
    batch_attention_mask = []
    batch_q_ids = []
    batch_q_mask = []
    batch_a_ids = []
    batch_unmemorize_mask = []
    batch_answer_indexes = []
    has_questions = False
    
    for sample in batch:
        
        if instruct:
            article_ids = sample['raw_tokenized_article']
            INSTRUCT_PROMPT = "Generate the entire rest of this text from the start, continuing until you reach the end: "

            article_prefix = tokenizer.decode(article_ids[starting_offset:starting_offset+prime_length], skip_special_tokens=True)
            prompt_conv = [
                {"role": "system", "content": ""},
                {"role": "user", "content": f"{INSTRUCT_PROMPT} {article_prefix}"}
            ]
            instruct_prompt = tokenizer.apply_chat_template(prompt_conv, tokenize=True, add_generation_prompt=True)            
            answer_index = len(instruct_prompt)
            prime_ids = torch.tensor(instruct_prompt, device=device)
            attention_mask = torch.ones(prime_ids.shape, dtype=torch.long, device=device)
        else:
            article_ids = sample['article_ids'][starting_offset:]
            attention_mask = sample['article_mask'][starting_offset:]            
            article_ids = article_ids[attention_mask == 1][starting_offset:]
            prime_ids = article_ids[:prime_length]  
            attention_mask = torch.ones(prime_ids.shape, dtype=torch.long, device=device)
            answer_index = prime_length

        batch_answer_indexes.append(answer_index)
        batch_prime_ids.append(vllm.inputs.TokensPrompt(prompt_token_ids=prime_ids))
        batch_article_ids.append(article_ids)
        batch_attention_mask.append(attention_mask)
        batch_unmemorize_mask.append(sample['unmemorize_mask'])
               
        if 'q_ids' in sample:
            has_questions = True
            batch_q_ids.append(sample['q_ids'])
            batch_a_ids.append(sample['a_ids'])
            batch_q_mask.append(sample['q_mask'])
        else:
            batch_q_ids.append([])
            batch_a_ids.append([])
            batch_q_mask.append([])
    
    batch_attention_mask = torch.stack(batch_attention_mask).to(device)
    
    if instruct:
        max_new_tokens = max(len(ids) for ids in batch_article_ids)
    else:
        max_new_tokens = max(len(ids) for ids in batch_article_ids) - prime_length
    if max_new_tokens > 0:
        with torch.no_grad():
            if greedy:
                temperature = 0
            else:
                temperature = 0.6

            generation_kwargs = SamplingParams(  
                max_tokens = max_new_tokens,  
                n= 1,  
                temperature = temperature 
            )  
            article_outputs = model.generate(
                batch_prime_ids,
                generation_kwargs
            )
        
        results = []        
        if askQuestions and has_questions and starting_offset == 0:
            all_q_ids = []
            all_q_mask = []
            for sample_q_ids, sample_q_mask in zip(batch_q_ids, batch_q_mask):
                if len(sample_q_ids) > 0:
                    all_q_ids.extend(sample_q_ids)
                    all_q_mask.extend(sample_q_mask)
            
            if all_q_ids:
                all_q_ids=vllm.inputs.TokensPrompt(prompt_token_ids=all_q_ids)
                all_q_mask = torch.stack(all_q_mask).to(device)
                
                with torch.no_grad():
                    if greedy:
                        temperature = 0.01
                    else:
                        temperature = 0.6  

                    answer_generation_kwargs = SamplingParams(  
                        max_tokens = 50,  
                        n= 1,  
                        temperature = temperature 
                    )  

                    answer_outputs = model.generate(
                        all_q_ids,
                        answer_generation_kwargs
                    )
                    
        answer_idx = 0
        for i, (article_output, article_ids, answer_index, q_ids, a_ids) in enumerate(zip(article_outputs, batch_article_ids, batch_answer_indexes, batch_q_ids, batch_a_ids)):
            article_output = article_output.outputs[0].token_ids
            if instruct:
                output_length = len(article_output) - answer_index                
                generated_article = tokenizer.decode(article_output[answer_index+prime_length:], skip_special_tokens=True)
                article_text = tokenizer.decode(article_ids[prime_length:], skip_special_tokens=True)                
            else:
                output_length = min(len(article_output), len(article_ids)) - prime_length
                generated_article = tokenizer.decode(article_output[prime_length:prime_length+output_length], skip_special_tokens=True)
                article_text = tokenizer.decode(article_ids[prime_length:], skip_special_tokens=True)
            article_match = generated_article.strip() == article_text.strip()
            
            qa_results = []
            if askQuestions and has_questions and starting_offset == 0:
                for q, a in zip(q_ids, a_ids):
                    generated_answer = tokenizer.decode(answer_outputs[answer_idx][len(q):], skip_special_tokens=True)
                    answer_text = tokenizer.decode(a, skip_special_tokens=True)
                    answer_match = compare_first_chars(generated_answer, answer_text) 
                    qa_results.append((tokenizer.decode(q, skip_special_tokens=True), answer_match, generated_answer, answer_text))
                    answer_idx += 1
            
            # Pass the actual dataset index instead of batch-relative index
            results.append((batch_sample_indices[i], article_match, generated_article, article_text, qa_results))
    else:
        results = []    
    return results

def process_batch_results(results, starting_offset=0, prime_length=NUM_PPRIME_TOKENS, askQuestions=False, logger=None, logging_folder=None):
    for dataset_idx, article_match, generated_article, article_text, qa_results in results:
        # not enough text left in sample
        if starting_offset > 50 and len(generated_article) < starting_offset + 100:
            continue

        if starting_offset == 0:
            logger.info("**********************************************")
        else:
            logger.info("----------------------------------------------")
        logger.info(f"Sample {dataset_idx} Offset {starting_offset} Prime length {prime_length}:")
        
        if not article_match:
            logger.info("Article: Mismatch")
        else:
            logger.info("Article: Match")
        print_differences(article_text, generated_article, logger, logging_folder, dataset_idx, starting_offset)

        if askQuestions and starting_offset == 0 and qa_results:
            for question, answer_match, generated_answer, answer_text in qa_results:
                if not answer_match:
                    logger.info(f"\nQuestion: {question}")
                    logger.info("Answer: Mismatch")
                    logger.info("Label:      %s", answer_text.strip()[0])
                    gen_answer = generated_answer.strip()
                    
                    if len(gen_answer) > 0:
                        logger.info("Generated:  %s", gen_answer[0])
                    else:
                        logger.info("Generated:  [empty]")                    
            
            total_match = sum(answer_match for _, answer_match, _, _ in qa_results)
            total_count = len(qa_results)
            logger.info(f"\nQuestions: {total_match}/{total_count} ({(total_match/total_count)*100:.1f}%)")

def main():
    args = setup_args()
    logger = setup_logging(args.logging_folder, args.greedy )
    
    logger.info("test.py arguments:")
    for arg, val in vars(args).items():
        logger.info(f"   {arg}: {val}")
    logger.info("")
    
    model = LLM(model=args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    raw_dataset = load_from_disk(args.dataset)
    train_dataset = CustomDataset(raw_dataset, tokenizer, model, instruct=args.instruct)
    
    num_samples = len(train_dataset) if args.sample_count == -1 else min(args.sample_count, len(train_dataset))
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_id in range(num_batches):
        start_idx = batch_id * BATCH_SIZE
        end_idx = min((batch_id + 1) * BATCH_SIZE, num_samples)
        batch = []
        for i in range(start_idx, end_idx):
            item = train_dataset[i]
            item['raw_tokenized_article'] = train_dataset.tokenized_article(i)
            batch.append(item)
        batch_indices = list(range(start_idx, end_idx))  # Pass actual dataset indices
        
        ask_questions = False
        for offset in [0, 50, 100, 150, 200]:
            for prime_length in [8, 10, 15, 20]:
                results = generate_and_compare_batch(model, args.instruct, tokenizer, batch, batch_indices, starting_offset=offset, prime_length=prime_length, 
                                                    askQuestions=ask_questions, greedy=args.greedy)
                process_batch_results(results, starting_offset=offset, prime_length=prime_length, askQuestions=ask_questions, logger=logger, 
                                    logging_folder=args.logging_folder)
                ask_questions = False

if __name__ == "__main__":
    main()
