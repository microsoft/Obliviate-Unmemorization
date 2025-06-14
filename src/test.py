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
MAX_TEST_SAMPLES = 10

import logging
import os
import sys
from pathlib import Path

def setup_logging(log_dir: str | Path,
                  *,
                  greedy: bool = False,
                  insert: bool = False,
                  level: int = logging.INFO) -> logging.Logger:

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    filename = "test_insert.log" if insert else \
               "test_greedy.log" if greedy else "test.log"

    logging.basicConfig(
        level=level,                       # root level
        format="%(message)s",
        handlers=[
            logging.FileHandler(Path(log_dir) / filename, mode="w"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True                         # overwrite any earlier config
    )

    # Return a namespaced logger for the caller’s module
    return logging.getLogger(__name__)


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
    parser.add_argument("--insert", action="store_true", default=False,
                        help="Insert target tokens at specified positions")    
    parser.add_argument("--config", type=str, default=None,
                        help="Config name for targeted token test (e.g., 10-5-1)")
    return parser.parse_args()

def find_longest_common_substring(s1, s2):
    """
    Find the longest common substring between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        (length, substring): Length of the LCS and the actual substring
    """
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

def find_longest_common_word_substring(s1, s2):
    """Find the longest common substring at the word level."""
    if not s1 or not s2:
        return 0, ""
    
    # Split into words
    words1 = s1.split()
    words2 = s2.split()
    
    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    end_pos = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i-1] == words2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
    
    # Join the words back into a string
    longest_word_substring = ' '.join(words1[end_pos - max_length:end_pos])
    return max_length, longest_word_substring

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
            'longest_common_substring': 0,
            'longest_common_word_substring': 0
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
    
    # Calculate longest common substring (character level)
    lcs_length, lcs_text = find_longest_common_substring(text1, text2)
    logger.info("Longest common substring length (characters): %d", lcs_length)
    logger.info("Longest common substring (characters): %s", lcs_text)
    
    # Calculate longest common substring at word level
    lcs_word_length, lcs_word_text = find_longest_common_word_substring(text1, text2)
    logger.info("Longest common substring length (words): %d", lcs_word_length)
    logger.info("Longest common substring (words): %s", lcs_word_text)
    
    # Prepare metrics dictionary
    metrics = {
        'sample_id': sample_id,
        'edit_distance': edit_dist,
        'rouge_scores': rouge_scores,
        'bleu_scores': bleu_scores,
        'longest_common_substring': lcs_length,
        'longest_common_word_substring': lcs_word_length
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
        # Convert PyTorch tensor to a list before passing to vLLM
        batch_prime_ids.append(vllm.inputs.TokensPrompt(prompt_token_ids=prime_ids.tolist() if isinstance(prime_ids, torch.Tensor) else prime_ids))
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
                # Convert PyTorch tensor to a list before passing to vLLM if needed
                all_q_ids = vllm.inputs.TokensPrompt(prompt_token_ids=all_q_ids.tolist() if isinstance(all_q_ids, torch.Tensor) else all_q_ids)
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
            
        # Include offset and prime_length in the metric file identification
        file_suffix = f"{dataset_idx}_offset{starting_offset}_prime{prime_length}"
        print_differences_with_suffix(article_text, generated_article, logger, logging_folder, dataset_idx, offset=starting_offset, file_suffix=file_suffix)

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

def print_differences_with_suffix(text1, text2, logger, logging_folder, sample_id, offset=0, file_suffix=None):   
    logger.info("")
    logger.info("*** Label:\n %s", text1)
    logger.info("*** Generated:\n %s", text2)
    
    if file_suffix is None:
        file_suffix = str(sample_id)
        
    # Compute metrics
    metrics = compute_and_log_metrics_with_suffix(text1, text2, logger, logging_folder, sample_id, offset, file_suffix)
    return metrics

def compute_and_log_metrics_with_suffix(text1, text2, logger, logging_folder, sample_id, offset, file_suffix):
    """Compute and log various metrics between two texts with empty text handling"""
   
    # Check for empty texts
    if not text1 or not text2:
        logger.info("Warning: Empty text detected. Returning zero scores.")
        metrics = {
            'sample_id': sample_id,
            'offset': offset,
            'edit_distance': 0 if not text1 and not text2 else len(text1 or text2),
            'rouge_scores': {
                'rouge1': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rouge2': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rougeL': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0},
                'rougeLsum': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}
            },
            'bleu_scores': {'bleu': 0.0},
            'longest_common_substring': 0,
            'longest_common_word_substring': 0
        }
        
        # Save metrics files with suffix
        rouge_file = os.path.join(logging_folder, f'rouge_sample_{file_suffix}.json')
        bleu_file = os.path.join(logging_folder, f'bleu_sample_{file_suffix}.json')
        metrics_file = os.path.join(logging_folder, f'metrics_sample_{file_suffix}.json')
        
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
    
    # Calculate longest common substring (character level)
    lcs_length, lcs_text = find_longest_common_substring(text1, text2)
    logger.info("Longest common substring length (characters): %d", lcs_length)
    logger.info("Longest common substring (characters): %s", lcs_text)
    
    # Calculate longest common substring at word level
    lcs_word_length, lcs_word_text = find_longest_common_word_substring(text1, text2)
    logger.info("Longest common substring length (words): %d", lcs_word_length)
    logger.info("Longest common substring (words): %s", lcs_word_text)
    
    # Prepare metrics dictionary
    metrics = {
        'sample_id': sample_id,
        'offset': offset,
        'edit_distance': edit_dist,
        'rouge_scores': rouge_scores,
        'bleu_scores': bleu_scores,
        'longest_common_substring': lcs_length,
        'longest_common_word_substring': lcs_word_length
    }
    
    # Save metrics files with suffix for uniqueness
    rouge_file = os.path.join(logging_folder, f'rouge_sample_{file_suffix}.json')
    bleu_file = os.path.join(logging_folder, f'bleu_sample_{file_suffix}.json')
    metrics_file = os.path.join(logging_folder, f'metrics_sample_{file_suffix}.json')
    
    save_metrics_to_json(rouge_scores, rouge_file)
    save_metrics_to_json(bleu_scores, bleu_file)
    save_metrics_to_json(metrics, metrics_file)
    
    return metrics

def generate_with_targeted_tokens(model, tokenizer, batch, batch_sample_indices, config_params, logger, logging_folder):
    """
    Generate text with model but insert target tokens at specified positions.
    This function runs only when greedy flag is passed and instruct is not true.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        batch: Batch of samples
        batch_sample_indices: Indices of samples in the dataset
        config_params: Dictionary with start, stride, and span parameters
        logger: Logger object
        logging_folder: Path to logging folder
    
    Returns:
        List of results with metrics
    """
    device = None
    results = []
    
    # Extract config parameters
    offset = config_params['start']
    stride = config_params['stride']
    span = config_params['span']
    
    logger.info(f"Running targeted token test with config: offset={offset}, stride={stride}, span={span}")
    
    for i, sample in enumerate(batch):
        # Get article tokens and mask
        article_ids = sample['article_ids']
        article_mask = sample['article_mask']
        unmemorize_mask = sample['unmemorize_mask']
        
        # Get the prime tokens (tokens before the starting offset)
        prime_ids = article_ids[:offset+1]
        # Convert PyTorch tensor to a list before passing to vLLM
        prime_ids_prompt = vllm.inputs.TokensPrompt(prompt_token_ids=prime_ids.tolist() if isinstance(prime_ids, torch.Tensor) else prime_ids)
        
        # Identify target positions based on the config parameters
        target_positions = []
        current_pos = offset + 1 
        
        while current_pos < len(article_ids) and article_mask[current_pos] != 0:
            # Add span tokens at current position as targets
            for j in range(span):
                if current_pos + j < len(article_ids) and article_mask[current_pos + j] != 0:
                    target_positions.append(current_pos + j)
            
            # Move to next position after stride
            current_pos += stride + 1
        
        # Log the target positions
        logger.info(f"Sample {batch_sample_indices[i]}: Identified {len(target_positions)} target positions")
        for pos in target_positions:
            token = article_ids[pos].item()
            token_text = tokenizer.decode([token])
            logger.info(f"  Target token at position {pos}: {token_text}")
        
        # Generate text in segments, inserting target tokens
        generated_ids = []
        current_pos = offset + 1
        
        # Add the prime tokens first
        # Convert to list if it's a tensor
        prime_list = prime_ids.tolist() if isinstance(prime_ids, torch.Tensor) else prime_ids
        generated_ids.extend(prime_list)
        
        while current_pos < len(article_ids) and article_mask[current_pos] != 0:
            # Determine end of current segment (before next target)
            next_target = None
            for pos in target_positions:
                if pos > current_pos:
                    next_target = pos
                    break
            
            if next_target is None:
                # No more targets, generate to the end
                segment_length = len(article_ids) - current_pos
            else:
                # Generate up to the next target
                segment_length = next_target - current_pos
            
            if segment_length > 0:
                # Generate the segment
                generation_kwargs = SamplingParams(
                    max_tokens=segment_length,
                    n=1,
                    temperature=0  # Greedy generation
                )
                
                # Create a prompt from the current generated text
                # Convert to list if it's a tensor
                current_prompt = vllm.inputs.TokensPrompt(prompt_token_ids=generated_ids)
                
                # Generate the segment
                segment_output = model.generate(
                    current_prompt,
                    generation_kwargs
                )
                
                # Extract the generated tokens (excluding the prompt)
                segment_tokens = segment_output[0].outputs[0].token_ids
                
                # Add the generated segment
                generated_ids.extend(segment_tokens)
                
            # Move to the next position
            current_pos += segment_length                
            
            # Add the target token if we're at a target position
            if current_pos in target_positions:
                target_token = article_ids[current_pos].item()
                generated_ids.append(target_token)
                    
        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[offset:], skip_special_tokens=True)
        article_text = tokenizer.decode(article_ids[offset:], skip_special_tokens=True)
        
        # Calculate metrics
        logger.info(f"Sample {batch_sample_indices[i]}: Generated text with target tokens inserted")
        file_suffix = f"{batch_sample_indices[i]}_offset{offset}_config{config_params['start']}-{config_params['stride']}-{config_params['span']}"
        metrics = compute_and_log_metrics_with_suffix(
            article_text, 
            generated_text, 
            logger, 
            logging_folder, 
            batch_sample_indices[i], 
            offset,
            file_suffix
        )
        
        # Add to results
        results.append((
            batch_sample_indices[i],
            generated_text == article_text,  # Check if exact match
            generated_text,
            article_text,
            []  # No QA results
        ))
    
    return results

def collect_stats_from_metrics_files(logging_folder, target_offset=0, target_prime_length=10):
    """
    Collects statistics from metrics files in the logging folder.
    If target_offset and target_prime_length are None, collect across all tests.
    Otherwise, filter by the specified configuration.
    
    Args:
        logging_folder: Path to the logging folder
        target_offset: Target offset value to filter metrics, or None for all
        target_prime_length: Target prime length value to filter metrics, or None for all
    """
    # Get all metrics files first
    metrics_files = [f for f in os.listdir(logging_folder) 
                     if f.startswith('metrics_sample_') and f.endswith('.json')]
    
    # If specific targets are provided, filter the files
    if target_offset is not None and target_prime_length is not None:
        target_pattern = f"_offset{target_offset}_prime{target_prime_length}.json"
        filtered_files = [f for f in metrics_files if target_pattern in f]
        
        # Only use filtered files if we found any
        if filtered_files:
            metrics_files = filtered_files
    
    # Also load Rouge and BLEU files
    rouge_files = [f for f in os.listdir(logging_folder) 
                  if f.startswith('rouge_sample_') and f.endswith('.json')]
    bleu_files = [f for f in os.listdir(logging_folder) 
                 if f.startswith('bleu_sample_') and f.endswith('.json')]
    
    # Map file names to their base names for matching
    def get_base_name(filename):
        # Remove prefix and suffix
        if filename.startswith('metrics_sample_'):
            return filename[len('metrics_sample_'):-5]  # Remove .json
        elif filename.startswith('rouge_sample_'):
            return filename[len('rouge_sample_'):-5]
        elif filename.startswith('bleu_sample_'):
            return filename[len('bleu_sample_'):-5]
        return filename
    
    metrics_to_base = {f: get_base_name(f) for f in metrics_files}
    rouge_to_base = {f: get_base_name(f) for f in rouge_files}
    bleu_to_base = {f: get_base_name(f) for f in bleu_files}
    
    # Initialize stats collector - collect per sample first
    sample_metrics = {}
    
    # Read all metrics files
    for file_name in metrics_files:
        base_name = metrics_to_base[file_name]
        file_path = os.path.join(logging_folder, file_name)
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
                
                # Check if this is a targeted metrics file with offset information and we're filtering
                if target_offset is not None and target_prime_length is not None:
                    if 'offset' in metrics and metrics['offset'] != target_offset:
                        continue
                
                # Initialize sample metrics
                sample_data = {
                    'edit_distance': metrics.get('edit_distance', 0),
                    'longest_common_substring': metrics.get('longest_common_substring', 0),
                    'longest_common_word_substring': metrics.get('longest_common_word_substring', 0),
                    'rouge1': 0,
                    'rouge2': 0,
                    'rougeL': 0,
                    'rougeLsum': 0,
                    'bleu': 0
                }
                
                # Find corresponding Rouge file
                matching_rouge_files = [rf for rf, rb in rouge_to_base.items() if rb == base_name]
                if matching_rouge_files:
                    rouge_file = matching_rouge_files[0]
                    rouge_path = os.path.join(logging_folder, rouge_file)
                    try:
                        with open(rouge_path, 'r') as rf:
                            rouge_scores = json.load(rf)
                            # Handle both dictionary and float value formats
                            for rouge_type in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
                                if rouge_type in rouge_scores:
                                    rouge_value = rouge_scores[rouge_type]
                                    # Check if the value is a dictionary or direct float
                                    if isinstance(rouge_value, dict) and 'fmeasure' in rouge_value:
                                        sample_data[rouge_type] = rouge_value['fmeasure']
                                    elif isinstance(rouge_value, (int, float)):
                                        sample_data[rouge_type] = rouge_value
                    except Exception as e:
                        print(f"Error reading Rouge file {rouge_file}: {e}")
                
                # Find corresponding BLEU file
                matching_bleu_files = [bf for bf, bb in bleu_to_base.items() if bb == base_name]
                if matching_bleu_files:
                    bleu_file = matching_bleu_files[0]
                    bleu_path = os.path.join(logging_folder, bleu_file)
                    try:
                        with open(bleu_path, 'r') as bf:
                            bleu_scores = json.load(bf)
                            if 'bleu' in bleu_scores:
                                sample_data['bleu'] = bleu_scores['bleu']
                    except Exception as e:
                        print(f"Error reading BLEU file {bleu_file}: {e}")
                
                # Check if all values are zero for this sample
                all_zero = all(value == 0 for value in sample_data.values())
                if not all_zero:
                    sample_metrics[base_name] = sample_data
                        
        except Exception as e:
            print(f"Error reading metrics file {file_name}: {e}")
    
    # Now aggregate the filtered samples
    all_metrics = {
        'edit_distance': [],
        'longest_common_substring': [],
        'longest_common_word_substring': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'rougeLsum': [],
        'bleu': []
    }
    
    for sample_data in sample_metrics.values():
        for metric_name in all_metrics.keys():
            all_metrics[metric_name].append(sample_data[metric_name])
    
    # Compute statistics
    stats = {}
    for metric_name, values in all_metrics.items():
        if values:  # Only compute if we have values
            stats[metric_name] = {
                'mean': sum(values) / len(values),
                'max': max(values),
                'min': min(values),
                'count': len(values)
            }
    
    return stats

def main():
    args = setup_args()
    
    # Check if logs already exist before proceeding
    if args.greedy:
        log_file = os.path.join(args.logging_folder, 'test_greedy.log')
        summary_file = os.path.join(args.logging_folder, 'test-greedy.json')
    elif args.insert:
        log_file = os.path.join(args.logging_folder, 'test_insert.log')
        summary_file = os.path.join(args.logging_folder, 'test_insert.json')
    else:
        log_file = os.path.join(args.logging_folder, 'test.log')
        summary_file = os.path.join(args.logging_folder, 'test.json')
    
    # Skip testing if log already exists
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        print(f"Log file {summary_file} already exists. Skipping test run.")
        
        # Only generate summary if it doesn't exist yet
        if not os.path.exists(summary_file):
            print(f"Generating summary statistics from existing metrics...")
            summary_stats = collect_stats_from_metrics_files(args.logging_folder, target_offset=None, target_prime_length=None)
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            print(f"Summary statistics saved to {summary_file}")
        return
    
    # If log doesn't exist or is empty, proceed with testing
    logger = setup_logging(
        args.logging_folder,
        greedy=args.greedy,
        insert=args.insert,
        level=logging.INFO   
    )
    
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
    
    num_samples = len(train_dataset) if args.sample_count == -1 else min(args.sample_count, min(MAX_TEST_SAMPLES, len(train_dataset)))
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Number of tested samples: {num_samples}")
    
    # Run the targeted token test if conditions are met
    if args.insert and not args.instruct and args.config:
        # Parse the config name to get parameters
        config_parts = args.config.split('-')
        if len(config_parts) == 3:
            config_params = {
                'start': int(config_parts[0]),
                'stride': int(config_parts[1]),
                'span': int(config_parts[2])
            }
            
            # Load the config file if it exists
            config_file = f"config/runs/{args.config}.json"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config_params = json.load(f)
            
            logger.info(f"Running targeted token test with config: {config_params}")
            
            for batch_id in range(num_batches):
                start_idx = batch_id * BATCH_SIZE
                end_idx = min((batch_id + 1) * BATCH_SIZE, num_samples)
                batch = []
                for i in range(start_idx, end_idx):
                    item = train_dataset[i]
                    item['raw_tokenized_article'] = train_dataset.tokenized_article(i)
                    batch.append(item)
                batch_indices = list(range(start_idx, end_idx))
                
                results = generate_with_targeted_tokens(
                    model, 
                    tokenizer, 
                    batch, 
                    batch_indices, 
                    config_params, 
                    logger, 
                    args.logging_folder
                )
            
            # After all samples are processed, compute and save summary statistics
            summary_stats = collect_stats_from_metrics_files(args.logging_folder, target_offset=None, target_prime_length=None)
            summary_file = os.path.join(args.logging_folder, 'test_insert.json')
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            logger.info(f"Summary statistics saved to {summary_file}")
                
        else:
            logger.error(f"Invalid config format: {args.config}. Expected format: start-stride-span (e.g., 10-5-1)")
    
    else:
        all_results = []
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
            for offset in [0, 20, 50, 100, 200]:
                for prime_length in [10, 50, 100, 200]:
                    results = generate_and_compare_batch(model, args.instruct, tokenizer, batch, batch_indices, starting_offset=offset, prime_length=prime_length, 
                                                        askQuestions=ask_questions, greedy=args.greedy)
                    process_batch_results(results, starting_offset=offset, prime_length=prime_length, askQuestions=ask_questions, logger=logger, 
                                        logging_folder=args.logging_folder)
                    all_results.extend(results)
                    ask_questions = False
        
        # After all samples are processed, compute and save summary statistics
        summary_stats = collect_stats_from_metrics_files(args.logging_folder, target_offset=None, target_prime_length=None)
        if args.greedy:
            summary_file = os.path.join(args.logging_folder, 'test-greedy.json')
        else:
            summary_file = os.path.join(args.logging_folder, 'test.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        logger.info(f"Summary statistics saved to {summary_file}")

if __name__ == "__main__":
    main()