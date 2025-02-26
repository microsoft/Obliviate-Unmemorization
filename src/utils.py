from torch.utils.data import Dataset
import torch
import json
import re
import torch.nn.functional as F
from tokenizers import pre_tokenizers, Regex

INSTRUCT_PROMPT = "Generate the entire rest of this text, continuing until you reach the end: "
INSTRUCT_PROMPT_PREFIX = 15

class CustomDataset(Dataset):

    def __init__(self, dataset, tokenizer, model, unmemorize=False, 
                 unmemorize_start = 7, unmemorize_stride = 8, unmemorize_span = 1, 
                 unmemorize_smart_stride = False,
                 unmemorize_smart_select = False,
                 unmemorize_top_k = 10,
                 max_length=512, instruct=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.instruct = instruct
        
        # save the unmemorize parameters
        self.unmemorize_start = unmemorize_start
        self.unmemorize_stride = unmemorize_stride
        self.unmemorize_span = unmemorize_span
        self.unmemorize_smart_stride = unmemorize_smart_stride
        self.unmemorize_smart_select = unmemorize_smart_select
        self.unmemorize_top_k = unmemorize_top_k
        self.answer_index = []
        
        # tokenize the articles ommitting special tokens
        self.tokenized_articles = [tokenizer.encode(article, add_special_tokens=False) for article in dataset['article']]
        
        # Tokenize all articles at once
        if self.instruct:
            prompts = [' '.join(item['article'].split()[:INSTRUCT_PROMPT_PREFIX]) for item in self.dataset]
            conversations = []
            
            for prompt, item in zip(prompts, self.dataset):
                prompt_conv = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": f"{INSTRUCT_PROMPT} {prompt}"}
                ]
                prompt_only = self.tokenizer.apply_chat_template(prompt_conv, tokenize=False, add_generation_prompt=True)
                
                conversations.append([
                    {"role": "system", "content": ""},
                    {"role": "user", "content": f"{INSTRUCT_PROMPT} {prompt}"},
                    {"role": "assistant", "content": item['article']}
                ])
                self.answer_index.append(len(self.tokenizer.encode(prompt_only)))

            formatted_articles = [
                self.tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=True
                ) for conv in conversations
            ]
            
            # Get unpadded lengths
            no_pad_lengths =  [min(self.max_length, len(self.tokenizer.encode(article))) for article in formatted_articles]
            
            # Do padded encoding
            self.encoded_articles = self.tokenizer(
                formatted_articles,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )
            
            # Calculate padding lengths and adjust indices
            pad_lengths = [self.encoded_articles.input_ids.size(1) - orig_len for orig_len in no_pad_lengths]
            self.answer_index = [idx + pad_len for idx, pad_len in zip(self.answer_index, pad_lengths)]
        else:
            # Original tokenization for non-instruct mode
            self.encoded_articles = self.tokenizer(
                [item['article'] for item in self.dataset],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True,
                return_tensors='pt'
            )
            [self.answer_index.append(0) for _ in range(len(self.dataset))]
        
        # generate the unmemorize mask
        self.target_logits = []
        self.unmemorize_mask = []
        
        # always get the mask
        if unmemorize == True:
            self._apply_unmemorize()
            self._get_target_logits()
        
        # Tokenize all questions and answers     
        self.has_mcq = False  
        if 'mcq_questions' in self.dataset[0]:
            self.has_mcq = True
            self.encoded_questions = []
            self.encoded_options = []
            self.encoded_answers = []            
            for item in self.dataset:
                mcq = json.loads(item['mcq_questions'])
                questions = [q['question'] for q in mcq]
                options = [q['options'] for q in mcq]
                options_flat = [f'A:{d["A"]}' + '\n' + f'B:{d["B"]}' + '\n' + f'C:{d["C"]}' + '\n' + f'D:{d["D"]}' + '\n\nThe answer is letter' for d in options]
                answers = [q['correct_answer'] for q in mcq]
                
                # concatenate the questions and options
                questions_and_options = [f"{q}\n{a}" \
                                    for q, a in zip(questions, options_flat)]

                if self.instruct:
                    # Format questions as chat conversations
                    q_conversations = [
                        [{"role": "user", "content": q}] 
                        for q in questions_and_options
                    ]
                    formatted_questions = [
                        self.tokenizer.apply_chat_template(
                            conv,
                            tokenize=False,
                            add_generation_prompt=True
                        ) for conv in q_conversations
                    ]
                    encoded_q = self.tokenizer(
                        formatted_questions,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                else:
                    encoded_q = self.tokenizer(
                        questions_and_options,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                
                # Options and answers don't need chat format
                encoded_options = self.tokenizer(
                    options_flat,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,    
                    return_tensors='pt'
                )
                encoded_a = self.tokenizer(
                    answers,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                self.encoded_questions.append(encoded_q)
                self.encoded_options.append(encoded_options)
                self.encoded_answers.append(encoded_a)
                
    def _get_target_logits(self):
        BATCH_SIZE = 32
        total_samples = len(self.encoded_articles['input_ids'])
        num_batches = (total_samples + BATCH_SIZE - 1) // BATCH_SIZE

        # Special tokens to skip
        skip_tokens = {
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.bos_token_id,
            self.tokenizer.sep_token_id if hasattr(self.tokenizer, 'sep_token_id') else -1,
            self.tokenizer.cls_token_id if hasattr(self.tokenizer, 'cls_token_id') else -1
        }
        skip_tokens.discard(-1)  # Remove placeholder value if it was added
        
        def matches_capitalization(orig_text: str, new_text: str) -> bool:
            """Check if new_text matches the capitalization pattern of orig_text"""
            # Strip spaces for capitalization check
            orig_stripped = orig_text.lstrip(' -')
            new_stripped = new_text.lstrip(' -')
            
            # Both empty or whitespace
            if not orig_stripped or not new_stripped:
                return True
                
            # Check first letter capitalization
            orig_is_upper = orig_stripped[0].isupper()
            new_is_upper = new_stripped[0].isupper()
            
            return orig_is_upper == new_is_upper
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, total_samples)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=self.encoded_articles['input_ids'][start_idx:end_idx],
                    attention_mask=self.encoded_articles['attention_mask'][start_idx:end_idx],
                    labels=self.encoded_articles['input_ids'][start_idx:end_idx],
                )            
                
            for batch_index in range(start_idx, end_idx):
                local_index = batch_index - start_idx
                unmemorize_mask = self.encoded_articles['unmemorize_mask'][batch_index]
                
                sample_logits = {
                    'tokens': {}
                }
                
                for i in range(len(outputs.logits[local_index])-1):
                    response_probs = torch.softmax(outputs.logits[local_index][i], dim=-1)

                    if unmemorize_mask[i+1] == 1:
                        # Get the actual token we want to remove
                        article_token = self.encoded_articles['input_ids'][batch_index][i+1].item()
                        original_token_text = self.tokenizer.decode([article_token])
                        requires_space = original_token_text.startswith(' ') or original_token_text.startswith('-')
                        
                        # Get more tokens than we need initially to allow for filtering
                        k = min(6 * self.unmemorize_top_k, len(response_probs))  # Increased for additional capitalization filtering
                        top_probs, top_tokens = torch.topk(response_probs, k=k)
                        
                        # Filter out article token and get valid tokens
                        valid_indices = []
                        for j in range(len(top_tokens)):
                            token = top_tokens[j].item()
                            token_text = self.tokenizer.decode([token])
                            
                            # Skip the article token entirely
                            if token == article_token:
                                continue
                            
                            if self.unmemorize_smart_select == True:
                                    
                                # For highest probability token only (after removing article token)
                                if len(valid_indices) == 0:
                                    # Skip if it's a special token
                                    if token in skip_tokens:
                                        continue
                                
                                # Check space requirement
                                if requires_space:
                                    if not (token_text.startswith(' ') or token_text.startswith('-')):
                                        continue
                                
                                # Check capitalization
                                if not matches_capitalization(original_token_text, token_text):
                                    continue
                                    
                            valid_indices.append(j)
                            if len(valid_indices) >= self.unmemorize_top_k:
                                break
                        
                        # If we don't have enough tokens, get more from the distribution
                        while len(valid_indices) < self.unmemorize_top_k:
                            k = min(len(response_probs), k + self.unmemorize_top_k)
                            top_probs, top_tokens = torch.topk(response_probs, k=k)
                            
                            # Continue filtering from where we left off
                            for j in range(len(valid_indices), len(top_tokens)):
                                token = top_tokens[j].item()
                                token_text = self.tokenizer.decode([token])
                                
                                if token != article_token:
                                    # Apply both space and capitalization requirements
                                    if requires_space:
                                        if not (token_text.startswith(' ') or token_text.startswith('-')):
                                            continue
                                    if not matches_capitalization(original_token_text, token_text):
                                        continue
                                    valid_indices.append(j)
                                    if len(valid_indices) >= self.unmemorize_top_k:
                                        break
                            
                            # Break if we've looked through all possible tokens
                            if k == len(response_probs):
                                break
                        
                        # Take the top self.unmemorize_top_k valid tokens (or all we could find)
                        valid_indices = valid_indices[:self.unmemorize_top_k]
                        top_probs = top_probs[valid_indices]
                        top_tokens = top_tokens[valid_indices]
                    else:
                        top_probs, top_tokens = torch.topk(response_probs, k=self.unmemorize_top_k)

                    # normalize top_probs
                    top_probs = top_probs / top_probs.sum()

                    sample_logits['tokens'][i] = {
                        'top_tokens': top_tokens.tolist(),
                        'top_probs': top_probs.tolist(),
                    }

                self.target_logits.append(sample_logits)
            
    def _get_token_info(self, encoding, word_index, outputs):
        """Get token info and probability for a given word index"""
        token_id = encoding[word_index]
        token = self.tokenizer.decode(token_id)
        prob = F.softmax(outputs.logits[0][word_index-1], dim=-1)
        return token_id, token, prob[token_id]
        
    def _find_valid_word_token(self, encoding, word_ids, word_index, outputs):
        while word_index > 0:
            _, token, _ = self._get_token_info(encoding, word_index, outputs)
            # Check if token is start of word (not punctuation/space)  
            if not re.match(r'^[^\w\s]', token[0]):
                word_id = word_ids[word_index]
                # And check we're at start of word
                if word_index == 0 or word_ids[word_index-1] != word_id:
                    break
            word_index -= 1
        return word_index

    def _create_word_ids(self, encoded_tokens):
        # Decode tokens back to text
        decoded_tokens = [self.tokenizer.decode([token]) for token in encoded_tokens]
        
        word_ids = []
        current_word_id = -1     
        
        for i, token in enumerate(decoded_tokens):
            # Skip if empty token
            if not token:
                word_ids.append(-1)
                continue
                
            # Check if token starts new word
            is_new_word = False
            
            # If first token
            if i == 0:
                is_new_word = True
            else:
                # Get first char of current token
                first_char = token[0]
                
                # Check if this is part of a contraction
                is_contraction = re.match(r"(?i:^'s|^'t|^'re|^'ve|^'m|^'ll|^'d)", token)
                
                # Check if it matches pattern for word start
                if not re.match(r'[^\w]', first_char):
                    # It's a letter/number - check if previous token ended a word
                    prev_token = decoded_tokens[i-1]
                    if (not prev_token or prev_token[-1].isspace() or 
                        (re.match(r'[^\w\s]', prev_token[-1]) and not prev_token[-1] == "'")):
                        is_new_word = True
                        
                # If it's a contraction, keep same word_id
                elif is_contraction:
                    is_new_word = False
                # Special case for punctuation/spaces
                elif re.match(r'[^\w]', first_char):
                    is_new_word = True
                        
            if is_new_word:
                current_word_id += 1
                    
            word_ids.append(current_word_id)
        
        return word_ids
            
    def _apply_unmemorize(self):
        self.encoded_articles['unmemorize_mask'] = []
        self.encoded_articles['article_unmemorize_ids'] = []

        for _, (encoded_article, attention_mask) in enumerate(zip(self.encoded_articles['input_ids'], 
                                                                self.encoded_articles['attention_mask'])):

            # get probability of each token
            with torch.no_grad():
                outputs = self.model(input_ids=encoded_article.unsqueeze(0), 
                            attention_mask=attention_mask.unsqueeze(0), 
                            labels=encoded_article.unsqueeze(0))
            
            # initialize the unmemorize mask to all 0s
            unmemorize_mask = torch.zeros_like(attention_mask)
            
            decoded_text = self.tokenizer.decode(encoded_article)
            encoding_ids = self.tokenizer(decoded_text, return_offsets_mapping=True)
            api_word_ids = encoding_ids.word_ids()

            # Get the offset mapping
            encoding = self.encoded_articles['input_ids'][0]
            word_ids = self._create_word_ids(encoding)
            
            # Find the first non-zero element in the attention mask and skip the
            # [CLS] token
            start_index = (attention_mask != 0).nonzero(as_tuple=True)[0][0].item()+1
            
            index = start_index + self.unmemorize_start
            while index < len(encoded_article) and attention_mask[index] != 0:
                
                span_index = 0
                while span_index < self.unmemorize_span:
                    word_index = index 
                    word_id = word_ids[index]
                    
                    if self.unmemorize_smart_stride == True:

                        # Find a suitable word to unmemorize
                        found_unmemorize = False
                        while found_unmemorize == False:                        
                            found_unmemorize = True
                            
                            # skip the special tokens
                            word_index = self._find_valid_word_token(encoding, word_ids, word_index, outputs)
                            _, token, prob = self._get_token_info(encoding, word_index, outputs)
                            

                            # can't unmemorize tokens that have a probability of 1.0
                            if prob == 1.0:
                                word_index -= 1
                                word_id = word_ids[word_index]
                                found_unmemorize = False
                    else:
                        _, token, _ = self._get_token_info(encoding, word_index, outputs)

                    # Set the unmemorize mask
                    unmemorize_mask[word_index] = 1
                    
                    # Move to the next span position
                    span_index += 1
                    index = min(index + span_index, len(encoded_article))
                    
                # Move to the next stride sposition
                index = min(index + self.unmemorize_stride, len(encoded_article))
            
            # Add the unmemorize mask and article_unmemorize_ids for this article
            self.encoded_articles['unmemorize_mask'].append(unmemorize_mask)

        # Convert lists to tensors
        self.encoded_articles['unmemorize_mask'] = torch.stack(self.encoded_articles['unmemorize_mask'])

    def tokenized_article(self, idx):
        return self.tokenized_articles[idx]
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        # Get pre-tokenized article
        article_ids = self.encoded_articles['input_ids'][idx]
        article_mask = self.encoded_articles['attention_mask'][idx]

        # get target logits
        if self.target_logits:
            target_logits = self.target_logits[idx]
        else:
            # provide value that's not None
            target_logits = {'tokens': {}}
            
        # check if it has an unmemorize mask key in the dictionary
        if self.encoded_articles.get('unmemorize_mask') is not None: 
            unmemorize_mask = self.encoded_articles['unmemorize_mask'][idx]
        else:
            # provide value that's not None
            unmemorize_mask = torch.zeros_like(article_mask)
        
        # Get pre-tokenized questions and answers
        if self.has_mcq:
            q_ids = self.encoded_questions[idx]['input_ids']
            q_mask = self.encoded_questions[idx]['attention_mask']
            a_ids = self.encoded_answers[idx]['input_ids']
            a_mask = self.encoded_answers[idx]['attention_mask']
            o_ids = self.encoded_options[idx]['input_ids']
            o_mask = self.encoded_options[idx]['attention_mask']
            
            return {
                'article_ids': article_ids,
                'article_mask': article_mask,
                'unmemorize_mask': unmemorize_mask,
                'target_logits': target_logits,
                'answer_index': self.answer_index[idx],
                'q_ids': q_ids,
                'q_mask': q_mask,
                'a_ids': a_ids,
                'a_mask': a_mask,
                'o_ids': o_ids,
                'o_mask': o_mask,
            }   
        else:
            return {
                'article_ids': article_ids,
                'article_mask': article_mask,
                'unmemorize_mask': unmemorize_mask,
                'target_logits': target_logits,
                'answer_index': self.answer_index[idx]
            }
        

def get_unmemorize_probabilities(logits, labels, attention_mask, unmemorize_mask):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()    
    if unmemorize_mask is not None:
        shift_unmemorize_mask = unmemorize_mask[..., 1:].contiguous()
    else:
        shift_unmemorize_mask = attention_mask[..., :-1].contiguous()
    
    # Get the vocabulary size from the logits
    vocab_size = logits.size(-1)
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(shift_logits, dim=2)
    
    # Flatten the tensors for easier indexing
    flat_probabilities = probabilities.view(-1, vocab_size)
    flat_input_ids = shift_labels.view(-1)
    flat_unmemorize_mask = shift_unmemorize_mask.view(-1)
    
    # Get the indices where unmemorize_mask is True
    unmemorize_indices = flat_unmemorize_mask.nonzero().squeeze()
    
    # Get the corresponding input_ids and probabilities
    target_input_ids = flat_input_ids[unmemorize_indices]
    target_probabilities = flat_probabilities[unmemorize_indices]
    
    # Extract the probabilities of the target tokens
    result_probabilities = target_probabilities[torch.arange(len(target_input_ids)), target_input_ids]
    
    # Set probabilities of 1.0 to 0.0 (with small epsilon for floating point comparison)
    if unmemorize_mask is not None:
        epsilon = 1e-15
        result_probabilities = torch.where(
            (result_probabilities >= 1.0 - epsilon), 
            torch.zeros_like(result_probabilities),
            result_probabilities
        )
    
    return result_probabilities
        
def calculate_kl_loss(logger, model, tokenizer, outputs, labels, attention_mask, target_logits, unmemorize_mask, debug = False):
    logits = outputs[..., :-1, :].contiguous()  # Shape: [batch_size, seq_len-1, vocab_size]
    shift_labels = labels[..., 1:].contiguous()       
    shift_attention_mask = attention_mask[..., 1:].contiguous()
    shift_unmemorize_mask = unmemorize_mask[..., 1:].contiguous()
    
    # Initialize total loss
    total_kl_loss = 0.0
    batch_size = logits.size(0)
    target_logit_idx = 0
    
    for batch_idx in range(batch_size):
        # Find where the actual sequence starts (first 1 in attention mask)
        valid_tokens = shift_attention_mask[batch_idx].nonzero().squeeze(-1)
        if len(valid_tokens) == 0:
            continue
        
        # Process only the valid token positions
        for pos in valid_tokens:
            # Get predicted distribution for current position
            pred_logits = logits[batch_idx, pos]
            pred_probs = torch.softmax(pred_logits, dim=-1)
                
            top_tokens_tensor = target_logits['tokens'][pos.item()]['top_tokens']
            top_probs_tensor = target_logits['tokens'][pos.item()]['top_probs']
            
            # Construct the vectors for this batch index
            top_tokens = torch.tensor([tensor[batch_idx].item() for tensor in top_tokens_tensor], device=model.device)
            top_probs = torch.tensor([tensor[batch_idx].item() for tensor in top_probs_tensor], 
                                            device=model.device, dtype=pred_probs.dtype)            
            # Create target distribution tensor
            target_dist = torch.zeros_like(pred_probs)
            target_dist[top_tokens] = top_probs.clone().detach()

            if debug == True and batch_idx == 0 and pos.item()-valid_tokens[0]:
                # get top predicted token 
                top_pred_token = torch.argmax(pred_probs)
                
                # get top target token
                top_target_token = top_tokens[0]
                
                if top_pred_token == top_target_token:
                    logger.info(f"{pos.item()-valid_tokens[0]-1}: {tokenizer.decode([top_pred_token])}")
                else:
                    logger.info(f"{pos.item()-valid_tokens[0]-1}: *** {tokenizer.decode([top_pred_token])} vs {tokenizer.decode([top_target_token])}")

            # Calculate KL divergence for this position
            if pos.item()-valid_tokens[0]:
                token_kl = F.kl_div(
                    torch.log(pred_probs + 1e-10),  # Add small epsilon to avoid log(0)
                    target_dist,
                    reduction='sum'
                )
            else:
                token_kl = 0.0
                            
            # get probabilities of label ids for unmemorize mask tokens 
            if shift_unmemorize_mask[batch_idx, pos] == 1:
                # Get predicted probability for the target token
                target_token = shift_labels[batch_idx, pos]
                target_prob = pred_probs[target_token]
                
                # Add negative log probability to KL loss
                token_kl = target_prob * 100
                
                # if the target probability is less than 0.01, no need
                # to push it further down
                if target_prob < 0.01:
                    token_kl = 0
                    
                # if target prob is 1.0, then leave it since it won't budge
                if target_prob == 1.0:
                    if debug == True:
                        logger.info(f"   [{pos-1}] Target Prob is 1.0 - skipping")
                    token_kl = 0

                if debug == True and batch_idx == 0 and pos.item()-valid_tokens[0]:
                    logger.info(f"   Unmemorize Loss: {token_kl}")                

            else:
                if debug == True and batch_idx == 0 and pos.item()-valid_tokens[0]:
                    logger.info(f"   Loss: {token_kl}")                
            
            total_kl_loss += token_kl
            target_logit_idx += 1
    
    # Normalize by total number of tokens
    total_tokens = shift_attention_mask.sum()
    if total_tokens > 0:
        total_kl_loss = total_kl_loss / total_tokens
    
    if debug == True:            
        logger.info(f"   Total Unmemorize Loss: {total_kl_loss}")                      
    return total_kl_loss
