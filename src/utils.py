from torch.utils.data import Dataset
import torch
import json
import math
import re
import string
import torch.nn.functional as F
from tokenizers import pre_tokenizers, Regex

INSTRUCT_PROMPT = "Generate the entire rest of this text, continuing until you reach the end: "
INSTRUCT_PROMPT_PREFIX = 15

TARGET_LOSS_MULTIPLIER = 1

def is_spacing_or_punctuation(token):
    """
    Check if a token is spacing (including zero-width) or punctuation (including bullets and dashes).
    """
    if not token:
        return True
    
    # Check for whitespace (including newlines, tabs, etc.)
    if token.isspace():
        return True
    
    # Check for zero-width characters
    zero_width_chars = {
        '\u200b',  # Zero Width Space
        '\u200c',  # Zero Width Non-Joiner
        '\u200d',  # Zero Width Joiner
        '\u2060',  # Word Joiner
        '\ufeff',  # Zero Width No-Break Space
    }
    if any(char in token for char in zero_width_chars):
        return True
    
    # Check for punctuation (including standard and extended)
    # Standard punctuation
    if any(char in string.punctuation for char in token):
        return True
    
    # Extended punctuation including bullets and special dashes
    extended_punctuation = {
        '•', '‣', '◦', '▪', '▫', '‰', '‱',  # Bullets and special symbols
        '–', '—', '―', '‒',  # Various dashes
        ''', ''', '"', '"', '…',  # Smart quotes and ellipsis
        '¡', '¿', '§', '¶', '†', '‡',  # Additional punctuation
    }
    if any(char in token for char in extended_punctuation):
        return True
    
    # Check if token contains only punctuation/spacing characters
    if token.strip() == '':
        return True
    
    return False

class CustomDataset(Dataset):

    def __init__(self, dataset, tokenizer, model, unmemorize=False, 
                 unmemorize_start = 7, unmemorize_stride = 8, unmemorize_span = 1, 
                 unmemorize_smart_stride = False,
                 unmemorize_smart_select = False,
                 unmemorize_top_k = 10,
                 max_length=512, instruct=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        if hasattr(model, 'to') and callable(getattr(model, 'to')):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = model.to(device)
        else:
            # If it's an LLM or another object without a 'to' method, use as is
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
        BATCH_SIZE = 24
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
        
        device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'cpu'
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, total_samples)

            # Move the batch tensors to the correct device
            batch_input_ids = self.encoded_articles['input_ids'][start_idx:end_idx].to(device)
            batch_attention_mask = self.encoded_articles['attention_mask'][start_idx:end_idx].to(device)
            batch_labels = self.encoded_articles['input_ids'][start_idx:end_idx].to(device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels,
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
                        k = min(6 * self.unmemorize_top_k, len(response_probs))  # Increased for additional filtering
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
                                
                                # Skip all spacing and punctuation for smart_select
                                if is_spacing_or_punctuation(token_text):
                                    continue
                                
                                # Check space requirement
                                if requires_space:
                                    if not (token_text.startswith(' ') or token_text.startswith('-')):
                                        continue
                                    
                                # if original is whitespace or newline, must pick a non-whitespace token or newline
                                if original_token_text.isspace() and not (token_text.isspace() or token_text == '\n'):
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
                                    if self.unmemorize_smart_select == True:

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
            # Handle empty tokens gracefully
            if token and not re.match(r'^[^\w\s]', token[0]):
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
        """
        Build `self.encoded_articles['unmemorize_mask']`, ensuring that we never pick
        punctuation/space tokens for unmemorization. Modified to only search forward
        from the current position, not backward.
        """
        self.encoded_articles['unmemorize_mask'] = []
        self.encoded_articles['article_unmemorize_ids'] = []

        for encoded_article, attention_mask, answer_index in zip(
            self.encoded_articles['input_ids'],
            self.encoded_articles['attention_mask'],
            self.answer_index
        ):
            
            # Move each to the correct device
            encoded_article = encoded_article.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

            # Do a single‐sample forward to get logits/probs
            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoded_article.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                    labels=encoded_article.unsqueeze(0)
                )

            # Initialize mask to zeros
            unmemorize_mask = torch.zeros_like(attention_mask)

            # Figure out where the real tokens start (skip any initial padding or special tokens)
            # We skip index 0 because it's usually [CLS] or similar
            start_index = (attention_mask != 0).nonzero(as_tuple=True)[0][0].item() + 1

            # skip prompt 
            start_index += answer_index

            # Our "raw" index where we attempt the first unmemorize
            index = start_index + self.unmemorize_start
            L = len(encoded_article)

            while index < L and attention_mask[index] != 0:
                old_index = index
                span_counter = 0
                chosen_spot = None

                # We will attempt up to unmemorize_span tokens "in this window",
                # each time skipping punctuation/spacing until we find a valid token.
                while span_counter < self.unmemorize_span and index < L and attention_mask[index] != 0:
                    # We start with a candidate = index
                    candidate = index
                    found_valid = False

                    # Only search forward from the current candidate position
                    while candidate < L and attention_mask[candidate] != 0:
                        token_id, token_text, prob = self._get_token_info(encoded_article, candidate, outputs)
                        txt = token_text.strip()
                        if (not token_text) or (prob == 1.0) or is_spacing_or_punctuation(token_text) or txt.isdigit():
                            candidate += 1
                            continue
                        # Otherwise, we found a valid token
                        found_valid = True
                        break

                    if found_valid:
                        # Mark this candidate for unmemorization
                        unmemorize_mask[candidate] = 1
                        chosen_spot = candidate
                        span_counter += 1
                        # Move index forward by 1, so if span>1 we look "just after" this candidate next
                        index = candidate + 1
                    else:
                        # No valid token found: abort this span
                        break

                # After finishing this span, if we did choose something, jump to chosen_spot + stride;
                # otherwise jump from original index + stride.
                if chosen_spot is not None:
                    next_index = chosen_spot + self.unmemorize_stride + 1
                    # If that somehow goes backwards, force at least old_index+1
                    index = max(next_index, old_index + 1)
                else:
                    index = old_index + self.unmemorize_stride + 1 

            # Append this article's mask
            self.encoded_articles['unmemorize_mask'].append(unmemorize_mask)

        # Stack them into a tensor of shape [num_articles, seq_len]
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

def get_adaptive_scaling_factor(logger, num_samples, min_scale=1.0, max_scale=3.0, 
                               ref_small=100, ref_large=1000, power=1.0):
    """
    Calculate adaptive scaling factor based on dataset size.
    
    Args:
        num_samples: Number of samples in the dataset
        min_scale: Minimum scaling factor (for small datasets)
        max_scale: Maximum scaling factor (upper limit)
        ref_small: Reference size for small datasets
        ref_large: Reference size for large datasets
        power: Power factor to adjust curve shape (1.0=linear, >1=slower growth)
    
    Returns:
        Scaling factor for the loss
    """
    
    return 1
    
    # For small datasets, use min_scale
    if num_samples <= ref_small:
        return min_scale
    
    # For extremely large datasets, cap at ref_large * 10 to prevent excessive scaling
    effective_samples = min(num_samples, ref_large * 10)
    
    # Normalized position using logarithmic scale
    log_samples = math.log(effective_samples)
    log_small = math.log(ref_small)
    log_large = math.log(ref_large)
    
    # Calculate normalized position in log space
    normalized_position = (log_samples - log_small) / (log_large - log_small)
    
    # Apply power adjustment for non-linear scaling and clamp to [0, 1]
    normalized_position = min(1.0, normalized_position) ** power
    
    # Linear interpolation between min_scale and max_scale
    scale_factor = min_scale + normalized_position * (max_scale - min_scale)
    
    return scale_factor


def calculate_target_loss(logger, model, tokenizer, outputs, labels, attention_mask, unmemorize_mask, debug=False):
    # Constants for loss calculation
    EPSILON = 1e-9
    TARGET_LOSS_EXPONENT = 2   # Keep the same as before
    TARGET_PROB_THRESHOLD = 1e-4  # Keep the same as before
    BARRIER_WEIGHT = 0.4  # Keep the same as before
    THRESHOLD_WEIGHT = 0.000615  # Reduced from 0.6 to prevent explosion

    # Get logits and shift tensors
    logits = outputs[..., :-1, :].contiguous()  # Shape: [batch_size, seq_len-1, vocab_size]
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()
    shift_unmemorize_mask = unmemorize_mask[..., 1:].contiguous()

    # Apply softmax to get probabilities
    batch_size = logits.size(0)
    probabilities = torch.softmax(logits, dim=2)

    # Track total loss and number of unmemorize tokens
    total_loss = 0.0 
    unmemorize_token_count = 0
    
    # Track statistics for debugging
    if debug:
        all_target_probs = []
        all_losses = []

    for batch_idx in range(batch_size):
        # Find where the actual sequence starts (first 1 in attention mask)
        valid_tokens = shift_attention_mask[batch_idx].nonzero().squeeze(-1)
        if len(valid_tokens) == 0:
            continue

        # Process only the valid token positions
        for pos in valid_tokens:
            # Only focus on tokens marked for unmemorization
            if shift_unmemorize_mask[batch_idx, pos] == 1:
                # Get the target token for this position
                target_token = shift_labels[batch_idx, pos]
                # Get the probability of the target token
                pred_probs = probabilities[batch_idx, pos]
                target_prob = pred_probs[target_token]
                
                if debug:
                    all_target_probs.append(target_prob.item())

                # Calculate the combined loss
                # 1. Barrier component: penalizes as prob approaches 1
                barrier_component = (-torch.log(1 - target_prob + EPSILON)) ** TARGET_LOSS_EXPONENT
                
                # 2. Threshold component: directly targets specific threshold
                if target_prob > TARGET_PROB_THRESHOLD:
                    ratio = target_prob / TARGET_PROB_THRESHOLD
                    threshold_component = torch.log(ratio + EPSILON) * ratio
                else:
                    # Small reward for being under threshold (optional)
                    threshold_component = torch.tensor(0.0, device=logits.device)
                
                # Combine the components with weighting
                current_token_loss = (BARRIER_WEIGHT * barrier_component + 
                                      THRESHOLD_WEIGHT * threshold_component)
                
                total_loss += current_token_loss
                unmemorize_token_count += 1

    # Calculate average loss
    if unmemorize_token_count > 0: 
        target_loss = total_loss / unmemorize_token_count
    else:
        target_loss = torch.tensor(0.0, device=logits.device) 

    if debug:
        logger.info(f"Target Loss: {target_loss.item() if isinstance(target_loss, torch.Tensor) else target_loss}")
        if all_target_probs:
            logger.info(f"Target Probabilities: min={min(all_target_probs):.6f}, " 
                        f"max={max(all_target_probs):.6f}, "
                        f"mean={sum(all_target_probs)/len(all_target_probs):.6f}")
            logger.info(f"Tokens above threshold: {sum(1 for p in all_target_probs if p > TARGET_PROB_THRESHOLD)} "
                        f"of {len(all_target_probs)} ({sum(1 for p in all_target_probs if p > TARGET_PROB_THRESHOLD)/len(all_target_probs)*100:.1f}%)")
            
            # Log distribution of probabilities for better insights
            prob_buckets = [0] * 6  # [<1e-6, <1e-5, <1e-4, <1e-3, <1e-2, >=1e-2]
            for p in all_target_probs:
                if p < 1e-6: prob_buckets[0] += 1
                elif p < 1e-5: prob_buckets[1] += 1
                elif p < 1e-4: prob_buckets[2] += 1
                elif p < 1e-3: prob_buckets[3] += 1
                elif p < 1e-2: prob_buckets[4] += 1
                else: prob_buckets[5] += 1
                
            bucket_names = ["<1e-6", "<1e-5", "<1e-4", "<1e-3", "<1e-2", ">=1e-2"]
            for i, (name, count) in enumerate(zip(bucket_names, prob_buckets)):
                logger.info(f"  {name}: {count} tokens ({count/len(all_target_probs)*100:.1f}%)")
    
    return target_loss 

        
def calculate_kl_loss(logger, model, tokenizer, num_samples, 
                      outputs, labels, attention_mask, target_logits, unmemorize_mask, debug = False):
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
                # Get top 5 predicted tokens and their probabilities
                top5_pred_values, top5_pred_indices = torch.topk(pred_probs, min(5, len(pred_probs)))
                top5_pred_tokens = [tokenizer.decode([idx.item()]) for idx in top5_pred_indices]
                
                # Get top 5 target tokens and their probabilities
                top5_target_indices = top_tokens[:min(5, len(top_tokens))]
                top5_target_values = top_probs[:min(5, len(top_probs))]
                top5_target_tokens = [tokenizer.decode([idx.item()]) for idx in top5_target_indices]
                
                # Print position
                logger.info(f"Position {pos.item()-valid_tokens[0]}:")
                
                # Print headers
                logger.info(f"{'TARGET TOKENS':<15} {'PROB':<8} | {'PREDICTED TOKENS':<15} {'PROB':<8}")
                logger.info("-" * 50)
                
                # Print top 5 tokens side by side
                for i in range(max(len(top5_target_tokens), len(top5_pred_tokens))):
                    target_token = top5_target_tokens[i] if i < len(top5_target_tokens) else ""
                    target_prob = f"{top5_target_values[i].item():.4f}" if i < len(top5_target_values) else ""
                    pred_token = top5_pred_tokens[i] if i < len(top5_pred_tokens) else ""
                    pred_prob = f"{top5_pred_values[i].item():.4f}" if i < len(top5_pred_values) else ""
                    
                    logger.info(f"{target_token:<15} {target_prob:<8} | {pred_token:<15} {pred_prob:<8}")
                
                # Highlight match or mismatch for the top token
                if len(top5_pred_indices) > 0 and len(top5_target_indices) > 0:
                    if top5_pred_indices[0] == top5_target_indices[0]:
                        logger.info(f"✓ Top tokens match: {top5_pred_tokens[0]}")
                    else:
                        logger.info(f"✗ Top tokens differ: Target={top5_target_tokens[0]} vs Predicted={top5_pred_tokens[0]}")

            # Always calculate KL divergence for all positions, including unmemorize tokens
            # This properly pushes the model toward the target distribution
            if pos.item()-valid_tokens[0]:
                token_kl = F.kl_div(
                    torch.log(pred_probs + 1e-10),  # Add small epsilon to avoid log(0)
                    target_dist,
                    reduction='sum'
                )
            else:
                token_kl = 0.0
                
            # Add debug information for unmemorize tokens
            if shift_unmemorize_mask[batch_idx, pos] == 1:
                # Get predicted probability for the target token for debugging only
                target_token = shift_labels[batch_idx, pos]
                target_prob = pred_probs[target_token]

                if debug == True and batch_idx == 0 and pos.item()-valid_tokens[0]:
                    logger.info(f"   Unmemorize KL Loss: {token_kl}")
                    logger.info(f"   Current target token prob: {target_prob:.6f}")
            else:
                if debug == True and batch_idx == 0 and pos.item()-valid_tokens[0]:
                    logger.info(f"   Standard KL Loss: {token_kl}")                
            
            total_kl_loss += token_kl
            target_logit_idx += 1
    
    # Normalize by total number of tokens
    total_tokens = shift_attention_mask.sum()
    if total_tokens > 0:
        total_kl_loss = total_kl_loss / total_tokens
    
    scale_factor = get_adaptive_scaling_factor(logger, num_samples)
    total_kl_loss = total_kl_loss * scale_factor
    if debug == True:            
        logger.info(f"Total KL Loss (scale {scale_factor}): {total_kl_loss}")                 
    return total_kl_loss


def calculate_combined_loss( kl_loss, target_loss):
    
    # Combine KL loss and target loss
    combined_loss = kl_loss + TARGET_LOSS_MULTIPLIER * target_loss
    return combined_loss