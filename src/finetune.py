#!/usr/bin/env python
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import datasets
import torch
import torch.nn as nn

import torch.nn.functional as F
import transformers
from datasets import load_dataset
from datasets import load_from_disk
from huggingface_hub import Repository
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import get_full_repo_name
from transformers.utils.versions import require_version

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import CustomDataset, calculate_kl_loss, get_unmemorize_probabilities, calculate_target_loss, calculate_combined_loss

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--unmemorize",
        action="store_true",
        default=False,
        help="Whether to unmemorize the dataset.",
    )
    parser.add_argument(
        "--unmemorize_start",
        type=int,
        default=0,
        help="The start index to unmemorize.",
    )
    parser.add_argument(
        "--unmemorize_stride",
        type=int,
        default=8,
        help="The stride of the unmemorize tokens.",
    )
    parser.add_argument(
        "--unmemorize_span",
        type=int,
        default=1,
        help="The span of the unmemorize tokens.",
    )  
    parser.add_argument(
        "--unmemorize_smart_stride",
        action="store_true",
        default=False,
        help="Adjust unmemorize positions to find good tokens.",
    )    
    parser.add_argument(
        "--unmemorize_smart_select",
        action="store_true",
        default=False,
        help="Adjust unmemorize token selection to find good tokens.",
    )        
    parser.add_argument(
        "--unmemorize_sample_count", 
        type=int,
        default=-1,
        help="The number of samples to memorize.",
    )
    parser.add_argument(
        "--unmemorize_top_k",
        type=int,
        default=10,
        help="The top k tokens to use for k/l loss.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        choices=[None, "kl", "combined"],
        help="Loss type to use for unmemorization (None for default behavior, 'kl' for KL loss only, 'combined' for combined loss only).",
    )
    parser.add_argument(
        "--instruct_model",
        action="store_true",
        default=False,
        help="Is an instruct model.",
    )    

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print debug information.",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default=None,
        help="Output file for the log.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    # New Code #
    # Whether to load the best model at the end of training
    parser.add_argument(
        "--load_best_model",
        action="store_true",
        help="Whether to load the best model at the end of training",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"`, and `"dvclive"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def evaluate(args, model, tokenizer, num_samples, eval_dataloader, accelerator):
    import torch.distributed as dist
    from accelerate import DistributedType

    model.eval()

    check_only_one = True
    losses = []
    metrics = {
        'max_prob': 0.0,
        'min_prob': 1.0,
        'median_prob': 0.0,
        'max_span_prob': 0.0,
        'kl_loss': 0.0,
        'target_loss': 0.0
    }

    batches_with_probs = 0
    batch_count = 0
    all_probs = []

    for _, batch in enumerate(eval_dataloader):
        batch_count += 1
        with torch.no_grad():
            outputs = model(
                input_ids=batch['article_ids'],
                attention_mask=batch['article_mask'],
                labels=batch['article_ids']
            )

        if args.unmemorize:
            kl_loss = calculate_kl_loss(
                logger, model, tokenizer, num_samples, 
                outputs.logits,
                batch['article_ids'],
                batch['article_mask'],
                batch['target_logits'],
                batch['unmemorize_mask'],
                check_only_one
            )
            target_loss = calculate_target_loss(
                logger, model, tokenizer,
                outputs.logits,
                batch['article_ids'],
                batch['article_mask'],
                batch['unmemorize_mask'],
                check_only_one
            )
            loss = calculate_combined_loss(kl_loss, target_loss)

            # accumulate scalars
            metrics['kl_loss']     += kl_loss.item()     if isinstance(kl_loss, torch.Tensor)     else kl_loss
            metrics['target_loss'] += target_loss.item() if isinstance(target_loss, torch.Tensor) else target_loss

            # gather unmemorize probabilities
            probs = get_unmemorize_probabilities(
                outputs.logits,
                batch['article_ids'],
                batch['article_mask'],
                batch['unmemorize_mask']
            )
            if probs.numel() > 0:
                metrics['max_prob'] = max(metrics['max_prob'], probs.max().item())
                metrics['min_prob'] = min(metrics['min_prob'], probs.min().item())
                all_probs.extend(p.item() for p in probs)
                batches_with_probs += 1

                # span product
                span = args.unmemorize_span
                if len(probs) >= span:
                    num_spans = len(probs) // span
                    for i in range(num_spans):
                        prod = torch.prod(probs[i*span:(i+1)*span]).item()
                        metrics['max_span_prob'] = max(metrics['max_span_prob'], prod)
            else:
                logger.info("Empty probability tensor encountered, skipping metrics for this batch")

            # debug first batch only
            if check_only_one:
                check_only_one = False
                try:
                    debugprobs = get_unmemorize_probabilities(
                        outputs.logits[0].unsqueeze(0),
                        batch['article_ids'][0].unsqueeze(0),
                        batch['article_mask'][0].unsqueeze(0),
                        None
                    )
                    # find first 1 in batch[0] attention mask
                    first_label = (batch['article_mask'][0] == 1).nonzero(as_tuple=True)[0][0].item()
                    for i in range(1, len(debugprobs)):
                        if True: 
                            logger.info(f"[{i}] unmemorize: {batch['unmemorize_mask'][0][first_label+i]}")
                            logger.info(f"  probs:      {debugprobs[i-1]}")
                            logger.info(f"  token:      {tokenizer.decode(batch['article_ids'][0][first_label+i])}")
                            
                            # get top token from the batch target_logits
                            top_tokens_tensor = batch['target_logits']['tokens'][first_label+i-1]['top_tokens']
                            
                            # Construct the vectors for this batch index
                            top_tokens = torch.tensor([tensor[0].item() for tensor in top_tokens_tensor], 
                                                    device=model.device)
                            logger.info(f"  top tokens: {tokenizer.decode(top_tokens[0])}")
                except Exception as e:
                    logger.info(f"Error in debug probabilities calculation: {e}")

        else:
            loss = outputs.loss
            losses.append(
                accelerator.gather_for_metrics(
                    loss.repeat(args.per_device_eval_batch_size)
                )
            )

    # finalize unmemorize metrics
    if args.unmemorize and batch_count > 0:
        metrics['kl_loss']     /= batch_count
        metrics['target_loss'] /= batch_count

        if all_probs:
            all_probs.sort()
            mid = len(all_probs) // 2
            if len(all_probs) % 2 == 0:
                metrics['median_prob'] = 0.5 * (all_probs[mid-1] + all_probs[mid])
            else:
                metrics['median_prob'] = all_probs[mid]
        else:
            logger.info("No probabilities collected for median calculation")
            metrics['median_prob'] = 0.0

    # finalize memorize metrics
    else:
        if losses:
            all_loss_tensor = torch.cat(losses)
            metrics['eval_loss'] = torch.mean(all_loss_tensor).item()
            logger.info(f"Evaluate (memorize): eval_loss={metrics['eval_loss']:.6f}")

    #
    # === CROSS-RANK SYNCHRONIZATION ===
    #
    if accelerator.distributed_type != DistributedType.NO:
        # Gather each metric scalar across ranks
        for key in ['kl_loss', 'target_loss', 'max_prob', 'min_prob', 'median_prob', 'max_span_prob', 'eval_loss']:
            # Skip keys that weren’t set locally
            if key not in metrics:
                continue
            # create 1-element tensor
            t = torch.tensor(metrics[key], device=accelerator.device, dtype=torch.float32)
            # sum across all processes
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            # mean
            metrics[key] = (t / accelerator.num_processes).item()

    logger.info(
        f"Evaluate: kl_loss={metrics['kl_loss']:.6f}, "
        f"target_loss={metrics['target_loss']:.6f}, "
        f"median_prob={metrics['median_prob']:.6f}"
    )
    return metrics


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment

    # when using DeepSpeed, the `gradient_accumulation_steps` is properly set from the DeepSpeed plugin/config
    # or from `accelerate launch` via `--gradient_accumulation_steps`  else
    # defaulting to the passed `args.gradient_accumulation_steps`
    accelerator = (
        Accelerator(
            log_with=args.report_to,
            project_dir=args.output_dir,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        if args.with_tracking
        else Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )     
    logger.info(accelerator.state, main_process_only=False)

    if args.logfile is not None:
        # create the parent directory
        os.makedirs(os.path.dirname(args.logfile), exist_ok=True)
        
        # Create a file handler
        file_handler = logging.FileHandler(args.logfile)
        file_handler.setLevel(logging.INFO)
        log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        date_format = "%m/%d/%Y %H:%M:%S"
        
        # Create a formatter with the specified date format
        formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logconfig = logging.getLogger()
        logconfig.addHandler(file_handler)    
        
    # print the arguments
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")           

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path,trust_remote_code=True)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, padding_side='left', use_fast=not args.use_slow_tokenizer,trust_remote_code=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left', use_fast=not args.use_slow_tokenizer,trust_remote_code=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # if no pad token, set it to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=True
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
    
    for param in model.parameters():
        if param.dtype not in [torch.float16]:
            logger.info("Model mixed precision. Converting to float16.")
            break

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # DataLoaders creation:
    if args.dataset_name is not None:
        raw_dataset = load_from_disk(f"./{args.dataset_name}")
    
        # only take first sample
        if args.unmemorize:
            if args.unmemorize_sample_count != -1:
                raw_dataset = raw_dataset.select(range(min(args.unmemorize_sample_count, len(raw_dataset))))
                
        elif args.unmemorize_sample_count > 0:
            raw_dataset = raw_dataset.select(range(min(args.unmemorize_sample_count, len(raw_dataset))))       
        logger.info(f"Dataset loaded: {min(args.unmemorize_sample_count, len(raw_dataset))}")
        logger.info(f"Number of samples: {len(raw_dataset)}") 
        logger.info(f"length of article: {len(tokenizer.encode(raw_dataset[0]['article'], add_special_tokens=False))}")

        # first article
        logger.info(f"First article: {raw_dataset[0]['article'][:100]}")

        with accelerator.main_process_first():
            train_dataset = CustomDataset(raw_dataset, tokenizer, model, args.unmemorize,
                                        args.unmemorize_start, args.unmemorize_stride, args.unmemorize_span,
                                        args.unmemorize_smart_stride,
                                        args.unmemorize_smart_select,
                                        args.unmemorize_top_k,
                                        instruct=args.instruct_model )

        # duplicate the data 100 times
        num_samples = len(train_dataset) 
        if args.unmemorize:
            num_copies = max(1, 1000 // len(train_dataset))
            #num_copies = 2
            train_dataset = torch.utils.data.ConcatDataset([train_dataset]*num_copies)    
        else:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset]*50)
        
        eval_dataset = train_dataset
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.

        train_dataset = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{args.validation_split_percentage}%:]",
            **dataset_args,
        )
        num_samples = len(train_dataset) 
        if args.validation_split_percentage != "0" and "validation" not in raw_datasets.keys():
            eval_dataset = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
        else: 
            eval_dataset = train_dataset

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # New Code #
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    from offload_adam import StateOffloadAdamW
    optimizer_cls = (
        StateOffloadAdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # New Code #
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        )


    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    )    

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    logger.info(f"")
    # Only show the progress bar once on each machine.
    completed_steps = 0
    starting_epoch = 0
    best_metric = None
    best_metric_checkpoint = None
    
    metrics = evaluate(args, model, tokenizer, num_samples, eval_dataloader, accelerator)
    if metrics['max_prob'] > 0:
        logger.info(f"max_prob: {metrics['max_prob']} max_span_prob: {metrics['max_span_prob']} median_prob: {metrics['median_prob']}")

    # loss type starts as kl, then switches to target then combined
    if args.loss_type is not None:
        loss_type = args.loss_type
        logger.info(f"Using fixed loss type: {loss_type}")
    else:
        loss_type = "kl"
        logger.info("Using dynamic loss type switching (default behavior)")
    last_loss = -1

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0

        active_dataloader = train_dataloader

        total_batches = len(active_dataloader)
        progress_bar = tqdm(total=total_batches, desc=f"Epoch {epoch}", 
                                disable=not accelerator.is_local_main_process)
        for _, batch in enumerate(active_dataloader):

            with accelerator.accumulate(model):
                # Process active batch                
                outputs = model(input_ids=batch['article_ids'], attention_mask=batch['article_mask'], labels=batch['article_ids'])               
                if args.unmemorize:
                    kl_loss = calculate_kl_loss(logger, model, tokenizer, num_samples, outputs.logits, batch['article_ids'], 
                                             batch['article_mask'], batch['target_logits'], batch['unmemorize_mask'])               
                    target_loss = calculate_target_loss(logger, model, tokenizer, outputs.logits, batch['article_ids'],
                                            batch['article_mask'], batch['unmemorize_mask'])
                    if loss_type == "kl":
                        loss = kl_loss
                    elif loss_type == "target":
                        loss = target_loss
                    elif loss_type == "combined":
                        loss = calculate_combined_loss(kl_loss, target_loss)
                else:   
                    loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()   
                progress_bar.update(1)                       
    
            if accelerator.sync_gradients:
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        progress_bar.close()

        metrics = evaluate(args, model, tokenizer, num_samples, eval_dataloader, accelerator)
        
        # Set the evaluation loss based on loss type
        if args.unmemorize:
            if loss_type == "kl":
                eval_loss = metrics['kl_loss']
            elif loss_type == "target":
                eval_loss = metrics['target_loss']
            elif loss_type == "combined":
                # Apply a scaling factor to target_loss when combining
                eval_loss = calculate_combined_loss(metrics['kl_loss'], metrics['target_loss'])
            
            logger.info(f"epoch {epoch}: loss_type: {loss_type} eval_loss: {eval_loss:.6f}")
            if metrics['max_prob'] > 0:
                logger.info(f"epoch {epoch}: max_prob: {metrics['max_prob']:.6f} max_span_prob: {metrics['max_span_prob']:.6f} median_prob: {metrics['median_prob']:.6f}")
        else:
            # For memorize case, use the straight model loss
            eval_loss = metrics['eval_loss']
            logger.info(f"epoch {epoch}: eval_loss: {eval_loss:.6f}")

        if args.with_tracking:
            accelerator.log(
                {
                    "eval_loss": eval_loss,
                    "train_loss": total_loss / len(train_dataloader) if args.with_tracking else 0,
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )
        
        # Early stopping logic
        if args.unmemorize:
            # Apply different early stopping logic based on loss_type
            no_progress = False            
            if last_loss > 0 and last_loss - 0.001 <= eval_loss:
                logger.info(f"epoch {epoch}: No progress from {last_loss:.6f}: {eval_loss:.6f}")
                no_progress = True
            last_loss = eval_loss
            
            if args.loss_type is not None:
                # Fixed loss type - use simpler early stopping
                if loss_type == "kl" and (eval_loss < 0.06 or no_progress):
                    logger.info(f"epoch {epoch}: Terminating with fixed KL loss: {eval_loss:.6f}")
                    break
                elif loss_type == "combined" and (eval_loss < 0.10 or no_progress):
                    logger.info(f"epoch {epoch}: Terminating with fixed combined loss: {eval_loss:.6f}")
                    break
            else:
                # Dynamic loss type switching (original behavior)
                if loss_type == "kl" and (metrics['kl_loss'] < 0.06 or no_progress):
                    logger.info(f"epoch {epoch}: Switching from kl loss to target loss (kl_loss: {metrics['kl_loss']:.6f})")
                    loss_type = "combined"
                    last_loss = -1              
                elif loss_type == "combined" and (eval_loss < 0.10 or no_progress):
                    logger.info(f"epoch {epoch}: Unmemorize terminating due to combined loss < 0.25: {eval_loss:.6f}")
                    last_loss = -1
                    break
        else: 
            # For memorize case, use straight evaluation loss for early stopping
            if eval_loss < 0.015:
                logger.info(f"epoch {epoch}: Memorize terminating early due to low eval_loss: {eval_loss:.6f}")
                break
                                       
        if isinstance(checkpointing_steps, str) and checkpointing_steps == "epoch":
            accelerator.save_state(os.path.join(args.output_dir, f"epoch_{epoch}"))
            
        # get mod of epoch and checkpointing_steps 
        if isinstance(checkpointing_steps, int) and epoch % checkpointing_steps == 0 and epoch != 0:
            best_metric = metrics['eval_loss']
            best_metric_checkpoint = os.path.join(args.output_dir, f"best_checkpoint")
            accelerator.save_state(best_metric_checkpoint)
            accelerator.print(f"New best metric: {best_metric} at epoch {epoch}")
            accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")
            output_dir = f"epoch_{epoch}_perpelxity_loss_{metrics['eval_loss']}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
   
    if args.output_dir is not None:
        accelerator.wait_for_everyone()        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )     
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                    # Handle both tensor and float cases safely
                    if isinstance(eval_loss, torch.Tensor):
                        json_eval_loss = eval_loss.item()
                    else:
                        json_eval_loss = float(eval_loss)
                    json.dump({"eval_loss": json_eval_loss}, f)
            
if __name__ == "__main__":
    main()
