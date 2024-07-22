import logging
import numpy as np
import math
import os
import sys
import glob
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping
from pathlib import Path
import datasets
import torch
from datasets import load_dataset, concatenate_datasets

import transformers
from transformers import AutoConfig, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM


def generate_cache():
    model_name_or_path = 'meta-llama/Meta-Llama-3-8B'
    block_size = 2048
    dataset_dir = sys.argv[1] # '/lscratch/31044685/uniref50_txts'
    cache_dir = '/data/zhongz2/data/cache_dir'
    data_cache_dir = f'/data/zhongz2/data/uniref50_cache_{block_size}'

    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    tokenizer.add_eos_token = True


    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        return output
        

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    # with main_process_first(desc="dataset map tokenization and grouping"):
    lm_datasets = []
    path = Path(dataset_dir)
    files = [file.name for file in path.glob("*.txt")]
    print(files)
    for idx, file in enumerate(files):
        data_file = os.path.join(path, file)
        filename = ''.join(file.split(".")[:-1])
        cache_path = os.path.join(data_cache_dir, filename+f"_{block_size}")
        os.makedirs(cache_path, exist_ok=True)
        # try:
        #     processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
        # except Exception:
        if True:
            cache_dir1 = os.path.join(data_cache_dir, filename+f"_text_{block_size}")
            os.makedirs(cache_dir1, exist_ok=True)
            raw_dataset = load_dataset("text", data_files=data_file, cache_dir=cache_dir1, keep_in_memory=False)
            tokenized_dataset = raw_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=8,
                remove_columns="text",
                load_from_cache_file=True,
                keep_in_memory=False,
                cache_file_names = {k: os.path.join(cache_dir1, 'tokenized.arrow') for k in raw_dataset},
                desc="Running tokenizer on dataset",
            )
            grouped_datasets = tokenized_dataset.map(
                group_texts,
                batched=True,
                num_proc=8,
                load_from_cache_file=True,
                keep_in_memory=False,
                cache_file_names = {k: os.path.join(cache_dir1, 'grouped.arrow') for k in tokenized_dataset},
                desc=f"Grouping texts in chunks of {block_size}",
            )
            processed_dataset = grouped_datasets
            processed_dataset.save_to_disk(cache_path)
        if idx == 0:
            lm_datasets = processed_dataset['train']
        else:
            assert lm_datasets.features.type == processed_dataset["train"].features.type
            lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])
    lm_datasets = lm_datasets.train_test_split(test_size=0.1)

    train_dataset = lm_datasets['train']
    eval_dataset = lm_datasets["test"]

    print('train', train_dataset)
    print('eval', eval_dataset)

if __name__ == '__main__':
    generate_cache()










