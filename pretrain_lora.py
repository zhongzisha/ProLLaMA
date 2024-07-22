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
from transformers import HfArgumentParser, TrainingArguments, Trainer

from peft import (
    TaskType,
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

def accuracy(predictions, references, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.

    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError: # quick fix by simply take the first example
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={})
    


@dataclass
class DataTrainingArguments:
    block_size: Optional[int] = field(default=None, metadata={})
    

@dataclass
class MyTrainingArguments(TrainingArguments):
    # trainable : Optional[str] = field(default="q_proj,v_proj")
    # lora_rank : Optional[int] = field(default=8)
    # lora_dropout : Optional[float] = field(default=0.1)
    # lora_alpha : Optional[float] = field(default=32.)
    # modules_to_save : Optional[str] = field(default=None)
    # debug_mode : Optional[bool] = field(default=False)
    # peft_path : Optional[str] = field(default=None)
    # use_flash_attention_2 : Optional[bool] = field(default=False)
    # double_quant: Optional[bool] = field(default=True)
    # quant_type: Optional[str] = field(default="nf4")
    # load_in_kbits: Optional[int] = field(default=16)
    full_finetuning : Optional[bool] = field(default=False)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_name_or_path = 'meta-llama/Meta-Llama-3-8B'
    block_size = 1024
    dataset_dir = '/lscratch/31044685/uniref50_txts'
    cache_dir = '/data/zhongz2/data/cache_dir'
    data_cache_dir = '/lscratch/31044685/cache_dir'
    data_cache_dir = '/data/zhongz2/data/uniref50_cache'

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
    # path = Path(dataset_dir)
    # files = [file.name for file in path.glob("*.txt")]
    # print(files)
    dirs = glob.glob(os.path.join(data_cache_dir, '*_text_{}'.format(block_size)))
    for idx in range(len(dirs)):
        cache_path = os.path.join(data_cache_dir, f"input{idx}_{block_size}")        
        processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
        
        if idx == 0:
            lm_datasets = processed_dataset['train']
        else:
            assert lm_datasets.features.type == processed_dataset["train"].features.type
            lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])

        if idx == 10: #  debug
            break

    lm_datasets = lm_datasets.train_test_split(test_size=0.1)

    train_dataset = lm_datasets['train']
    eval_dataset = lm_datasets["test"]

    print('train', train_dataset)
    print('eval', eval_dataset)

    device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation='flash_attention_2'
    )
    model.config.use_cache = False
    model_vocab_size = model.get_output_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    if model_vocab_size != tokenizer_vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # peft LoRA
    lora_rank=128
    lora_alpha=256
    lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
    lora_modules_to_save="embed_tokens,lm_head"
    lora_dropout=0.05
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
        target_modules=lora_trainable.split(','),
        inference_mode=False,
        r=lora_rank, 
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        modules_to_save=lora_modules_to_save.split(','))
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())
    print(model)

    if True: # 
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()
        elif hasattr(model.base_model, "get_input_embeddings"):
            def make_inputs_require_grad(_module, _input, _output):
                _output.requires_grad_(True)
            model.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()
    

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    print('begin training')
    trainer.train()


if __name__ == '__main__':
    main()










