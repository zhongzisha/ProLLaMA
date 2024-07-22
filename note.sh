
download uniref50.fasta.gz
gzip --decompress uniref50.fasta.gz


# the codes are based on Chinese-LLaMA-Alpaca-2
# Read the wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh) carefully before running the script
export CUDA_VISIBLE_DEVICES=0,1
# export WANDB_PROJECT="continual pretraining"
lr=5e-5
lora_rank=128
lora_alpha=256
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

pretrained_model=meta-llama/Meta-Llama-3-8B  # meta-llama/Llama-2-7b-hf #or your local path
cache_dir=/data/zhongz2/data/cache_dir
dataset_dir=/lscratch/${SLURM_JOB_ID}/uniref50_txts  # ./pretraining_dataset #your dataset path
data_cache=/lscratch/${SLURM_JOB_ID}/cache/
per_device_train_batch_size=4
gradient_accumulation_steps=8
block_size=2048
output_dir=/lscratch/${SLURM_JOB_ID}/output_dir/

deepspeed_config_file=ds_zero2_no_offload.json
torchrun --nproc_per_node 2 --rdzv-endpoint=localhost:29500 pretrain1.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.1 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --seed 42 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 400 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 4 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --load_in_kbits 16 \
    --bf16 \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --modules_to_save ${modules_to_save} \
    --report_to tensorboard \





deepspeed_config_file=ds_zero2_no_offload.json
torchrun --nproc_per_node 2 --rdzv-endpoint=localhost:29500 pretrain1.py \
    --deepspeed ${deepspeed_config_file}


export CUDA_VISIBLE_DEVICES=0,1
lr=5e-5
pretrained_model=meta-llama/Meta-Llama-3-8B  # meta-llama/Llama-2-7b-hf #or your local path
cache_dir=/data/zhongz2/data/cache_dir
per_device_train_batch_size=4
gradient_accumulation_steps=8
block_size=2048
output_dir=/lscratch/${SLURM_JOB_ID}/output_dir/
deepspeed_config_file=ds_zero2_no_offload.json
torchrun --nproc_per_node 2 --rdzv-endpoint=localhost:29500 pretrain1.py \
    --deepspeed ${deepspeed_config_file} \
    --seed 42 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 400 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --report_to tensorboard \
    --bf16 True --tf32 True


# peft
export CUDA_VISIBLE_DEVICES=0,1
lr=5e-5
pretrained_model=meta-llama/Meta-Llama-3-8B  # meta-llama/Llama-2-7b-hf #or your local path
cache_dir=/data/zhongz2/data/cache_dir
per_device_train_batch_size=4
gradient_accumulation_steps=8
block_size=2048
output_dir=/lscratch/${SLURM_JOB_ID}/output_dir/
deepspeed_config_file=ds_zero2_no_offload.json
torchrun --nproc_per_node 2 --rdzv-endpoint=localhost:29500 pretrain_lora.py \
    --deepspeed ${deepspeed_config_file} \
    --seed 42 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 400 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --report_to tensorboard \
    --bf16 True --tf32 True


