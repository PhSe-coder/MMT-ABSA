#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_OFFLINE=0
export DGLBACKEND=pytorch
output=$1
train_dir='./processed/dataset'
val_dir='./processed/dataset'
test_dir='./processed/dataset'
src_domain=$2
tgt_domain=$3
python -m torch.distributed.launch --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=12345 \
run.py \
--model_name "bert" \
--output_dir "${output}/${src_domain}-${tgt_domain}"  \
--train_file "${train_dir}/${src_domain}.train.txt" "${train_dir}/${tgt_domain}.train.txt" \
--test_file "${test_dir}/${tgt_domain}.test.txt" \
--do_train \
--do_predict \
--optimizer "adamW" \
--l2reg 0.1 \
--warmup 0.3 \
--lr "3e-5" \
--bert_lr "3e-5" \
--batch_size 16 \
--num_workers 16 \
--num_train_epochs 3 \
--seed $4 \
--initializer $5
# xavier_uniform_
# kaiming_uniform_