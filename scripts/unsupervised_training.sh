#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=0
output='./out/contrast'
train_dir='./processed/ent_desc'
domain=$1
python run.py -m torch.distributed.launch --nproc_per_node=1 --local_rank 0 \
--model_name "contrast" \
--output_dir "${output}/${domain}"  \
--train_file "${train_dir}/${domain}.txt" \
--do_train \
--device "cuda:0" \
--optimizer "rmsprop" \
--lr "1e-3" \
--bert_lr "5e-5" \
--batch_size 16 \
--num_train_epochs 20 \
--seed 42