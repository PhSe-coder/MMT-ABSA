#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=0
export MASTER_ADDR=localhost
export MASTER_PORT=58999
output='./out/mmt_base/'
src_train_dir='./processed/dataset'
tar_train_dir='./processed/dataset'
val_dir='./processed/dataset'
test_dir='./processed/dataset'
src_domain=$1
tar_domain=$2
python run.py -m torch.distributed.launch \
--nproc_per_node=1 \
--local_rank 0 \
--model_name "mmt" \
--output_dir "${output}${src_domain}-${tar_domain}"  \
--train_file "${src_train_dir}/${src_domain}.train.txt" "${tar_train_dir}/${tar_domain}.train.txt" \
--test_file "${test_dir}/${tar_domain}.test.txt" \
--do_train \
--do_predict \
--device "cuda:0" \
--optimizer "bertAdam" \
--lr "3e-5" \
--batch_size 16 \
--num_train_epochs 3 \
--do_eval \
--validation_file "${val_dir}/${tar_domain}.validation.txt"