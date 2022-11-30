#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=0
output='./out/bert_base/'
src_train_dir='./processed1/ag_tmp'
val_dir='./processed1/dataset'
test_dir='./processed1/dataset'
src_domain=$1
tar_domain=$2
python run.py -m torch.distributed.launch --nproc_per_node=1 --local_rank 0 \
--model_name "bert" \
--output_dir "${output}${src_domain}-${tar_domain}"  \
--train_file "${src_train_dir}/${src_domain}.train.txt" \
--test_file "${test_dir}/${tar_domain}.test.txt" \
--validation_file "${val_dir}/${tar_domain}.validation.txt" \
--do_train \
--do_predict \
--do_eval \
--device "cuda:0" \
--optimizer "rmsprop" \
--lr "1e-3" \
--bert_lr "5e-5" \
--batch_size 16 \
--num_train_epochs 3 \
--seed $3 \
--initializer $4
# xavier_uniform_
# kaiming_uniform_