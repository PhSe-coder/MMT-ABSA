#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=0
output=$1
src_train_dir='./processed/ag_tmp'
val_dir='./processed/dataset'
test_dir='./processed/dataset'
src_domain=$2
tar_domain=$3
python run.py -m torch.distributed.launch --nproc_per_node=1 --local_rank 0 \
--model_name "bert" \
--output_dir "${output}/${src_domain}-${tar_domain}"  \
--train_file "${src_train_dir}/${src_domain}.train.txt" \
--test_file "${test_dir}/${tar_domain}.test.txt" \
--validation_file "${val_dir}/${tar_domain}.validation.txt" \
--do_train \
--do_predict \
--do_eval \
--device "cuda:0" \
--optimizer "rmsprop" \
--lr "1e-4" \
--bert_lr "2e-5" \
--batch_size 16 \
--num_train_epochs 3 \
--seed $4 \
--initializer $5 \
--init_1 /root/graduation/out/contrast/laptop/contrast_laptop_221210_211929.pt
# xavier_uniform_
# kaiming_uniform_