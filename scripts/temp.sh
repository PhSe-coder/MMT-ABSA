#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=0
output='/root/autodl-tmp/out/mmt/'
src_domain=$1
tar_domain=$2
python trainers.py \
--accelerator "gpu" \
--devices 1 \
--log_every_n_steps 10 \
--model_name "mmt" \
--output_dir "${output}${src_domain}-${tar_domain}"  \
--train_file "/root/graduation/processed/tmp/${src_domain}.train.txt" \
"/root/graduation/processed/tmp/${tar_domain}.train.txt" \
"/root/graduation/processed/cont/${src_domain}.train.txt" \
"/root/graduation/processed/cont/${tar_domain}.train.txt" \
--test_file "/root/graduation/processed/dataset/${tar_domain}.test.txt" \
--validation_file "/root/graduation/processed/dataset/${src_domain}.validation.txt" \
--do_train \
--do_predict \
--do_eval \
--enable_progress_bar False \
--num_workers 16 \
--lr "2e-5" \
--tau 1 \
--eta 0.6 \
--alpha 0.02 \
--soft_loss_weight 0.01 \
--theta 0.99 \
--batch_size 16 \
--max_epochs 15 \
--default_root_dir "/root/autodl-tmp" \
--seed 42 \
--tune