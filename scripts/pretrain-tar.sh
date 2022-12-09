#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=0
output='./out/contrast'
tar_train_dir='./processed/ent_desc'
tar_domain=$1
python run.py -m torch.distributed.launch --nproc_per_node=1 --local_rank 0 \
--model_name "contrast" \
--output_dir "${output}/${tar_domain}"  \
--train_file "${tar_train_dir}/${tar_domain}.txt" \
--do_train \
--device "cuda:0" \
--optimizer "rmsprop" \
--lr "1e-3" \
--bert_lr "5e-5" \
--batch_size 16 \
--num_train_epochs 3 \
--init_1 "/root/graduation/out/bert_base/laptop-rest/bert_laptop_rest_val_pre_0.6_val_rec_0.3011_val_f1_0.401.pt"