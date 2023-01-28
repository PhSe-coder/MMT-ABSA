#!/bin/bash
domains=('rest' 'service' 'laptop' 'device')

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_OFFLINE=0
export DGLBACKEND=pytorch
output='./out/bert_base/'
train_dir='./processed/dataset'
val_dir='./processed/dataset'
test_dir='./processed/dataset'
for tgt_domain in ${domains[@]};
do
    for src_domain in  ${domains[@]};
    do
        if [ $src_domain != $tgt_domain ];
        then
            if [ $src_domain == 'laptop' -a  $tgt_domain == 'device' ];
            then
                continue
            fi
            if [ $src_domain == 'device' -a  $tgt_domain == 'laptop' ];
            then
                continue
            fi
	        python -m torch.distributed.launch --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=12345 \
            run.py \
            --model_name "bert" \
            --output_dir "${output}/${src_domain}-${tgt_domain}"  \
            --train_file "${train_dir}/${src_domain}.train.txt" "${train_dir}/${tgt_domain}.train.txt" \
            --validation_file "${test_dir}/${tgt_domain}.validation.txt" \
            --test_file "${test_dir}/${tgt_domain}.test.txt" \
            --do_train \
            --do_predict \
            --do_eval \
            --optimizer "adamW" \
            --l2reg 0.1 \
            --warmup 0.1 \
            --alpha $1 \
            --lr "3e-5" \
            --bert_lr "3e-5" \
            --batch_size 16 \
            --num_workers 16 \
            --num_train_epochs 3 \
            --seed 1 \
            --initializer kaiming_uniform_
        fi
    done
done