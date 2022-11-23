#!/bin/bash
domains=('rest' 'service' 'laptop' 'device')

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=0
output='./out/bert_base/'
train_dir='./data'
val_dir='./processed/dataset'
test_dir='./processed/dataset'
for tar_domain in ${domains[@]};
do
    for src_domain in  ${domains[@]};
    do
        if [ $src_domain != $tar_domain ];
        then
            if [ $src_domain == 'laptop' -a  $tar_domain == 'device' ];
            then
                continue
            fi
            if [ $src_domain == 'device' -a  $tar_domain == 'laptop' ];
            then
                continue
            fi
	        python run.py \
                -m torch.distributed.launch \
                --nproc_per_node=1 \
                --local_rank 0 \
                --model_name "bert" \
                --output_dir "${output}${src_domain}-${tar_domain}"  \
                --train_file "${train_dir}/${src_domain}.train.txt" \
                --test_file "${test_dir}/${tar_domain}.test.txt" \
                --validation_file "${val_dir}/${tar_domain}.validation.txt" \
                --do_train \
                --do_predict \
                --do_eval \
                --device "cuda:0" \
                --optimizer "rmsprop" \
                --lr "1e-3" \
                --bert_lr "3e-5" \
                --batch_size 16 \
                --num_train_epochs 3
        fi
    done
done