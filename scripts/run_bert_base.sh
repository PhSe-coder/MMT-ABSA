#!/bin/bash
domains=('rest' 'service' 'laptop' 'device')

export CUDA_VISIBLE_DEVICES=0,1
export TRANSFORMERS_OFFLINE=1
output='./out/bert_base/'
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
                --output_dir "${output}${src_domain}-${tar_domain}"  \
                --train_file "./data/${src_domain}.train.txt" \
                --test_file "./dataset/${tar_domain}.test.txt" \
                --do_train \
                --do_predict \
                --device "cuda:0" \
                --optimizer "bertAdam" \
                --lr "3e-5" \
                --batch_size 16 \
                --num_train_epochs 3
                --do_eval \
                --validation_file "./dataset/${tar_domain}.validation.txt" \
        fi
    done
done