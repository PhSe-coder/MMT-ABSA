#!/bin/bash
domains=('rest' 'service' 'laptop' 'device')

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=0
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
            if [ $src_domain == 'laptop' -a  $tar_domain == 'rest' ];
            then
                continue
            fi
	        ./scripts/temp.sh ${src_domain} ${tar_domain}
        fi
    done
done