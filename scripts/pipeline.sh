#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=0
output='/root/autodl-tmp/out/mmt'
tar_train_dir='./processed/dataset'
val_dir='./processed/dataset'
test_dir='./processed/dataset'
src_out='./out/bert'
src_domain=$1
tar_domain=$2

./scripts/pretrain-src.sh ${src_out} ${src_domain} ${tar_domain} 2 xavier_uniform_
./scripts/pretrain-src.sh ${src_out} ${src_domain} ${tar_domain} 42 xavier_uniform_
list=`ls ${src_out}/${src_domain}-${tar_domain}/*.pt`
c=0
for file in $list
do
  filelist[$c]=$file
  ((c++))
done
init_1=${filelist[0]}
init_2=${filelist[1]}
echo $init_1
echo $init_2
python run.py \
-m torch.distributed.launch \
--nproc_per_node=1 \
--local_rank 0 \
--model_name "mmt" \
--output_dir "${output}/${src_domain}-${tar_domain}"  \
--train_file "${tar_train_dir}/${tar_domain}.train.txt" \
--test_file "${test_dir}/${tar_domain}.test.txt" \
--validation_file "${val_dir}/${tar_domain}.validation.txt" \
--do_train \
--do_predict \
--do_eval \
--device "cuda:0" \
--optimizer "rmsprop" \
--lr "1e-3" \
--bert_lr "5e-5" \
--seed 42 \
--batch_size 16 \
--num_train_epochs 3 \
--init_1 ${init_1} \
--init_2 ${init_2}