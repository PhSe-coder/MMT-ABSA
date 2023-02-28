#!/bin/bash
# export HTTP_PROXY=http://127.0.0.1:7890
# export HTTPS_PROXY=http://127.0.0.1:7890
# export NO_PROXY=tagme.d4science.org
export PYTHONPATH=$PYTHONPATH:$(pwd)
# python ./preprocess/get_entity.py --src ./data
# # python ./preprocess/ent_replace.py --src ./data --dest ./processed/ent_tmp --max-workers 128
# python ./preprocess/split.py --src ./data --dest ./processed/tmp --split-rate 0.7
# python ./preprocess/ann.py --src ./data --dest ./processed/ann_tmp
# python ./preprocess/ent_link.py --src ./processed/ann_tmp --dest ./processed/ent_tmp
# python ./preprocess/dp.py --src ./processed/ann_tmp --dest ./processed/dp_tmp
python ./preprocess/augment.py --src ./processed/tmp --dest ./processed/dataset --alpha-sr 0.3 --num 3 --seed 1
python ./preprocess/augment.py --src ./processed/tmp --dest ./processed/cont_dataset --alpha-sr 0.3 --num 3 --seed 42
# python ./preprocess/cont.py --src ./processed/ag_tmp --dest ./processed/cont_tmp
# python ./preprocess/ann.py --src ./processed/ag_tmp --dest ./processed/dataset
# python ./preprocess/ann.py --src ./processed/cont_tmp --dest ./processed/cont_dataset

# python ./preprocess/ent_link.py --src ./processed/ag_tmp --dest ./processed/dataset
# python ./preprocess/final.py --src ./processed/ag_tmp --dest ./processed/dataset