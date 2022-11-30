#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

python ./preprocess/split.py --src ./data --dest ./processed/tmp
python ./preprocess/ann.py --src ./processed/tmp --dest ./processed/ann_tmp
python ./preprocess/dp.py --src ./processed/ann_tmp --dest ./processed/dp_tmp
python ./preprocess/ent.py --src ./processed/dp_tmp --dest ./processed/ent_tmp
python ./preprocess/augment.py --src ./processed/ent_tmp --dest ./processed/ag_tmp
python ./preprocess/final.py --src ./processed/ag_tmp --dest ./processed/dataset