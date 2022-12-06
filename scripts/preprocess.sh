#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

python ./preprocess/ent_link.py --src ./data --dest ./processed/ent_tmp
python ./preprocess/split.py --src ./processed/ent_tmp --dest ./processed/tmp
python ./preprocess/ann.py --src ./processed/tmp --dest ./processed/ann_tmp
python ./preprocess/dp.py --src ./processed/ann_tmp --dest ./processed/dp_tmp
python ./preprocess/augment.py --src ./processed/dp_tmp --dest ./processed/ag_tmp --alpha-sr 0.3 --alpha-ri 0.1 --alpha-rs 0.1
python ./preprocess/final.py --src ./processed/ag_tmp --dest ./processed/dataset