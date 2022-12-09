#!/bin/bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export NO_PROXY=tagme.d4science.org
export PYTHONPATH=$PYTHONPATH:$(pwd)
FILE=./wikidata5m_entity.txt
if [[ ! -f "$FILE" ]]; then
    wget -O /tmp/wikidata5m_alias.tar.gz https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1
    tar -zxvf /tmp/wikidata5m_alias.tar.gz -C ./
fi
python ./preprocess/get_entity.py --src ./data
# python ./preprocess/ent_replace.py --src ./data --dest ./processed/ent_tmp --max-workers 128
sleep 5s
python ./preprocess/split.py --src ./data --dest ./processed/tmp
python ./preprocess/ann.py --src ./processed/tmp --dest ./processed/ann_tmp
python ./preprocess/dp.py --src ./processed/ann_tmp --dest ./processed/dp_tmp
sleep 5s
python ./preprocess/augment.py --src ./processed/dp_tmp --dest ./processed/ag_tmp --alpha-sr 0.3 --alpha-ri 0.1 --alpha-rs 0.1
python ./preprocess/final.py --src ./processed/ag_tmp --dest ./processed/dataset