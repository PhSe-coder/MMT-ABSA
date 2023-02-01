#!/bin/bash
list="0.019 0.018 0.017 0.016 0.015 0.014 0.013 0.012 0.011 0.01"
for i in $list;
do
    ./scripts/run_bert_base.sh $i ;
done