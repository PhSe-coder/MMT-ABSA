#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python preprocess.py --dst ./processed/dataset

