import os.path as osp
from glob import glob
from os import listdir
from typing import Dict, List

__all__ = ['compare']

def compare(models_dir: str):
    models = listdir(models_dir)
    data: List[Dict[str, float]] = []
    for model in models:
        model_path = osp.join(models_dir, model)
        src_tar_list = listdir(model_path)
        f1_list: List[float] = []
        for file in glob(osp.join(model_path, "**/absa_prediction.txt")):
            with open(file, "r") as f:
                line = f.readline()
            f1_list.append(float(line.split()[-1]))
        src_tar_list.append("AVG")
        f1_list.append(sum(f1_list)/len(f1_list))
        data.append(dict(zip(src_tar_list, f1_list)))
    return data