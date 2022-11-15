import os.path as osp
from glob import glob

from torch.utils.data import random_split

__all__ = ['split']

def split(data_dir: str,output_dir: str, pattern: str="*.train.txt"):
    if "train" not in pattern:
        raise ValueError("parameter `pattern` must contain literal string 'train'")
    for file in glob(osp.join(data_dir, pattern)):
        with open(file, "r") as f:
            data = f.readlines()
        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        train_dataset, test_dataset = random_split(data, [train_size, test_size])
        with open(osp.join(output_dir, osp.split(file)[1]), "w") as f:
            f.writelines(train_dataset)
        with open(osp.join(output_dir, osp.split(file)[1].replace("train", "validation")), "w") as f:
            f.writelines(test_dataset)