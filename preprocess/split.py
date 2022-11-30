import logging
import os.path as osp
from argparse import ArgumentParser
from glob import glob
from os import makedirs
from shutil import copy

import torch
from torch.utils.data import random_split

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
parser = ArgumentParser(description="Data split")
parser.add_argument("--src", type=str, default="./data")
parser.add_argument("--dest", type=str, default="./processed/tmp")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--split-rate", type=float, default=0.8, help="split rate of train set")
args = parser.parse_args()
src = args.src
dest = args.dest


def split(file: str, output_dir: str, rate=0.8, seed=42):
    makedirs(output_dir, exist_ok=True)
    with open(file, "r") as f:
        data = f.readlines()
    train_size = int(rate * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = random_split(data, [train_size, test_size],
                                               torch.Generator().manual_seed(seed))
    with open(osp.join(output_dir, osp.split(file)[1]), "w") as f:
        f.writelines(train_dataset)
    with open(osp.join(output_dir, osp.split(file)[1].replace("train", "validation")), "w") as f:
        f.writelines(test_dataset)


logger.info("split train set into train/validation dataset, save in %s", dest)
for file in glob(osp.join(src, "*.train.txt")):
    split(file, dest, args.split_rate, args.seed)
logger.info("copy test dataset into %s", dest)
for file in glob(osp.join(src, "*.test.txt")):
    copy(file, dest)