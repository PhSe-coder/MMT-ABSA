import logging
import os.path as osp
from argparse import ArgumentParser
from glob import glob
from os import listdir, makedirs
from shutil import copy

import augment as ag

import example.double_propagation as dp
from mmt.utils import split

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
parser = ArgumentParser(description="Data preprocess")
parser.add_argument("--src", type=str, default="./data")
parser.add_argument("--dst", type=str, default="./processed/dataset")
args = parser.parse_args()
tmp_dst_1 = "./processed/tmp"
tmp_dst_2 = "./processed/dp_tmp"
tmp_dst_3 = "./processed/ag_tmp"
dp_batch_size = 512
num_aug = 16
assert num_aug % 2 == 0
logger.info("split train set into train/validation dataset, save in %s", tmp_dst_1)
split(args.src, tmp_dst_1)
logger.info("copy test dataset into %s", tmp_dst_1)
for file in glob("./data/*.test.txt"):
    copy(file, tmp_dst_1)
makedirs(tmp_dst_2, exist_ok=True)
for file in listdir(tmp_dst_1):
    logger.info("process dataset %s with double propagation algorithm", file)
    dp.run(dp.parser.parse_args(["--dataset", 
                                   osp.join(tmp_dst_1, file), 
                                   "--output-file", 
                                   osp.join(tmp_dst_2, file),
                                   "--batch-size",
                                   str(dp_batch_size)]))
makedirs(tmp_dst_3, exist_ok=True)
for file in listdir(tmp_dst_2):
    logger.info("augment dataset %s", file)
    ag.gen_eda(ag.ap.parse_args(["--input",
                                 osp.join(tmp_dst_2, file),
                                 "--output",
                                 osp.join(tmp_dst_3, file),
                                 "--num_aug",
                                 str(num_aug),
                                 "--alpha_sr", "0.05",
                                 "--alpha_rd",
                                 "0.1",
                                 "--alpha_ri",
                                 "0.1",
                                 "--alpha_rs",
                                 "0.1"]))
makedirs(args.dst, exist_ok=True)
for file in listdir(tmp_dst_3):
    logger.info("process dataset %s", file)
    with open(osp.join(tmp_dst_3, file), "r") as f, open(osp.join(args.dst, file), "w") as f1:
        line = f.readline()
        lines = []
        count = 0
        while line:
            count += 1
            lines.append(line)
            if count == num_aug:
                count = 0
                f1.writelines([f"{left.strip()}####{right.strip()}\n"
                               for left, right in zip(lines[:int(num_aug/2)], lines[int(num_aug/2):])])
                lines.clear()
            line = f.readline()
logger.info("Process OK.")