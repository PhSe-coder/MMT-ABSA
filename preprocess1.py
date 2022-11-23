import logging
import os.path as osp
from argparse import ArgumentParser
from glob import glob
from os import listdir, makedirs
from shutil import copy

import example.augment as ag

import example.annotation as ann
from mmt.utils import split

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
parser = ArgumentParser(description="Data preprocess")
parser.add_argument("--src", type=str, default="./data")
parser.add_argument("--dst", type=str, default="./processed1/dataset")
args = parser.parse_args()
tmp_dst_1 = "./processed1/tmp"
tmp_dst_2 = "./processed1/ann_tmp"
tmp_dst_3 = "./processed1/ag_tmp"
pos_file = './processed1/pos.txt'
deprel_file = './processed1/deprel.txt'
ann_batch_size = 512
num_aug = 4
assert num_aug % 2 == 0
alpha_sr = 0.05
alpha_rd = 0
alpha_ri = 0
alpha_rs = 0
logger.info("split train set into train/validation dataset, save in %s", tmp_dst_1)
split(args.src, tmp_dst_1)
logger.info("copy test dataset into %s", tmp_dst_1)
for file in glob("./data/*.test.txt"):
    copy(file, tmp_dst_1)

makedirs(tmp_dst_2, exist_ok=True)
for file in listdir(tmp_dst_1):
    input_file = osp.join(tmp_dst_1, file)
    output_file = osp.join(tmp_dst_2, file)
    if "train" not in input_file:
        logger.info("copy %s dataset into %s", file, tmp_dst_2)
        copy(input_file, tmp_dst_2)
        continue
    logger.info("process dataset %s with double propagation algorithm", file)
    ann.run(
        ann.parser.parse_args([
            "--dataset", input_file, "--output-file", output_file, "--batch-size",
            str(ann_batch_size)
        ]))


with open(pos_file, "w") as pos_writer, open(deprel_file, "w") as deprel_writer:
    pos_tag_list = []
    deprel_tag_list = []
    for file in listdir(tmp_dst_2):
        with open(file, "r") as f:
            for line in f:
                text, gold_labels, pos_tag_labels, delrel_tag_labels = line.strip().split("***")
                pos_tag_list.extend([
                    _ for item in set(pos_tag_labels.split())
                    for _ in [f"B-{item[1:]}", f"I-{item[1:]}"] if _ not in pos_tag_list
                ])
                deprel_tag_list.extend([
                    _ for item in set(delrel_tag_labels.split())
                    for _ in [f"B-{item[1:]}", f"I-{item[1:]}"] if _ not in deprel_tag_list
                ])
    pos_writer.writelines([tag + '\n' for tag in pos_tag_list])
    deprel_writer.writelines([tag + '\n' for tag in deprel_tag_list])

makedirs(tmp_dst_3, exist_ok=True)
for file in listdir(tmp_dst_2):
    input_file = osp.join(tmp_dst_2, file)
    output_file = osp.join(tmp_dst_3, file)
    if "train" not in input_file:
        logger.info("copy %s dataset into %s", file, tmp_dst_3)
        copy(input_file, tmp_dst_3)
        continue
    logger.info("augment dataset %s", file)
    ag.gen_eda(
        ag.ap.parse_args([
            "--input", input_file, "--output", output_file, "--num_aug",
            str(num_aug), "--alpha_sr",
            str(alpha_sr), "--alpha_rd",
            str(alpha_rd), "--alpha_ri",
            str(alpha_ri), "--alpha_rs",
            str(alpha_rs)
        ]))
makedirs(args.dst, exist_ok=True)
for file in listdir(tmp_dst_3):
    logger.info("process dataset %s", file)
    input_file = osp.join(tmp_dst_3, file)
    output_file = osp.join(args.dst, file)
    if "train" not in input_file:
        logger.info("copy %s dataset into %s", file, args.dst)
        copy(input_file, args.dst)
        continue
    with open(input_file, "r") as f, open(output_file, "w") as f1:
        line = f.readline()
        lines = []
        count = 0
        while line:
            count += 1
            lines.append(line)
            if count == 2:
                count = 0
                f1.writelines([
                    f"{left.strip()}####{right.strip()}\n"
                    for left, right in zip(lines[:1], lines[1:])
                ])
                lines.clear()
            line = f.readline()
logger.info("Process OK.")