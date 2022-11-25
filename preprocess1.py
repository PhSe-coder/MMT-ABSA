import logging
import os.path as osp
from argparse import ArgumentParser
from glob import glob
from os import listdir, makedirs
from shutil import copy

import example.augment as ag

import example.annotation as ann
from mmt.utils import split

from transformers import BertTokenizer

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
    logger.info("process dataset %s with pos,dep annotation", file)
    ann.run(
        ann.parser.parse_args([
            "--dataset", input_file, "--output-file", output_file, "--batch-size",
            str(ann_batch_size)
        ]))

with open(pos_file, "w") as pos_writer, open(deprel_file, "w") as deprel_writer:
    pos_tag_dict = {}
    deprel_tag_dict = {}
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for file in glob(osp.join(tmp_dst_2, "*.train.txt")):
        with open(file, "r") as f:
            for line in f:
                for token, label, pos, deprel in zip(
                        *[item.split() for item in line.strip().split("***")]):
                    if label == 'O':
                        continue
                    res = tokenizer.wordpiece_tokenizer.tokenize(token)
                    assert pos != 'O'
                    assert deprel != 'O'
                    pos_tag_dict[f"B{pos[1:]}"] = pos_tag_dict.get(f"B{pos[1:]}", 0) + 1
                    pos_tag_dict[f"I{pos[1:]}"] = pos_tag_dict.get(f"I{pos[1:]}", 0) + len(res[1:])
                    deprel_tag_dict[f"B{deprel[1:]}"] = deprel_tag_dict.get(f"B{deprel[1:]}", 0) + 1
                    deprel_tag_dict[f"I{deprel[1:]}"] = deprel_tag_dict.get(f"I{deprel[1:]}",
                                                                            0) + len(res[1:])
    pos_tag_num = sum(count for _, count in pos_tag_dict.items())
    deprel_tag_num = sum(count for _, count in deprel_tag_dict.items())
    thr = 0.9
    sorted_pos_tag_items = sorted([(tag, num) for tag, num in pos_tag_dict.items()],
                                  key=lambda item: item[1],
                                  reverse=True)
    for i in range(len(pos_tag_dict)):
        if sum(count for _, count in sorted_pos_tag_items[:i]) / pos_tag_num > thr:
            break
    b_unk_count = sum(num for tag, num in sorted_pos_tag_items[i:] if tag.startswith("B"))
    i_unk_count = sum(num for tag, num in sorted_pos_tag_items[i:] if tag.startswith("I"))
    valid_pos_list = sorted_pos_tag_items[:i]
    valid_pos_list.append(('B-[UNK]', b_unk_count))
    valid_pos_list.append(('I-[UNK]', i_unk_count))
    valid_pos_list = sorted(valid_pos_list, key=lambda item: item[1], reverse=True)
    pos_writer.writelines([f"{tag} {count/pos_tag_num}\n" for tag, count in valid_pos_list])
    sorted_deprel_tag_items = sorted([(tag, num) for tag, num in deprel_tag_dict.items()],
                                  key=lambda item: item[1],
                                  reverse=True)
    for i in range(len(deprel_tag_dict)):
        if sum(count for _, count in sorted_deprel_tag_items[:i]) / deprel_tag_num > thr:
            break
    b_unk_count = sum(num for tag, num in sorted_deprel_tag_items[i:] if tag.startswith("B"))
    i_unk_count = sum(num for tag, num in sorted_deprel_tag_items[i:] if tag.startswith("I"))
    valid_deprel_list = sorted_deprel_tag_items[:i]
    valid_deprel_list.append(('B-[UNK]', b_unk_count))
    valid_deprel_list.append(('I-[UNK]', i_unk_count))
    valid_deprel_list = sorted(valid_deprel_list, key=lambda item: item[1], reverse=True)
    deprel_writer.writelines(
        [f"{tag} {count/deprel_tag_num}\n" for tag, count in valid_deprel_list])

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