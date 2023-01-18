import logging
import os
import os.path as osp
from argparse import ArgumentParser
from glob import glob
from os import listdir, makedirs

from typing import List
from transformers import BertTokenizer

import example.annotation as ann

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
parser = ArgumentParser(description="POS and dep")
parser.add_argument("--src", type=str, default="./processed/tmp")
parser.add_argument("--dest", type=str, default="./processed/ann_tmp")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--ann-save-dir",
                    type=str,
                    default="./ann",
                    help="directory saving the statistical results of pos and dep")
args = parser.parse_args()
src = args.src
dest = args.dest
ann_batch_size = args.batch_size
save_dir = args.ann_save_dir

logging.info("Generating pos and dep annotations for training files.")
assert osp.exists(src), f"{src} path not exists"
makedirs(dest, exist_ok=True)
for file in listdir(src):
    input_file = osp.join(src, file)
    output_file = osp.join(dest, file)
    logger.info("process dataset %s with pos,dep annotation", file)
    ann.run(
        ann.parser.parse_args([
            "--dataset", input_file, "--output-file", output_file, "--batch-size",
            str(ann_batch_size)
        ]))

logging.info("Analyzing annotation statistics for training files.")

assert osp.exists(dest), f"{dest} path not exists"
os.makedirs(save_dir, exist_ok=True)
for domain in ['service', 'laptop', 'device', 'rest']:
    with open(osp.join(save_dir, f"{domain}_pos.txt"),
              "w") as pos_writer, open(osp.join(save_dir, f"{domain}_deprel.txt"),
                                       "w") as deprel_writer:
        pos_tag_dict = {}
        deprel_tag_dict = {}
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        sentences: List[str] = []
        for file in glob(osp.join(dest, f"{domain}.*.txt")):
            sentences.extend(open(file, "r").read().splitlines())
        for sentence in sentences:
            for token, label, pos, deprel, head in zip(
                    *[item.split() for item in sentence.strip().split("***")]):
                # if label == 'O':
                #     continue
                pos_tag_dict[pos] = pos_tag_dict.get(pos, 0) + 1
                deprel_tag_dict[deprel] = deprel_tag_dict.get(deprel, 0) + 1
        pos_tag_num = sum(count for _, count in pos_tag_dict.items())
        valid_pos_list = sorted(pos_tag_dict.items(), key=lambda item: item[1], reverse=True)
        pos_writer.writelines([f"{tag} {count/pos_tag_num}\n" for tag, count in valid_pos_list])
        deprel_tag_num = sum(count for _, count in deprel_tag_dict.items())
        valid_deprel_list = sorted(deprel_tag_dict.items(), key=lambda item: item[1], reverse=True)
        deprel_writer.writelines(
            [f"{tag} {count/deprel_tag_num}\n" for tag, count in valid_deprel_list])
logging.info("Annotation Ready.")