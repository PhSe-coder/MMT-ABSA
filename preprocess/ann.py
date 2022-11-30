import logging
import os
import os.path as osp
from argparse import ArgumentParser
from glob import glob
from os import listdir, makedirs
from shutil import copy

from transformers import BertTokenizer

import example.annotation as ann

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
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
    if "train" not in input_file:
        logger.info("copy %s dataset into %s", file, dest)
        copy(input_file, dest)
        continue
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
        for file in glob(osp.join(dest, "laptop.train.txt")):
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
                        pos_tag_dict[f"I{pos[1:]}"] = pos_tag_dict.get(f"I{pos[1:]}", 0) + len(
                            res[1:])
                        deprel_tag_dict[f"B{deprel[1:]}"] = deprel_tag_dict.get(
                            f"B{deprel[1:]}", 0) + 1
                        deprel_tag_dict[f"I{deprel[1:]}"] = deprel_tag_dict.get(
                            f"I{deprel[1:]}", 0) + len(res[1:])
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