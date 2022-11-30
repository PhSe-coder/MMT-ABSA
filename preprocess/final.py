import logging
import os.path as osp
from argparse import ArgumentParser
from os import listdir, makedirs
from shutil import copy

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
parser = ArgumentParser(description="Data merge")
parser.add_argument("--src", type=str, default="./processed/ent_tmp")
parser.add_argument("--dest", type=str, default="./processed/dataset")
args = parser.parse_args()
src = args.src
makedirs(args.dest, exist_ok=True)
for file in listdir(src):
    logger.info("process dataset %s", file)
    input_file = osp.join(src, file)
    output_file = osp.join(args.dest, file)
    if "train" not in input_file:
        logger.info("copy %s dataset into %s", file, args.dest)
        copy(input_file, args.dest)
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
logger.info("Ready!")