import logging
import os.path as osp
from argparse import ArgumentParser
from os import listdir, makedirs
from shutil import copy

import example.extraction as ext

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
parser = ArgumentParser(description="Extract entities")
parser.add_argument("--src", type=str, default="./data")
parser.add_argument("--dest", type=str, default="./processed/ent_tmp")
parser.add_argument("--batch-size", type=int, default=512)
args = parser.parse_args()
src = args.src
dest = args.dest
batch_size = args.batch_size

logging.info("Extracting entities for training files.")
assert osp.exists(src), f"{src} path not exists"
makedirs(dest, exist_ok=True)
for file in listdir(src):
    input_file = osp.join(src, file)
    output_file = osp.join(dest, file)
    if "train" not in input_file:
        logger.info("copy %s dataset into %s", file, dest)
        copy(input_file, dest)
        continue
    logger.info("Extracting entities for dataset %s", file)
    ext.run(
        ext.parser.parse_args([
            "--dataset", input_file, "--output-file", output_file, "--batch-size",
            str(batch_size)
        ]))