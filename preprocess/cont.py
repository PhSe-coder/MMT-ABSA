import logging
import os.path as osp
from argparse import ArgumentParser
from os import listdir, makedirs
from shutil import copy

import example.contrast as cont

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
parser = ArgumentParser(description="Generate contrast data")
parser.add_argument("--src", type=str, default="./processed/ag_tmp")
parser.add_argument("--dest", type=str, default="./processed/cont_tmp")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--contrast-words-save-dir",
                    type=str,
                    default="./cont",
                    help="directory saving the contrast words of aspects")
args = parser.parse_args()
src = args.src
dest = args.dest
batch_size = args.batch_size
logging.info("Generating contrast data.")
assert osp.exists(src), f"{src} path not exists"
makedirs(dest, exist_ok=True)
for file in listdir(src):
    input_file = osp.join(src, file)
    output_file = osp.join(dest, file)
    if "train" not in input_file:
        logger.info("copy %s dataset into %s", file, dest)
        copy(input_file, dest)
        continue
    logger.info("process dataset %s with contrast data", file)
    cont.run(
        cont.parser.parse_args([
            "--dataset", input_file, "--output-file", output_file, "--batch-size",
            str(batch_size)
        ]))