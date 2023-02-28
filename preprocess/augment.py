import logging
import os.path as osp
from argparse import ArgumentParser
from os import listdir, makedirs
from shutil import copy
import random
import example.augment as ag

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
parser = ArgumentParser(description="Data augment")
parser.add_argument("--src", type=str, default="./processed/dp_tmp")
parser.add_argument("--dest", type=str, default="./processed/ag_tmp")
parser.add_argument("--num",
                    type=int,
                    default=4,
                    help="number of augmented sentences per original sentence")
parser.add_argument("--alpha-sr",
                    type=float,
                    default=0.1,
                    help="percent of words in each sentence to be replaced by synonyms")
parser.add_argument("--alpha-ri",
                    type=float,
                    default=0,
                    help="percent of words in each sentence to be inserted")
parser.add_argument("--alpha-rs",
                    type=float,
                    default=0,
                    help="percent of words in each sentence to be swapped")
parser.add_argument("--alpha-rd",
                    type=float,
                    default=0,
                    help="percent of words in each sentence to be deleted")
parser.add_argument("--seed",
                    type=int,
                    default=1)
args = parser.parse_args()
src = args.src
dest = args.dest
num_aug = args.num
# assert num_aug % 2 == 0
alpha_sr = args.alpha_sr
alpha_rd = args.alpha_rd
alpha_ri = args.alpha_ri
alpha_rs = args.alpha_rs
random.seed(args.seed)
makedirs(dest, exist_ok=True)
for file in listdir(src):
    input_file = osp.join(src, file)
    output_file = osp.join(dest, file)
    if "train" not in input_file:
        logger.info("copy %s dataset into %s", file, dest)
        copy(input_file, dest)
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
logging.info("Augment Ready.")