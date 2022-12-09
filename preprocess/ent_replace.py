import logging
import os.path as osp
from argparse import ArgumentParser
from os import listdir, makedirs
from shutil import copy

import example.extraction as ext

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
parser = ArgumentParser(description="Replace entities")
parser.add_argument("--src", type=str, default="./data")
parser.add_argument("--dest", type=str, default="./processed/ent_tmp")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--vector-dir", type=str, default="./processed/vectors")
parser.add_argument("--max-workers",
                    type=int,
                    default=16,
                    help="number of workers used to send requests")
args = parser.parse_args()
src = args.src
dest = args.dest
batch_size = args.batch_size
vector_dir = args.vector_dir
logging.info("Extracting entities for training files.")
assert osp.exists(src), f"{src} path not exists"
makedirs(dest, exist_ok=True)
for file in listdir(src):
    domain = file.split('.')[0]
    input_file = osp.join(src, file)
    output_file = osp.join(dest, file)
    logger.info("Extracting entities for dataset %s", file)
    ext.run(
        ext.parser.parse_args([
            "--dataset", input_file, "--output-file", output_file, "--batch-size",
            str(batch_size), "--mean-vec",
            osp.join(vector_dir, domain + "_mean_vec.npy"), "--max-workers",
            str(args.max_workers)
        ]))