import logging
import os.path as osp
from argparse import ArgumentParser
from os import listdir, makedirs

import example.ent_link as ext

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
parser = ArgumentParser(description="Entity linking")
parser.add_argument("--src", type=str, default="./data")
parser.add_argument("--dest", type=str, default="./processed/ent_link")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--max-workers",
                    type=int,
                    default=128,
                    help="number of workers used to send requests")
args = parser.parse_args()
src = args.src
dest = args.dest
batch_size = args.batch_size
logging.info("Linking entities for training files.")
assert osp.exists(src), f"{src} path not exists"
makedirs(dest, exist_ok=True)
for file in listdir(src):
    if 'device.train' in file or 'laptop.train' in file:
        continue
    domain = file.split('.')[0]
    input_file = osp.join(src, file)
    output_file = osp.join(dest, file)
    logger.info("Linking entities for dataset %s", file)
    ext.run(
        ext.parser.parse_args([
            "--dataset", input_file, "--output-file", output_file, "--batch-size",
            str(batch_size), "--max-workers",
            str(args.max_workers)
        ]))