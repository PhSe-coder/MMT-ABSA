from glob import glob
import os
import os.path as osp
from shutil import copy
N = -1
output_dir = f"./processed/divide/{N}"
input_dir = './processed/dataset'
os.makedirs(output_dir, exist_ok=True)
for file in glob(osp.join(input_dir, "*.train.txt")):
    lines = []
    with open(file, "r") as f:
        if N == -1:
            lines = [line for line in f]
        else:
            for line in f:
                text, others = line.split("***", maxsplit=1)
                if len(text.split(' ')) <= N:
                    lines.append('{}***{}'.format(text, others))
    with open(osp.join(output_dir, osp.basename(file)), "w") as f:
        f.writelines(lines)
for file in glob(osp.join(input_dir, "*.txt")):
    if 'train.txt' not in file:
        copy(file, output_dir)