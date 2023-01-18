import sys
import os
import random
random.seed(42)
src_domain_train_file = sys.argv[1]
tgt_domain_train_file = sys.argv[2]
sep = "***"
output_file = sys.argv[3]
os.makedirs(os.path.dirname(output_file), exist_ok=True)
writer = open(output_file, "w")
with open(src_domain_train_file, "r") as f:
    sentences = [sep.join((line.strip(), '1\n')) for line in f]
with open(tgt_domain_train_file, "r") as f:
    sentences.extend([sep.join((line.strip(), '0\n')) for line in f])
random.shuffle(sentences)
writer.writelines(sentences)
writer.close()
