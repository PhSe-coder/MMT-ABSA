# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import argparse
from os.path import basename, dirname, join

from tqdm import tqdm

from mmt.eda import Word, eda

ap = argparse.ArgumentParser()
ap.add_argument("--sep", default="***", type=str, help="data item seperator")
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug",
                type=int,
                default=9,
                help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr",
                type=float,
                default=0.1,
                help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri",
                type=float,
                default=0.1,
                help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs",
                type=float,
                default=0.1,
                help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd",
                type=float,
                default=0.1,
                help="percent of words in each sentence to be deleted")


# generate more data with standard augmentation
def gen_eda(args):
    if args.alpha_sr == args.alpha_ri == args.alpha_rs == args.alpha_rd == 0:
        ap.error('At least one alpha should be greater than zero')
    output_file = args.output if args.output else join(dirname(args.input), 'eda_' +
                                                       basename(args.input))
    writer = open(output_file, 'w')
    count = sum(1 for _ in open(args.input, 'rb'))
    with open(args.input, 'r') as dataset, open(output_file, 'w') as writer:
        for _, line in tqdm(enumerate(dataset), total=count):
            items = line.strip().split(args.sep)
            if len(items[0].split(" ")) == 1:
                continue
            words = tuple(Word(*item) for item in zip(*(item.split() for item in items)))
            aug_sentences = eda(words, args.alpha_sr, args.alpha_ri, args.alpha_rs, args.alpha_rd,
                                args.num_aug)
            writer.writelines(sentence + '\n' for sentence in aug_sentences)
    print("generated augmented sentences with eda for " + args.input + " to " + output_file +
          " with num_aug=" + str(args.num_aug))


if __name__ == "__main__":
    # generate augmented sentences and output into a new file
    gen_eda(ap.parse_args())