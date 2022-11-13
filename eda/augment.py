# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou
from eda import eda, Word

#arguments to be parsed from command line
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", type=int, default=9, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", type=float, default=0.1, help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", type=float, default=0.1, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", type=float, default=0.1, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", type=float, default=0.1, help="percent of words in each sentence to be deleted")
args = ap.parse_args()

# the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))

if args.alpha_sr == args.alpha_ri == args.alpha_rs == args.alpha_rd == 0:
     ap.error('At least one alpha should be greater than zero')

# generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):

    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').read().splitlines()

    for _, line in enumerate(lines):
        sentence, labels = line.split("***")[0:2]
        words = tuple(Word(token, label) for token, label in zip(sentence.split(), labels.split()))
        aug_sentences = eda(words, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            writer.write(aug_sentence + '\n')

    writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(num_aug))

if __name__ == "__main__":
    # generate augmented sentences and output into a new file
    gen_eda(args.input, 
            output,
            num_aug=args.num_aug,
            alpha_sr=args.alpha_sr, 
            alpha_ri=args.alpha_ri, 
            alpha_rs=args.alpha_rs, 
            alpha_rd=args.alpha_rd)