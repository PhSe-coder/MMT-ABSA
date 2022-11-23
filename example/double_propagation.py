import logging
import threading
import time
from argparse import ArgumentParser
from queue import Empty, Queue
from typing import List

import nltk
from nltk.corpus import opinion_lexicon
from tqdm import tqdm

from mmt.double_propagation import *

logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
nltk.download('opinion_lexicon')
nltk.download('omw-1.4')
step = 0
tagging_schemas = ['t', 'bio']

parser = ArgumentParser(description='Annotate a absa dataset by Double Propagation Algorithm')
parser.add_argument("--dataset", required=True, help="Dataset to be annotated")
parser.add_argument("--output-file", required=True, help="Path where the annotation result should to be saved")
parser.add_argument("--epoch-nums", type=int, default=3, help="iteration numbers of the algorithm")
parser.add_argument("--batch-size", type=int, default=32, help="annotation batch size")
parser.add_argument("--opinion-file", type=str, default='./data/Lessico-Opinion.txt', help="opinion file path")
parser.add_argument("--target-file", type=str, default='./data/Lessico-Target.txt', help="target file path")
parser.add_argument("--schema", type=str, default="t", choices=tagging_schemas, help="annotaion schema for each tag")

class Producer(threading.Thread):
    def __init__(self, args, name: str, queue: Queue, sep="***"):
        threading.Thread.__init__(self, name=name)
        self.queue = queue
        self.args = args
        self.sep = sep
        # file line count
        self.count = sum(1 for _ in open(self.args.dataset, "rb"))

    def put(self, text_list: List[str], rest_list: List[str]):
        sentences = annotation_plus(text_list)
        for i, sentence in enumerate(sentences):
            self.queue.put((sentence, rest_list[i]))

    def run(self):
        global step
        for i in range(self.args.epoch_nums):
            with open(self.args.dataset, "r") as f:
                text_list = []
                rest_list = []
                for idx, line in tqdm(enumerate(f), total=self.count, desc="epoch {}".format(i)):
                    try:
                        text, rest = line.split(self.sep, maxsplit=1)
                    except ValueError:
                        logger.warn("ValueError occurs in line %d", idx)
                        text = line.split(self.sep, maxsplit=1)
                        rest = ''
                    text_list.append(text)
                    rest_list.append(rest)
                    if idx % self.args.batch_size == 0:
                        self.put(text_list, rest_list)
                        text_list.clear()
                        rest_list.clear()
                if len(text_list) != 0:
                    self.put(text_list, rest_list)
                    text_list.clear()
                    rest_list.clear()
            time.sleep(3)
            step = i + 1
        print("%s finished!" % self.getName())


class Consumer(threading.Thread):
    def __init__(self, args, name: str, queue: Queue, opinion_file: str, target_file: str, sep="***"):
        threading.Thread.__init__(self,name=name)
        self.data = queue
        self.args = args
        self.sep = sep
        self.positive_words: set[str] = set(opinion_lexicon.positive())
        self.negtive_words: set[str] = set(opinion_lexicon.negative())
        with open(opinion_file) as f:
            ops = f.read().splitlines()
        with open(target_file) as f:
            tars = f.read().splitlines()
        self.target_set = set(tars)
        self.opinion_set = set(ops)

    def run(self):
        global step
        with open(self.args.output_file, "w") as f:
            while True:
                try:
                    sentence, rest = self.data.get(timeout=8)
                    rule = Rule(sentence, self.positive_words, self.negtive_words)
                    tar_dict, _ = rule.propagation(self.target_set, self.opinion_set)
                    if step != self.args.epoch_nums - 1:
                        continue
                    text = sentence.text
                    tar_senti = sorted(tar_dict.items(), key=lambda d:d[0][0])
                    labels: List[str] = []
                    cur = 0
                    i = 0
                    while i in range(len(text.split())):
                        i += 1
                        try:
                            if cur >= len(tar_senti):
                                labels.append('O')
                                continue
                            if i < tar_senti[cur][0][0]:
                                labels.append('O')
                            elif i == tar_senti[cur][0][0]:
                                if self.args.schema == 't':
                                    labels.append("T-"+tar_senti[cur][1])
                                elif self.args.schema == 'bio':
                                    labels.append("B-"+tar_senti[cur][1])
                                else:
                                    raise ValueError("tagging schema must in {}. Got {} instead."
                                                     .format(tagging_schemas, self.args.schema))
                                if len(tar_senti[cur][0]) == 1:
                                    cur += 1
                            elif i > tar_senti[cur][0][0] and i < tar_senti[cur][0][1]:
                                if self.args.schema == 't':
                                    labels.append("T-"+tar_senti[cur][1])
                                elif self.args.schema == 'bio':
                                    labels.append("I-"+tar_senti[cur][1])
                                else:
                                    raise ValueError("tagging schema must in {}. Got {} instead."
                                                     .format(tagging_schemas, self.args.schema))
                            elif i == tar_senti[cur][0][1]:
                                if self.args.schema == 't':
                                    labels.append("T-"+tar_senti[cur][1])
                                elif self.args.schema == 'bio':
                                    labels.append("I-"+tar_senti[cur][1])
                                else:
                                    raise ValueError("tagging schema must in {}. Got {} instead."
                                                     .format(tagging_schemas, self.args.schema))
                                cur += 1
                        except IndexError as e:
                            # process overlapped aspect terms
                            cur += 1
                            i -= 1
                            logger.debug(f"{text} has OVERLAPPED aspect terms")
                    assert len(text.split()) == len(labels)
                    if rest == '':
                        f.write(f"{text}{self.sep}{' '.join(labels)}\n")
                    else:
                        f.write(f"{text}{self.sep}{rest.strip()}{self.sep}{' '.join(labels)}\n")
                except Empty:
                    break
        print("%s finished!" % self.getName())

def run(args):
    queue = Queue()
    producer = Producer(args, 'Producer', queue)
    consumer = Consumer(args, 'Consumer', queue, args.opinion_file, args.target_file)
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()
    print('All threads finished!')

if __name__ == '__main__':
    run(parser.parse_args())