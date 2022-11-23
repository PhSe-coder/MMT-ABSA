import logging
import threading
import time
from argparse import ArgumentParser
from queue import Empty, Queue
from typing import List

from tqdm import tqdm
from mmt.double_propagation import annotation_plus

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tagging_schemas = ['t', 'bio']

parser = ArgumentParser(description='Annotate a absa dataset with dep and pos tag')
parser.add_argument("--dataset", required=True, help="Dataset to be annotated")
parser.add_argument("--output-file",
                    required=True,
                    help="Path where the annotation result should to be saved")
parser.add_argument("--batch-size", type=int, default=32, help="annotation batch size")


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
        with open(self.args.dataset, "r") as f:
            text_list = []
            rest_list = []
            for idx, line in tqdm(enumerate(f), total=self.count, desc="{}".format(self.args.dataset)):
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
        print("%s finished!" % self.getName())


class Consumer(threading.Thread):

    def __init__(self,
                 args,
                 name: str,
                 queue: Queue,
                 sep="***"):
        threading.Thread.__init__(self, name=name)
        self.data = queue
        self.args = args
        self.sep = sep

    def run(self):
        with open(self.args.output_file, "w") as f:
            while True:
                try:
                    sentence, rest = self.data.get(timeout=8)
                    text: str = sentence.text
                    assert self.sep not in rest
                    gold_labels: List[str] =  rest.split()
                    doc: List[dict] = sentence.to_dict()
                    assert len(gold_labels) == len(doc)
                    pos_labels, deprel_labels = [], []
                    for word, label in zip(doc, gold_labels):
                        pos, deprel = word['xpos'], word['deprel']
                        pos_label = "T-{}".format(pos) if label != 'O' else 'O'
                        deprel_label = "T-{}".format(deprel) if label != 'O' else 'O'
                        pos_labels.append(pos_label)
                        deprel_labels.append(deprel_label)
                    assert len(gold_labels) == len(deprel_labels)
                    assert len(gold_labels) == len(pos_labels)
                    f.write(
                        f"{self.sep.join([text, rest.strip(), ' '.join(pos_labels), ' '.join(deprel_labels)])}\n"
                    )
                except Empty:
                    break
        print("%s finished!" % self.getName())


def run(args):
    queue = Queue()
    producer = Producer(args, 'Producer', queue)
    consumer = Consumer(args, 'Consumer', queue)
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()
    print('All threads finished!')


if __name__ == '__main__':
    run(parser.parse_args())