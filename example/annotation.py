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

parser = ArgumentParser(description='Annotate a absa dataset with dep and pos tag')
parser.add_argument("--sep", default="***", type=str, help="data item seperator")
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
            for idx, line in tqdm(enumerate(f),
                                  total=self.count,
                                  desc="{}".format(self.args.dataset)):
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

    def __init__(self, args, name: str, queue: Queue, sep="***"):
        threading.Thread.__init__(self, name=name)
        self.data = queue
        self.args = args
        self.sep = sep

    def run(self):
        with open(self.args.output_file, "w") as f:
            while True:
                try:
                    sentence, rest = self.data.get(timeout=15)
                    text: str = sentence.text
                    assert self.sep not in rest
                    gold_labels: List[str] = rest.split()
                    doc: List[dict] = sentence.to_dict()
                    assert len(gold_labels) == len(doc)
                    pos_labels, deprel_labels, heads = [], [], []
                    for word, _ in zip(doc, gold_labels):
                        pos, deprel, head = word['xpos'], word['deprel'], word['head']
                        pos_labels.append(pos)
                        deprel_labels.append(deprel)
                        heads.append(str(head))
                    assert len(gold_labels) == len(deprel_labels)
                    assert len(gold_labels) == len(pos_labels)
                    assert len(gold_labels) == len(heads)
                    if len(text.split(" ")) == 1:
                        continue
                    f.write(
                        f"{self.sep.join([text, rest.strip(), ' '.join(pos_labels), ' '.join(deprel_labels), ' '.join(heads)])}\n"
                    )
                except Empty:
                    break
        print("%s finished!" % self.getName())


def run(args):
    queue = Queue()
    producer = Producer(args, 'Producer', queue, args.sep)
    consumer = Consumer(args, 'Consumer', queue, args.sep)
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()
    print('All threads finished!')


if __name__ == '__main__':
    run(
        parser.parse_args([
            "--dataset", "./processed/tmp/device.train.txt", "--output-file",
            "./processed/ann_tmp/device.train.txt"
        ]))
