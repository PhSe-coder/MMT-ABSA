import logging
import threading
import time
from argparse import ArgumentParser
from queue import Empty, Queue
from typing import List
from random import choice
from tqdm import tqdm
from wikipedia import search
from retry import retry

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = ArgumentParser(description='Generate a absa dataset with contrast aspects')
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

    @retry(tries=10, delay=3)
    def _search(self, words, aspect_idx, offset):
        return search(' '.join(words[aspect_idx[offset]:aspect_idx[-1] + 1]), results=30)

    def put(self, text_list: List[str], label_list: List[str]):
        for i, text in enumerate(text_list):
            labels = label_list[i].strip()
            words = text.split(' ')
            if all(label == 'O' for label in labels.split(" ")):
                contrast_text = text
                contrast_labels = labels
            else:
                aspects = []
                aspect = []
                labels = labels.split(" ")
                for idx, label in enumerate(labels):
                    if label != 'O':
                        aspect.append(idx)
                    elif aspect:
                        aspects.append(aspect.copy())
                        aspect.clear()
                    else:
                        pass
                if aspect:
                    aspects.append(aspect.copy())
                    aspect.clear()
                offset = -1
                aspect_idx = choice(aspects)
                result = []
                while result == []:
                    offset += 1
                    if offset >= len(aspect_idx):
                        aspects.remove(aspect_idx)
                        if not aspects:
                            break
                        aspect_idx = choice(aspects)
                        offset = 0
                    try:
                        result = self._search(words, aspect_idx, offset)
                    except Exception as e:
                        aspects = []
                        break
                    result = [
                        res for res in result if len(res.split(' ')) == len(aspect_idx) - offset
                    ]
                if aspects:
                    result = words[aspect_idx[0]:aspect_idx[offset]] + choice(result).split(' ')
                    contrast_text = words[:aspect_idx[0]] + result + words[aspect_idx[-1] + 1:]
                    contrast_labels = labels[:aspect_idx[0]] + [labels[aspect_idx[-1]]] * len(
                        result) + labels[aspect_idx[-1] + 1:]
                    assert len(contrast_text) == len(contrast_labels)
                    contrast_text = ' '.join(contrast_text)
                    contrast_labels = ' '.join(contrast_labels)
                else:
                    contrast_text = text
                    contrast_labels = label_list[i].strip()
            assert len(contrast_text.split(" ")) == len(text.split(" "))
            assert contrast_labels.split(" ") == label_list[i].strip().split(
                " "), f"{contrast_text} {text} {contrast_labels} {label_list[i].strip()}"
            self.queue.put((contrast_text, contrast_labels))

    def run(self):
        with open(self.args.dataset, "r") as f:
            text_list = []
            label_list = []
            for idx, line in tqdm(enumerate(f),
                                  total=self.count,
                                  desc="{}".format(self.args.dataset)):
                try:
                    text, label = line.split(self.sep, maxsplit=1)
                except ValueError:
                    logger.warn("ValueError occurs in line %d", idx)
                    text = line.split(self.sep, maxsplit=1)
                    label = ''
                text_list.append(text)
                label_list.append(label)
                if idx % self.args.batch_size == 0:
                    self.put(text_list, label_list)
                    text_list.clear()
                    label_list.clear()
            if len(text_list) != 0:
                self.put(text_list, label_list)
                text_list.clear()
                label_list.clear()
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
                    text, labels = self.data.get(timeout=120)
                    if len(text.split(" ")) == 1:
                        continue
                    f.write(f"{self.sep.join([text, labels.strip()])}\n")
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
            "--dataset", "./processed/ag_tmp/rest.train.txt", "--output-file",
            "./processed/cont/rest.train.txt"
        ]))
