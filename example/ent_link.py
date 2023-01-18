import logging
import threading
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Empty, Queue
import os
from typing import List
from func_timeout import func_set_timeout
from retry import retry
from tqdm import tqdm
from mmt.double_propagation.stanza_annotation import annotation_plus
from utils.tag_utils import Annotate

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

tagging_schemas = ['t', 'bio']
ready = False
parser = ArgumentParser(description='Extract and replace entities fot a dataset')
parser.add_argument("--sep", default="***", type=str, help="data item seperator")
parser.add_argument("--dataset", required=True, help="Dataset to be extracted")
parser.add_argument("--output-file",
                    required=True,
                    help="Path where the extraction result should to be saved")
parser.add_argument("--batch-size", type=int, default=32, help="extraction batch size")
parser.add_argument("--max-workers",
                    type=int,
                    default=128,
                    help="number of workers used to send requests")


class Producer(threading.Thread):

    def __init__(self, args, name: str, queue: Queue, sep="***"):
        threading.Thread.__init__(self, name=name)
        self.queue = queue
        self.args = args
        self.sep = sep
        # file line count
        self.count = sum(1 for _ in open(self.args.dataset, "rb"))

    @retry(tries=5)  # retry when any exception occurs
    @func_set_timeout(30)  # set timeout for request
    def process(self, index: int, sentence: str):
        res_dict = Annotate(sentence, theta=0.05)
        return {index: res_dict}

    def put(self, text_list: List[str], rest_list: List[str]):
        results = {}
        with ThreadPoolExecutor(max_workers=self.args.max_workers) as t:
            for future in tqdm(as_completed(
                [t.submit(self.process, idx, sentence) for idx, sentence in enumerate(text_list)]),
                               total=len(text_list),
                               disable=True):  # disable the bar
                results.update(future.result())
        assert len(results) == len(text_list), len(results)
        for i, text in enumerate(text_list):
            self.queue.put((text, rest_list[i], results[i]))

    def run(self):
        with open(self.args.dataset, "r") as f:
            text_list = []
            rest_list = []
            for idx, line in tqdm(enumerate(f), total=self.count, desc=f"{self.args.dataset}"):
                try:
                    text, rest = line.split(self.sep, maxsplit=1)
                except ValueError:
                    logger.warn("ValueError occurs in line %d", idx)
                    text = line.split(self.sep, maxsplit=1)
                    rest = ''
                text_list.append(text)
                rest_list.append(rest)
                if (idx + 1) % self.args.batch_size == 0:
                    self.put(text_list, rest_list)
                    text_list.clear()
                    rest_list.clear()
            if len(text_list) != 0:
                self.put(text_list, rest_list)
                text_list.clear()
                rest_list.clear()
        global ready
        ready = True
        print("%s finished!" % self.getName())


class Consumer(threading.Thread):

    def __init__(self, args, name: str, queue: Queue, sep="***"):
        threading.Thread.__init__(self, name=name)
        self.data = queue
        self.args = args
        self.sep = sep
        self.count = sum(1 for _ in open(self.args.dataset, "rb"))

    def run(self):
        global ready
        os.makedirs(os.path.dirname(self.args.output_file), exist_ok=True)

        with open(self.args.output_file, "w") as f:
            res_dict = None
            t = tqdm(total=self.count, desc='consumer')
            while True:
                try:
                    text, rest, res_dict = self.data.get(timeout=5)
                    if res_dict:
                        anns = annotation_plus([res_dict[k][2] for k in res_dict])
                        for i, k in enumerate(res_dict):
                            res_dict[k] += (tuple(word.xpos for word in anns[i].words),)
                    f.write(self.sep.join([text, rest.strip(), str(res_dict) + '\n']))
                    t.update(1)
                except Empty:
                    if ready:
                        ready = False
                        break
            t.close()
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
            "--dataset", "./data/laptop.train.txt", "--output-file",
            "processed/ent_link/laptop.train.txt", "--batch-size", "512"
        ]))
