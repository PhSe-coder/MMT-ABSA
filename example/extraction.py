import logging
import threading
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Empty, Queue
from typing import List
from utils.tag_utils import Annotate
from tqdm import tqdm
from random import randint
from mmt.double_propagation import annotation_plus, SENTMENT_MAP

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tagging_schemas = ['t', 'bio']
ready = False

parser = ArgumentParser(description='Extract entities fot a dataset')
parser.add_argument("--sep", default="***", type=str, help="data item seperator")
parser.add_argument("--dataset", required=True, help="Dataset to be extracted")
parser.add_argument("--output-file",
                    required=True,
                    help="Path where the extraction result should to be saved")
parser.add_argument("--batch-size", type=int, default=32, help="extraction batch size")
parser.add_argument("--schema",
                    type=str,
                    default="t",
                    choices=tagging_schemas,
                    help="annotaion schema for each tag")


class Producer(threading.Thread):

    def __init__(self, args, name: str, queue: Queue, sep="***"):
        threading.Thread.__init__(self, name=name)
        self.queue = queue
        self.args = args
        self.sep = sep
        # file line count
        self.count = sum(1 for _ in open(self.args.dataset, "rb"))

    def process_ann(self, index: int, sentence: str):
        res_dict = Annotate(sentence, theta=0.05)
        if not res_dict:
            return {index: res_dict}
        begins = list(zip(*res_dict.keys()))[0]
        ends = list(zip(*res_dict.keys()))[1]
        for i in range(len(begins) - 1):
            assert begins[i] <= begins[i + 1]
            # preserve the longest entity
            if begins[i] == begins[i + 1] and ends[i] < ends[i + 1]:
                res_dict.pop((begins[i], ends[i]), None)
        # merge the entity with continuous index
        result = {}
        begins = list(zip(*res_dict.keys()))[0]
        ends = list(zip(*res_dict.keys()))[1]
        ent = {}
        for i in range(len(begins) - 1):
            if ends[i] + 1 == begins[i + 1]:
                ent[(begins[i], ends[i])] = res_dict.get((begins[i], ends[i]))
                ent[(begins[i + 1], ends[i + 1])] = res_dict.get((begins[i + 1], ends[i + 1]))
                continue
            if len(ent) != 0:
                keys = list(ent.keys())
                result[(keys[0][0], keys[-1][-1])] = tuple(ent.values())
                ent.clear()
            # prevent overlap
            if not any(set(range(begins[i], ends[i])) - set(range(*key)) for key in result.keys()):
                result[(begins[i], ends[i])] = res_dict.get((begins[i], ends[i]))
        if len(ent) != 0:
            ent_keys = list(ent.keys())
            result[(ent_keys[0][0], ent_keys[-1][-1])] = tuple(ent.values())
            ent.clear()
        # consider sentence with one entity
        if not result:
            result = res_dict
        # remove the overlap entites
        begins = list(zip(*result.keys()))[0]
        ends = list(zip(*result.keys()))[1]
        for i in range(len(begins) - 1):
            if ends[i] + 1 > begins[i + 1]:
                len1 = ends[i] - begins[i]
                len2 = ends[i + 1] - begins[i + 1]
                if len1 > len2:
                    result.pop((begins[i + 1], ends[i + 1]), None)
                elif len1 < len2:
                    result.pop((begins[i], ends[i]), None)
                else:
                    idx = randint(0, 1)
                    result.pop((begins[i + idx], ends[i + idx]), None)
        return {index: result}

    def put(self, text_list: List[str], rest_list: List[str], hard_label_list: List[str]):
        results = {}
        with ThreadPoolExecutor(max_workers=100) as t:
            for future in tqdm(as_completed([
                    t.submit(self.process_ann, idx, sentence)
                    for idx, sentence in enumerate(text_list)
            ]),
                               total=self.args.batch_size, disable=True): # disable the bar
                results.update(future.result())
        res = [result for _, result in sorted(results.items(), key=lambda item: item[0])]
        sentences = annotation_plus(text_list)
        for i, sentence in enumerate(sentences):
            self.queue.put((sentence, rest_list[i], hard_label_list[i], res[i]))

    def run(self):
        with open(self.args.dataset, "r") as f:
            text_list = []
            rest_list = []
            hard_label_list = []
            for idx, line in tqdm(enumerate(f), total=self.count, desc=f"{self.args.dataset}"):
                try:
                    text, rest = line.split(self.sep, maxsplit=1)
                except ValueError:
                    logger.warn("ValueError occurs in line %d", idx)
                    text = line.split(self.sep, maxsplit=1)
                    rest = ''
                text_list.append(text)
                rest_list.append(rest.rsplit(self.sep, 1)[0])
                hard_label_list.append(rest.rsplit(self.sep, 1)[-1])
                if (idx + 1) % self.args.batch_size == 0:
                    self.put(text_list, rest_list, hard_label_list)
                    text_list.clear()
                    rest_list.clear()
                    hard_label_list.clear()
            if len(text_list) != 0:
                self.put(text_list, rest_list, hard_label_list)
                text_list.clear()
                rest_list.clear()
                hard_label_list.clear()
        global ready
        ready = True


class Consumer(threading.Thread):

    def __init__(self, args, name: str, queue: Queue, sep="***"):
        threading.Thread.__init__(self, name=name)
        self.data = queue
        self.args = args
        self.sep = sep

    def run(self):
        global ready
        with open(self.args.output_file, "w") as f:
            while True:
                try:
                    sentence, rest, hard_labels, res = self.data.get(timeout=5)
                    hard_label_list = hard_labels.strip().split(' ')
                    text = sentence.text
                    tar_dict = {}
                    hard_labels_dict = {idx: label for idx, label in enumerate(hard_label_list) if label != 'O'}
                    for item in res:
                        idx_tuple = tuple(text[:idx].count(" ") for idx in item)
                        assert len(idx_tuple) == 2
                        idx_list = list(idx_tuple)
                        idx_list[-1] += 1
                        res_idx_list = [k in range(*idx_list) for k in hard_labels_dict]
                        if any(res_idx_list):
                            tar_dict[idx_tuple] = list(hard_labels_dict.items())[res_idx_list.index(True)][-1][-3:]
                        else:
                            tar_dict[idx_tuple] = SENTMENT_MAP[sentence.sentiment]
                    tar_senti = sorted(tar_dict.items(), key=lambda d: d[0][0])
                    labels: List[str] = []
                    cur = 0
                    i = 0
                    while i in range(len(text.split())):
                        try:
                            if cur >= len(tar_senti):
                                labels.append('O')
                                i += 1
                                continue
                            if i < tar_senti[cur][0][0]:
                                labels.append('O')
                            elif i == tar_senti[cur][0][0]:
                                if self.args.schema == 't':
                                    tag = "T-" + tar_senti[cur][1]
                                    labels.append(tag)
                                    if hard_label_list[i] == 'O':
                                        hard_label_list[i] = tag
                                elif self.args.schema == 'bio':
                                    tag = "B-" + tar_senti[cur][1]
                                    labels.append(tag)
                                    if hard_label_list[i] == 'O':
                                        hard_label_list[i] = tag
                                else:
                                    raise ValueError(
                                        "tagging schema must in {}. Got {} instead.".format(
                                            tagging_schemas, self.args.schema))
                                if len(tar_senti[cur][0]) == 1 or tar_senti[cur][0][0] == tar_senti[cur][0][1]:
                                    cur += 1
                            elif i > tar_senti[cur][0][0] and i < tar_senti[cur][0][1]:
                                if self.args.schema == 't':
                                    tag = "T-" + tar_senti[cur][1]
                                    labels.append(tag)
                                    if hard_label_list[i] == 'O':
                                        hard_label_list[i] = tag
                                elif self.args.schema == 'bio':
                                    tag = "I-" + tar_senti[cur][1]
                                    labels.append(tag)
                                    if hard_label_list[i] == 'O':
                                        hard_label_list[i] = tag
                                else:
                                    raise ValueError(
                                        "tagging schema must in {}. Got {} instead.".format(
                                            tagging_schemas, self.args.schema))
                            elif i == tar_senti[cur][0][1]:
                                if self.args.schema == 't':
                                    tag = "T-" + tar_senti[cur][1]
                                    labels.append(tag)
                                    if hard_label_list[i] == 'O':
                                        hard_label_list[i] = tag
                                elif self.args.schema == 'bio':
                                    tag = "I-" + tar_senti[cur][1]
                                    labels.append(tag)
                                    if hard_label_list[i] == 'O':
                                        hard_label_list[i] = tag
                                else:
                                    raise ValueError(
                                        "tagging schema must in {}. Got {} instead.".format(
                                            tagging_schemas, self.args.schema))
                                cur += 1
                        except IndexError as e:
                            # process overlapped aspect terms
                            cur += 1
                            i -= 1
                            logger.debug(f"{text} has OVERLAPPED aspect terms")
                        i += 1
                    assert len(text.split()) == len(labels)
                    if rest == '':
                        f.write(f"{text}{self.sep}{' '.join(labels)}\n")
                    else:
                        f.write(
                            f"{text}{self.sep}{rest.strip()}{self.sep}{' '.join(hard_label_list)}\n"
                        )
                except Empty:
                    if ready:
                        break


def run(args):
    queue = Queue()
    producer = Producer(args, 'Producer', queue, args.sep)
    consumer = Consumer(args, 'Consumer', queue, args.sep)
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()


if __name__ == '__main__':
    run(
        parser.parse_args([
            "--dataset", "processed1/dp_tmp/rest.train.txt", "--output-file",
            "processed1/ent_tmp/rest.train.txt", "--batch-size", "512"
        ]))
