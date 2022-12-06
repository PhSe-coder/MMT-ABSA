import logging
import threading
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Empty, Queue
from random import randint
import time
import os
from typing import List
from urllib3.exceptions import MaxRetryError
from requests.exceptions import ProxyError, JSONDecodeError
import numpy as np
import fasttext.util
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils.tag_utils import Annotate, get_base_classes_of_item

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
parser.add_argument("--entity-path",
                    default="./wikidata5m_entity.txt",
                    help="path of the wikidata entity file")
parser.add_argument("--mean-vec", help="mean vector numpy file path")
parser.add_argument("--schema",
                    type=str,
                    default="t",
                    choices=tagging_schemas,
                    help="annotaion schema for each tag")
parser.add_argument(
    "--entity-save-path",
    default='processed/ent',
    help="path to save the baseclass entities",
)


class Producer(threading.Thread):

    def __init__(self, args, name: str, queue: Queue, sep="***"):
        threading.Thread.__init__(self, name=name)
        self.queue = queue
        self.args = args
        self.sep = sep
        self.entity_path: str = args.entity_path
        # file line count
        self.count = sum(1 for _ in open(self.args.dataset, "rb"))
        fasttext.util.download_model('en', if_exists='ignore')  # English
        self.model = fasttext.load_model('cc.en.300.bin')
        self.mean_vec = np.load(args.mean_vec)
        self.entity_map = {}
        total = sum(1 for _ in open(self.entity_path, "rb"))
        logger.info(f"reading wikidata entity file {self.entity_path}")
        with open(self.entity_path, encoding='utf-8') as f:
            for line in tqdm(f, total=total):
                e_id, e_str = line.strip().split("\t", maxsplit=1)
                [self.entity_map.__setitem__(e, e_id) for e in e_str.split("\t")]
        pass

    def process(self, index: int, sentence: str):
        # start = time.time()
        # print(index, "start", time.time() - start)
        try:
            res_dict = Annotate(sentence, theta=0.05)
            if not res_dict:
                return {index: res_dict}
            # consider sentence with one entity
            if len(res_dict) == 1:
                result = res_dict
            else:
                begins = tuple(begin for begin, _ in res_dict.keys())
                ends = tuple(end for _, end in res_dict.keys())
                for i in range(len(begins) - 1):
                    assert begins[i] <= begins[i + 1]
                    # preserve the longest entity
                    if begins[i] == begins[i + 1] and ends[i] < ends[i + 1]:
                        res_dict.pop((begins[i], ends[i]), None)
                # merge the entity with continuous index
                result = {}
                begins = tuple(begin for begin, _ in res_dict.keys())
                ends = tuple(end for _, end in res_dict.keys())
                ent = {}
                for i in range(len(begins) - 1):
                    if ends[i] + 1 == begins[i + 1]:
                        ent[(begins[i], ends[i])] = res_dict.get((begins[i], ends[i]))
                        ent[(begins[i + 1], ends[i + 1])] = res_dict.get(
                            (begins[i + 1], ends[i + 1]))
                        continue
                    if len(ent) != 0:
                        keys = list(ent.keys())
                        result[(keys[0][0], keys[-1][-1])] = tuple(ent.values())
                        ent.clear()
                    # prevent overlap
                    if not any(
                            set(range(begins[i], ends[i])).intersection(set(range(*key)))
                            for key in result.keys()):
                        result[(begins[i], ends[i])] = res_dict.get((begins[i], ends[i]))
                if len(ent) != 0:
                    ent_keys = list(ent.keys())
                    result[(ent_keys[0][0], ent_keys[-1][-1])] = tuple(ent.values())
                    ent.clear()
                # if res_dict has exactly one entity after merge
                if not result:
                    result = res_dict
                # remove the overlap entites
                begins = tuple(begin for begin, _ in result.keys())
                ends = tuple(end for _, end in result.keys())
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
            res_result = {}
            # print(index, "before iter", time.time() - start)
            for k in result:
                baseclass_items = {}
                # for the merged entity, preserve the most similar entity
                if isinstance(result[k][0], tuple):
                    sims = []
                    for r in result[k]:
                        entity_title = r[2]
                        sim = float(
                            cosine_similarity(
                                self.mean_vec.reshape(1, -1),
                                np.array(self.model.get_word_vector(entity_title.lower())).reshape(
                                    1, -1)))
                        sims.append(sim)
                    result[k] = result[k][sims.index(max(sims))]
                entity_title = result[k][2]
                sim = float(
                    cosine_similarity(
                        self.mean_vec.reshape(1, -1),
                        np.array(self.model.get_word_vector(entity_title.lower())).reshape(1, -1)))
                # print(index, "base classes", time.time() - start)
                qid = self.entity_map.get(entity_title, None)
                # replace original item if the baseclass item can increase the domain similarity
                time.sleep(randint(0, 5))
                for item in get_base_classes_of_item(qid, self.args.entity_save_path):
                    _sim = float(
                        cosine_similarity(
                            self.mean_vec.reshape(1, -1),
                            np.array(self.model.get_word_vector(item.lower())).reshape(1, -1)))
                    if _sim > sim:
                        baseclass_items[item] = _sim
                items = tuple(item for item, _ in sorted(
                    baseclass_items.items(), key=lambda item: item[1], reverse=True))
                res_result[k] = (*result[k], items)
                # print(index, "one iter end", time.time() - start)
        except (MaxRetryError, ProxyError, JSONDecodeError) as e:
            logger.warning(e.args)
            time.sleep(1)
            return self.process(index, sentence)
        except Exception as e:
            raise e
        # print(index, "final", time.time() - start)
        return {index: res_result}

    def put(self, text_list: List[str], rest_list: List[str]):
        results = {}
        with ThreadPoolExecutor(max_workers=16) as t:
            for future in tqdm(as_completed(
                [t.submit(self.process, idx, sentence) for idx, sentence in enumerate(text_list)]),
                               total=len(text_list),
                               disable=False):  # disable the bar
                if future.exception():
                    future.set_result({})
                    logger.warning(future.exception().args)
                    exit(-1)
                results.update(future.result())
        res = [result for _, result in sorted(results.items(), key=lambda item: item[0])]
        for i, text in enumerate(text_list):
            self.queue.put((text, rest_list[i], res[i]))

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
                rest_list.append(rest.rsplit(self.sep, 1)[0])
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

    def run(self):
        global ready
        os.makedirs(os.path.dirname(self.args.output_file), exist_ok=True)
        with open(self.args.output_file, "w") as f:
            while True:
                try:
                    text, labels, res_dict = self.data.get(timeout=5)
                    tokens = text.strip().split()
                    labels = labels.strip().split()
                    offset = 0
                    tar_senti = []
                    for k, v in res_dict.items():
                        # skip if no baseclass item exists for the entity
                        if v[-1] == ():
                            continue
                        idx_tuple = tuple(text[:idx].count(" ") + offset for idx in k)
                        assert len(idx_tuple) == 2
                        # skip if token overlapping exists between extracted entity and aspect terms
                        sentis = set(
                            [labels[i][-3:] for i in range(idx_tuple[0], idx_tuple[1] + 1)])
                        if len(sentis) != 1:
                            continue
                        senti = "T-" + sentis.pop()
                        original_span = idx_tuple[1] + 1 - idx_tuple[0]
                        replaced_item = v[-1][0].split(" ")
                        replaced_span = len(replaced_item)
                        offset += (replaced_span - original_span)
                        tokens = tokens[:idx_tuple[0]] + replaced_item + tokens[idx_tuple[1] + 1:]
                        labels = labels[:idx_tuple[0]] + [senti
                                                          ] * replaced_span + labels[idx_tuple[1] +
                                                                                     1:]
                    tar_dict = {}
                    idx_list = []
                    tag = ''
                    for idx, label in enumerate(labels):
                        if label != 'O':
                            idx_list.append(idx)
                            tag = label
                            continue
                        if idx_list:
                            tar_dict[(idx_list[0], idx_list[-1])] = tag[-3:]
                            idx_list.clear()
                    if idx_list:
                        tar_dict[(idx_list[0], idx_list[-1])] = tag[-3:]
                        idx_list.clear()
                    tar_senti = list(tar_dict.items())
                    labels: List[str] = []
                    cur = 0
                    i = 0
                    while i in range(len(tokens)):
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
                                elif self.args.schema == 'bio':
                                    tag = "B-" + tar_senti[cur][1]
                                    labels.append(tag)
                                else:
                                    raise ValueError(
                                        "tagging schema must in {}. Got {} instead.".format(
                                            tagging_schemas, self.args.schema))
                                if len(tar_senti[cur]
                                       [0]) == 1 or tar_senti[cur][0][0] == tar_senti[cur][0][1]:
                                    cur += 1
                            elif i > tar_senti[cur][0][0] and i <= tar_senti[cur][0][1]:
                                if self.args.schema == 't':
                                    tag = "T-" + tar_senti[cur][1]
                                    labels.append(tag)
                                elif self.args.schema == 'bio':
                                    tag = "I-" + tar_senti[cur][1]
                                    labels.append(tag)
                                else:
                                    raise ValueError(
                                        "tagging schema must in {}. Got {} instead.".format(
                                            tagging_schemas, self.args.schema))
                                if i == tar_senti[cur][0][1]:
                                    cur += 1
                        except IndexError as e:
                            # process overlapped aspect terms
                            cur += 1
                            i -= 1
                            logger.debug(f"{text} has OVERLAPPED aspect terms")
                        i += 1
                    assert len(tokens) == len(labels), f"{tokens} {labels}"
                    f.write(f"{text}{self.sep}{' '.join(labels)}\n")
                except Empty:
                    if ready:
                        ready = False
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
            "--dataset", "./data/device.train.txt", "--output-file",
            "processed/ent_tmp/device.train.txt", "--batch-size", "512", "--mean-vec",
            "./processed/vectors/device_mean_vec.npy"
        ]))
