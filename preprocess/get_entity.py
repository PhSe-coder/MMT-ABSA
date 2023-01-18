import logging
import os
import pickle
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from operator import itemgetter
from typing import List, Tuple
from retry import retry
import fasttext
import fasttext.util
import nltk
import numpy as np
from func_timeout import func_set_timeout
import torch
from nltk.corpus import stopwords
from qwikidata.entity import WikidataItem
from qwikidata.linked_data_interface import get_entity_dict_from_api
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils.tag_utils import Annotate

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
parser = ArgumentParser(description="Extract entities")
parser.add_argument("--src", type=str, default="./data")
parser.add_argument("--entity-dest", type=str, default="./processed/entities")
parser.add_argument("--vector-dest", type=str, default="./processed/vectors")
parser.add_argument("--theta",
                    type=float,
                    default=0.05,
                    help="threshold for extracting valid entity")
parser.add_argument("--entity-path",
                    default="./wikidata5m_entity.txt",
                    help="path of the wikidata entity file")
parser.add_argument("--output-dir", default="./processed/ent_desc")
args = parser.parse_args()
nltk.download('stopwords')
os.makedirs(args.output_dir, exist_ok=True)
# stopwords
stopword_set = stopwords.words('english')
save_dir = args.entity_dest
vector_dir = args.vector_dest
os.makedirs(save_dir, exist_ok=True)
os.makedirs(vector_dir, exist_ok=True)
fasttext.util.download_model("en", if_exists='ignore')
model = fasttext.load_model('cc.en.300.bin')
total = sum(1 for _ in open(args.entity_path, "rb"))
logger.info(f"reading wikidata entity file {args.entity_path}")
entity_map = {}
with open(args.entity_path, encoding='utf-8') as f:
    for line in tqdm(f, total=total):
        e_id, e_str = line.strip().split("\t", maxsplit=1)
        [entity_map.__setitem__(e, e_id) for e in e_str.split("\t")]


def process_ann(sentence: str):
    result = {}
    _result = {}
    for score, mention, entity_title, entity_id, uri in Annotate(sentence, theta=0.05).values():
        if entity_title in result:
            if len(result[entity_title]) < len(mention):
                result[entity_title] = mention
        else:
            result[entity_title] = mention
        _result[mention] = entity_title
    return _result, result


for domain in ("rest", "laptop", "service", "device",):
    entities_tuple_list = []
    lines = []
    # get all sentences in a domain
    for file in glob(os.path.join("./data", f"{domain}.*.txt")):
        lines.extend([line.split("***") for line in open(file).read().splitlines()])
    with ThreadPoolExecutor(max_workers=256) as t:
        for future in tqdm(as_completed([t.submit(process_ann, line[0]) for line in lines]),
                           total=len(lines),
                           desc=f"Extracting eitities in {domain} domain"):
            entities_tuple_list.append(future.result())
    entities_dict_list = [entities_dict
                          for _, entities_dict in entities_tuple_list]  # {title: metion}
    _entities_dict = {
        k: v
        for entities_dict, _ in entities_tuple_list for k, v in entities_dict.items()
    }  # {metion: titile}
    # remove stopwords
    emtity_metions = [
        word for entities_dict in entities_dict_list for word in entities_dict.values()
        if word not in stopword_set
    ]
    # save domain entities with pickle
    with open(os.path.join(save_dir, domain + ".pkl"), "wb") as f:
        pickle.dump(emtity_metions, f)
    counters = Counter(emtity_metions)
    # sort entities by frequency, high -> low
    sorted_entities: List[Tuple[str, int]] = sorted(filter(lambda item: item[1] > 0,
                                                           counters.items()),
                                                    key=lambda item: item[1],
                                                    reverse=True)
    vec_dict = {}
    for entity in sorted_entities:
        e = entity[0]
        vec_dict[e] = model.get_word_vector(e.lower()) # case unsensitive for entities
    logger.info(f"compute mean vector for {domain} domain")
    getter = itemgetter(*[entity for entity, _ in sorted_entities[:10]])
    mean_vec = np.average(getter(vec_dict),
                          axis=0,
                          weights=[count for _, count in sorted_entities[:10]])
    path = os.path.join(args.vector_dest, domain + "_mean_vec")
    logger.info(f"save mean vector in {path}")
    np.save(path, mean_vec)
    res = np.array([
        cosine_similarity(mean_vec.reshape(1, -1), vec_dict[k].reshape(1, -1)) for k in vec_dict
    ]).squeeze()
    # topk = torch.topk(torch.from_numpy(res), int(0.6 * res.shape[0])).indices.tolist()
    # k = len([sim for sim in itemgetter(*topk)(res) if sim > 0.1])
    # topk = torch.topk(torch.from_numpy(res), k).indices.tolist()
    topk = torch.topk(torch.from_numpy(res), int(res.shape[0])).indices.tolist()
    ents = set([item[0] for item in itemgetter(*topk)(sorted_entities)])
    # path = os.path.join(args.output_dir, domain + ".txt")
    # with open(path, "w") as f:
    #     for k in tqdm(ents, desc="save entity with description"):
    #         f.write(f"{k}\n")
    # {entity_mention: entity_title}
    entity_dict = {entity: _entities_dict[entity] for entity in ents}
    entity_qid_dict = {entity: entity_map.get(title) for entity, title in entity_dict.items()}
    entity_desc_dict = {}

    @retry(tries=6) # retry when any exception occurs
    @func_set_timeout(30) # set timeout for request
    def get(e, q):
        return e, WikidataItem(get_entity_dict_from_api(q)).get_description()
    entity_qid_dict = {k: v for k, v in entity_qid_dict.items() if v is not None}
    with ThreadPoolExecutor(max_workers=16) as t:
        for future in tqdm(as_completed(
            [t.submit(get, entity, qid) for entity, qid in entity_qid_dict.items()]),
                           total=len(entity_qid_dict)):
            if future.exception() is None:
                result = future.result()
                entity_desc_dict[result[0]] = result[1]
    path = os.path.join(args.output_dir, domain + ".txt")
    # entity_title: entity_description
    with open(path, "w") as f:
        for k, v in tqdm(entity_desc_dict.items(), desc="save entity with description"):
            f.write(f"{k} is {v}.\n")
