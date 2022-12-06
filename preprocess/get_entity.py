import logging
import os
import pickle
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from operator import itemgetter
from typing import List, Tuple

import fasttext
import fasttext.util
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from utils.tag_utils import Annotate

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
parser = ArgumentParser(description="Extract entities")
parser.add_argument("--src", type=str, default="./data")
parser.add_argument("--entity-dest", type=str, default="./processed/entities")
parser.add_argument("--vector-dest", type=str, default="./processed/vectors")
args = parser.parse_args()
nltk.download('stopwords')
# stopwords
stopword_set = stopwords.words('english')
wnl = WordNetLemmatizer()
save_dir = args.entity_dest
vector_dir = args.vector_dest
os.makedirs(save_dir, exist_ok=True)
os.makedirs(vector_dir, exist_ok=True)
fasttext.util.download_model("en", if_exists='ignore')
model = fasttext.load_model('cc.en.300.bin')


def process_ann(sentence: str):
    result = {}
    for score, mention, entity_title, entity_id, uri in Annotate(sentence, theta=0.05).values():
        if entity_title in result:
            if len(result[entity_title]) < len(mention):
                result[entity_title] = mention
        else:
            result[entity_title] = mention
    return result.values()


for domain in ("rest", "laptop", "service", "device"):
    entities = []
    file = os.path.join(args.src, f"{domain}.train.txt")
    counts = sum(1 for _ in open(file))
    sentences = [line.split("***")[0] for line in open(file).read().splitlines()]
    with ThreadPoolExecutor(max_workers=256) as t:
        for future in tqdm(as_completed([t.submit(process_ann, sentence)
                                         for sentence in sentences]),
                           total=counts,
                           desc=f"Extracting eitities in {file}"):
            entities.extend(future.result())
    with open(os.path.join(save_dir, domain + ".pkl"), "wb") as f:
        pickle.dump(entities, f)
    with open(os.path.join(save_dir, domain + ".pkl"), "rb") as f:
        entities: List[str] = pickle.load(f)
    # lemmatization
    entities = [wnl.lemmatize(word, 'n').lower() for word in entities if word not in stopword_set]
    counters = Counter(entities)
    sorted_entities: List[Tuple[str, int]] = sorted(filter(lambda item: item[1] >= 5,
                                                           counters.items()),
                                                    key=lambda item: item[1],
                                                    reverse=True)
    vec_dict = {}
    for entity in sorted_entities:
        e = entity[0]
        vec_dict[e] = model.get_word_vector(e)
    logger.info(f"compute mean vector for {domain} domain")
    getter = itemgetter(*[entity for entity, _ in sorted_entities[:10]])
    mean_vec = np.average(getter(vec_dict),
                          axis=0,
                          weights=[count for _, count in sorted_entities[:10]])
    path = os.path.join(args.vector_dest, domain + "_mean_vec")
    logger.info(f"save mean vector in {path}")
    np.save(path, mean_vec)
    # res = np.array([
    #     cosine_similarity(mean_vec.reshape(1, -1), vec_dict[k].reshape(1, -1)) for k in vec_dict
    # ]).squeeze()
    # topk = torch.topk(torch.from_numpy(res), res.shape[0]).indices.tolist()
    # itemgetter(*topk)(sorted_entities), sorted_entities