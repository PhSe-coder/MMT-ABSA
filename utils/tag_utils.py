import logging
import os
from typing import List, Tuple, Union
import tagme
import pickle
from qwikidata.sparql import return_sparql_query_results

# 标注的“Authorization Token”，需要注册才有
tagme.GCUBE_TOKEN = "58cf013e-71b9-4d8d-a7c1-396f5e842bec-843339462"

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')


def ot2bio_absa(ts_tag_sequence: List[str]):
    new_ts_sequence: List[str] = []
    prev_pos = 'O'
    for cur_ts_tag in ts_tag_sequence:
        if 'T' not in cur_ts_tag:
            new_ts_sequence.append('O')
            cur_pos = 'O'
        else:
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
            new_ts_sequence.append('I-%s' % cur_sentiment if prev_pos != 'O' else 'B-%s' %
                                   cur_sentiment)
        prev_pos = cur_pos
    return new_ts_sequence


def Annotation_mentions(txt):
    """
    发现那些文本中可以是维基概念实体的概念
    :param txt: 一段文本对象，str类型
    :return: 键值对，键为本文当中原有的实体概念，值为该概念作为维基概念的概念大小，那些属于维基概念但是存在歧义现象的也包含其内
    """
    annotation_mentions = tagme.mentions(txt)
    dic = dict()
    for mention in annotation_mentions.mentions:
        try:
            dic[str(mention).split(" [")[0]] = str(mention).split("] lp=")[1]
        except:
            logger.error('error annotation_mention about ' + mention)
    return dic


def Annotate(txt, language="en", theta=0.1):
    """
    解决文本的概念实体与维基百科概念之间的映射问题
    :param txt: 一段文本对象，str类型
    :param language: 使用的语言 “de”为德语, “en”为英语，“it”为意语.默认为英语“en”
    :param theta:阈值[0, 1]，选择标注得分，阈值越大筛选出来的映射就越可靠，默认为0.1
    :return:键值对[(begin, end): (score, mention, entity_title, entity_id, uri)]
    """
    annotations = tagme.annotate(txt, lang=language)
    dic = dict()
    for ann in annotations.get_annotations(theta):
        try:
            dic[(ann.begin, ann.end)] = (ann.score, ann.mention, ann.entity_title, ann.entity_id,
                                         ann.uri())
        except:
            logger.error('error annotation about ' + ann)
    return dic


def get_base_classes_of_item(entity_id: Union[str, None],
                             entity_dir_path: str = None) -> Tuple[str]:
    if entity_id is None:
        return []
    if entity_dir_path:
        os.makedirs(entity_dir_path, exist_ok=True)
    path = os.path.join(entity_dir_path, entity_id + '.pkl')
    if os.path.exists(path):
        with open(path, "rb") as f:
            ret = pickle.load(f)
        return ret
    sparql_query = """
SELECT ?pLabel WHERE {{
  wd:{} wdt:P279 ?p .
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "en" .
   }}
}}
""".format(entity_id)
    results = return_sparql_query_results(sparql_query)
    ret = tuple(binding["pLabel"]["value"] for binding in results["results"]["bindings"])
    if entity_dir_path:
        with open(path, "wb") as f:
            pickle.dump(ret, f)
    return ret
