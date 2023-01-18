import logging
import os
from typing import List, Tuple, Union, Set
import tagme
import pickle
from qwikidata.sparql import return_sparql_query_results
# tagme token
tagme.GCUBE_TOKEN = "58cf013e-71b9-4d8d-a7c1-396f5e842bec-843339462"

# logger = logging.getLogger(__name__)
# logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')

TypedStringSpan = Tuple[str, Tuple[int, int]]


class InvalidTagSequence(Exception):

    def __init__(self, tag_sequence=None):
        super().__init__()
        self.tag_sequence = tag_sequence

    def __str__(self):
        return " ".join(self.tag_sequence)

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


def Annotation_mentions(txt, logger=None):
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
            if logger:
                logger.error('error annotation_mention about ' + mention)
            else:
                print('error annotation_mention about ' + mention)
    return dic


def Annotate(txt, language="en", theta=0.1, logger=None):
    """
    解决文本的概念实体与维基百科概念之间的映射问题
    :param txt: 一段文本对象，str类型
    :param language: 使用的语言 “de”为德语, “en”为英语，“it”为意语.默认为英语“en”
    :param theta:阈值[0, 1]，选择标注得分，阈值越大筛选出来的映射就越可靠，默认为0.1
    :return:键值对[(begin, end): (score, mention, entity_title, entity_id, uri)]
    """
    annotations = tagme.annotate(txt, lang=language)
    dic = dict()
    if annotations is None:
        return dic
    for ann in annotations.get_annotations(theta):
        try:
            dic[(ann.begin, ann.end)] = (ann.score, ann.mention, ann.entity_title, ann.entity_id,
                                         ann.uri())
        except:
            if logger:
                logger.error('error annotation about ' + ann)
            else:
                print('error annotation about ' + ann)

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
    assert entity_id.startswith("Q")
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


def bio_tags_to_spans(tag_sequence: List[str],
                      classes_to_ignore: List[str] = None) -> List[TypedStringSpan]:
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans. This function works properly when
    the spans are unlabeled (i.e., your labels are simply "B", "I", and "O").

    # Parameters

    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.

    # Returns

    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        if bio_tag not in ["B", "I", "O"]:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]
        if bio_tag == "O" or conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)