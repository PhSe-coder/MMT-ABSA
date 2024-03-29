import logging
from collections import ChainMap
from typing import Dict, Iterable, List, Set, Tuple, Union
import random
from .stanza_annotation import annotation
from stanza.models.common.doc import Sentence

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
random.seed(42)
MR = ('amod', 'advmod', 'rcmod')
JJ = ('JJ', 'JJS', 'JJR')
NN = ('NN', 'NNS', 'NNP', 'NNPS')
SENTMENT_MAP = {0: 'NEG', 1: 'NEU', 2: 'POS'}

__all__ = ['Rule', 'SENTMENT_MAP']


class Rule:

    def __init__(self, text: Union[str, Sentence], pos_words: Set[str], neg_words: Set[str]):
        if isinstance(text, str):
            sentence = annotation(text)
        elif isinstance(text, Sentence):
            sentence = text
        self.sentence = sentence
        self.doc: List[dict] = sentence.to_dict()
        self.pos_words = pos_words
        self.neg_words = neg_words

    def propagation(self, targets: Set[str], opinions: Set[str], ents: Set[str]):
        targ11 = self.R1_1(ents)
        targ12 = self.R1_2()
        op21 = self.R2_1(targets)
        op22 = self.R2_2(targets)
        targ31 = self.R3_1(targets)
        targ32 = self.R3_2(targets)
        op41 = self.R4_1(opinions)
        op42 = self.R4_2(opinions)
        targ51 = self.R5_1()
        op61 = self.R6_1(opinions)
        # if len(tar_dict) == 0:
        #     targ71 = self.R7_1(doc)
        targ8 = self.R8()
        tar_dict = ChainMap(targ11, targ12, targ31, targ32, targ51, targ8)
        # if len(tar_dict) == 0:
        #     if random.random() > 0.5:
        #         targ9 = self.R9(ents)
        #         tar_dict.update(targ9)
        op_set = op21 | op22 | op41 | op42 | op61
        targets.update(self.id2text(tar_dict.keys()))
        opinions.update(op_set)
        return tar_dict, op_set

    def id2text(self, ids: Iterable[Tuple], lower=True):
        doc = self.doc
        text = set()
        for item in ids:
            assert len(item) <= 2
            if len(item) == 1:
                iter_range = range(item[0] - 1, item[0])
            else:
                iter_range = range(item[0] - 1, item[1])
            _ = ' '.join(doc[i]['text'] for i in iter_range)
            text.add(_.lower() if lower else _)
        return text

    def target_expand(self, tar: dict):
        doc = self.doc
        # expand multi-word aspect term like "rose roll", "Nicoise salad", etc.
        res = list(
            filter(
                lambda d: d['head'] == tar['id'] and d['deprel'] == 'compound' and d['xpos'] in NN,
                doc))
        # expand multi-word aspect term like "office of the chair"
        # if len(res) == 0:
        #   res = list(filter(lambda d: d['head'] == tar['id'] and
        #                     d['deprel'] == 'nmod' and
        #                     d['xpos'] in NN, doc))
        # for r in res:
        #   _ = list(filter(lambda d: d['head'] == r['id'] and
        #               d['deprel'] in ('case', 'det', 'compound', 'amod') and
        #               d['xpos'] in ('IN', 'DT', *NN), doc))
        #   res.extend(_)
        res.append(tar)
        res = sorted(res, key=lambda d: d['id'])
        if len(res) == 1:
            return (res[0]['id'], )
        else:
            return (res[0]['id'], res[-1]['id'])

    def R1_1(self, ents: Set[str]):
        target_dict: Dict[Union[Tuple[int, int], Tuple[int]], str] = {}
        doc = self.doc
        for item in doc:
            if item['xpos'] in JJ:
                token = item["text"].lower()
                if token in self.pos_words:
                    label = 'POS'
                elif token in self.neg_words:
                    label = 'NEG'
                else:
                    label = SENTMENT_MAP.get(self.sentence.sentiment)
                head = item['head']
                if head == 0:
                    continue
                if item['deprel'] in MR:
                    # the id of token starts at 1 rather than 0
                    target = doc[head - 1]  # O-->O-dep-->T
                    # comments:
                    # if any noun's head is the target, point to the noun instead.
                    # example sentence: This is by far the best feature of etrade .
                    target_heads = tuple(filter(
                        lambda d: d['head'] == target['id'] and d['deprel'] in ['nmod'] and d[
                            'xpos'] in NN, doc))
                    for tok in target_heads:  # O-->O-dep-->H<--T-dep<--T
                        _tok = self.target_expand(tok)
                        if _tok not in target_dict:
                            target_dict[_tok] = label
                    # if property of something is good, this thing is likely to be aspect terms
                    # example sentence: The phone has a good screen.
                    target_heads_1 = tuple(
                        filter(
                            lambda d: d['head'] == target['head'] and d['deprel'] in [
                                'nsubj', 'obj'
                            ] and d['xpos'] in NN and d['id'] != target['id'] and d['text'] in ents,
                            doc))
                    for tok in target_heads_1:  # O-->O-dep-->H<--T-dep<--T
                        _tok = self.target_expand(tok)
                        if _tok not in target_dict:
                            target_dict[_tok] = label
                    if len(target_heads) + len(target_heads_1) == 0:
                        _target = self.target_expand(target)
                        if target['xpos'] in NN and _target not in target_dict:
                            target_dict[_target] = label
                        # find conj relationship aspect terms
                        target_heads = tuple(filter(
                        lambda d: d['head'] == target['id'] and d['deprel'] in ['conj'] and d[
                            'xpos'] in NN, doc))
                        for tok in target_heads:
                            _tok = self.target_expand(tok)
                            if _tok not in target_dict:
                                target_dict[_tok] = label
        return target_dict

    # example sentence: "iPod" is the best mp3 player.
    def R1_2(self):
        target_dict: Dict[Union[Tuple[int, int], Tuple[int]], str] = {}
        doc = self.doc
        for item in doc:
            if item['xpos'] in JJ:
                token = item["text"].lower()
                if token in self.pos_words:
                    label = 'POS'
                elif token in self.neg_words:
                    label = 'NEG'
                else:
                    label = SENTMENT_MAP.get(self.sentence.sentiment)
                head = item['head']
                if head == 0:
                    continue
                if item['deprel'] in MR:
                    # the id of token starts at 1 rather than 0
                    target = doc[head - 1]
                    if target['xpos'] in NN:
                        target_heads = tuple(
                            filter(
                                lambda d: d['head'] == target['id'] and d['deprel'] in ['nsubj'] and
                                d['xpos'] in NN, doc))
                        if len(target_heads) > 1:
                            logger.debug(
                                f"{' '.join([token['text'] for token in doc])} has MULTIPLE target heads."
                            )
                        for tok in target_heads:  # O-->O-dep-->H<--T-dep<--T
                            _tok = self.target_expand(tok)
                            if _tok not in target_dict:
                                target_dict[_tok] = label
        return target_dict

    def R2_1(self, known_targets: Set[str]):
        list_opinion = []
        doc = self.doc
        for item in doc:
            if item['deprel'] in MR and item['xpos'] in JJ \
              and doc[item['head']-1]['text'].lower() in known_targets:
                assert item['head'] > 0
                list_opinion.append(item['text'])
        return set(list_opinion)

    def R2_2(self, known_targets: Set[str]):
        list_opinion = []
        doc = self.doc
        for item in doc:
            if item['xpos'] in NN and item['deprel'] == 'nsubj' and item['text'].lower(
            ) in known_targets:
                assert item['head'] > 0
                res = filter(
                    lambda d: d['id'] == item['head'] and d['deprel'] in MR and d['xpos'] in JJ,
                    doc)
                tok_list = [r['text'] for r in res]
                list_opinion.extend(tok_list)
        return set(list_opinion)

    def R3_1(self, known_targets: Set[str]):
        doc = self.doc
        sentence = self.sentence
        target_dict: Dict[Union[Tuple[int, int], Tuple[int]], str] = {}
        for item in doc:
            if item['deprel'] == 'conj' and item['xpos'] in NN and item['text'].lower(
            ) in known_targets:
                assert item['head'] > 0
                head = doc[item['head'] - 1]
                if head['xpos'] in NN and head['text'].lower() in known_targets:
                    _tok = self.target_expand(item)
                    if _tok not in target_dict:
                        target_dict[_tok] = SENTMENT_MAP.get(sentence.sentiment)
        return target_dict

    def R3_2(self, known_targets: Set[str]):
        doc = self.doc
        sentence = self.sentence
        target_dict: Dict[Union[Tuple[int, int], Tuple[int]], str] = {}
        for item in doc:
            if item['deprel'] == 'obj' and item['text'].lower() in known_targets:
                assert item['head'] > 0
                head = doc[item['head'] - 1]
                target_heads = tuple(
                    filter(
                        lambda d: d['head'] == head['id'] and d['deprel'] == 'nsubj' and d['xpos']
                        in NN, doc))
                if len(target_heads) > 1:
                    logger.debug(
                        f"{' '.join([token['text'] for token in doc])} has MULTIPLE target heads")
                for tok in target_heads:
                    _tok = self.target_expand(tok)
                    if _tok not in target_dict:
                        target_dict[_tok] = SENTMENT_MAP.get(sentence.sentiment)
        return target_dict

    def R4_1(self, known_opinions: Set[str]):
        list_opinion = []
        doc = self.doc
        for item in doc:
            if item['deprel'] == 'conj' and item['xpos'] in JJ:
                assert item['head'] > 0
                head = doc[item['head'] - 1]
                if head['xpos'] in JJ and head['text'].lower() in known_opinions:
                    list_opinion.append(item['text'])
        return set(list_opinion)

    def R4_2(self, known_opinions: Set[str]):
        list_opinion = []
        doc = self.doc
        for item in doc:
            if item['deprel'] in MR and item['text'].lower() in known_opinions:
                assert item['head'] > 0
                head = doc[item['head'] - 1]
                opinion_heads = tuple(
                    filter(
                        lambda d: d['head'] == head['id'] and d['deprel'] in MR and d['xpos'] in JJ
                        and d['id'] != item['id'], doc))
                if len(opinion_heads) > 1:
                    logger.debug(
                        f"{' '.join([token['text'] for token in doc])} has MULTIPLE opinion heads")
                for opin in opinion_heads:
                    list_opinion.append(opin['text'])
        return set(list_opinion)

    # example sentence: service is good
    # example sentence: Email lists are alot more convenient than chat rooms .
    def R5_1(self):
        target_dict: Dict[Union[Tuple[int, int], Tuple[int]], str] = {}
        doc = self.doc
        for item in doc:
            if item['xpos'] in JJ:
                token = item["text"].lower()
                if token in self.pos_words:
                    label = 'POS'
                elif token in self.neg_words:
                    label = 'NEG'
                else:
                    label = SENTMENT_MAP.get(self.sentence.sentiment)
                target_heads = tuple(
                    filter(
                        lambda d: d['head'] == item['id'] and d['deprel'] in ['nsubj']
                        and d['xpos'] in NN, doc))
                if len(target_heads) > 1:
                    logger.debug(
                        f"{' '.join([token['text'] for token in doc])} has MULTIPLE target heads")
                for tok in target_heads:
                    _tok = self.target_expand(tok)
                    if _tok not in target_dict:
                        target_dict[_tok] = label
                    # find conj relationship aspect terms
                    # example sentence: The theory and demands are rigorous .
                    heads = tuple(filter(
                        lambda d: d['head'] == tok['id'] and d['deprel'] in ['conj'] and d[
                            'xpos'] in NN, doc))
                    for tok in heads:
                        _tok = self.target_expand(tok)
                        if _tok not in target_dict:
                            target_dict[_tok] = label
        return target_dict

    def R6_1(self, know_opinions: Set[str]):
        list_opinion = []
        doc = self.doc
        for item in doc:
            if item['deprel'] in ('nsubj',) and item['xpos'] in NN \
              and doc[item['head']-1]['xpos'] in JJ \
              and doc[item['head']-1]['text'].lower() in know_opinions:
                assert item['head'] > 0
                list_opinion.append(doc[item['head'] - 1]['text'])
        return set(list_opinion)

    def R7_1(self):
        target_dict: Dict[Union[Tuple[int, int], Tuple[int]], str] = {}
        doc = self.doc
        for item in doc:
            tok = None
            if item['deprel'] in ('obj', 'nsubj', 'root') and item['xpos'] in NN:
                tok = item
            elif item['deprel'] == 'det' and doc[item['head'] - 1]['xpos'] in NN:
                tok = doc[item['head'] - 1]
            if tok is not None:
                _tok = self.target_expand(tok)
                if _tok not in target_dict:
                    target_dict[_tok] = SENTMENT_MAP.get(self.sentence.sentiment)
        return target_dict

    def R8(self):
        target_dict: Dict[Union[Tuple[int, int], Tuple[int]], str] = {}
        doc = self.doc
        for item in doc:
            if item['deprel'] in ('xcomp', ) and item['xpos'] in JJ:
                assert item['head'] > 0
                head = doc[item['head'] - 1]
                target_heads = tuple(
                    filter(
                        lambda d: d['head'] == head['id'] and d['deprel'] in
                        ('nsubj', ) and d['xpos'] in NN, doc))
                if len(target_heads) > 1:
                    logger.debug(f"{self.sentence.text} has MULTIPLE target heads.")
                for tok in target_heads:  # O-->O-dep-->H<--T-dep<--T
                    _tok = self.target_expand(tok)
                    if _tok not in target_dict:
                        token = item['text'].lower()
                        if token in self.pos_words:
                            label = 'POS'
                        elif token in self.neg_words:
                            label = 'NEG'
                        else:
                            label = SENTMENT_MAP.get(self.sentence.sentiment)
                        target_dict[_tok] = label
        return target_dict

    def R9(self, ents: Set[str]):
        target_dict: Dict[Union[Tuple[int, int], Tuple[int]], str] = {}
        doc = self.doc
        for item in doc:
            tok = None
            if item['deprel'] in ('obj', 'nsubj',
                                  'nsubj:pass') and item['xpos'] in NN and item['text'] in ents:
                tok = item
            # example: E - Trade customer service
            # if item['deprel'] in ('compound', ):
            #     _tok = self.target_expand(item)
            #     # if _tok not in target_dict:
            #     #     target_dict[_tok] = SENTMENT_MAP.get(self.sentence.sentiment)
            #     words = self.id2text((_tok,), False).pop()
            #     if item['xpos'] in NN and words in ents:
            #         tok = doc[item['head'] - 1]
            if tok is not None:
                _tok = self.target_expand(tok)
                if _tok not in target_dict:
                    target_dict[_tok] = SENTMENT_MAP.get(self.sentence.sentiment)
        return target_dict