from typing import List, Tuple


def absa_evaluate(pred_Y: List[List[str]], gold_Y: List[List[str]]):
    """evaluate function for end2end aspect based sentiment analysis, with labels: {B,I}-{POS, NEG, NEU} and O

    Parameters
    ----------
    pred_Y : List[List[str]]
        predicted tags
    gold_Y : List[List[str]]
        gold standard tags (i.e., post-processed labels)

    Returns
    -------
    Tuple[float, float, str]
        precision, recall, micro f1
    """
    assert len(gold_Y) == len(pred_Y)
    TP, FN, FP = 0, 0, 0
    for i in range(len(gold_Y)):
        gold = gold_Y[i]
        pred = pred_Y[i]
        assert len(gold) == len(pred)
        gold_aspects = tag2aspect_sentiment(gold)
        pred_aspects = tag2aspect_sentiment(pred)
        hits_num = match(pred=pred_aspects, gold=gold_aspects)
        TP += hits_num
        FP += (len(pred_aspects) - hits_num)
        FN += (len(gold_aspects) - hits_num)

    pre = float(TP) / float(TP + FP + 0.00001)
    rec = float(TP) / float(TP + FN + 0.00001)
    f1 = 2 * pre * rec / (pre + rec + 0.00001)

    return pre, rec, f1


def tag2aspect(tag_sequence):
    """
    convert BIO tag sequence to the aspect sequence
    :param tag_sequence: tag sequence in BIO tagging schema
    :return:
    """
    ts_sequence = []
    beg = -1
    for index, ts_tag in enumerate(tag_sequence):
        if ts_tag == 'O':
            if beg != -1:
                ts_sequence.append((beg, index - 1))
                beg = -1
        else:
            cur = ts_tag.split('-')[0]  # unified tags
            if cur == 'B':
                if beg != -1:
                    ts_sequence.append((beg, index - 1))
                beg = index

    if beg != -1:
        ts_sequence.append((beg, index))
    return ts_sequence

def tag2aspect_sentiment(ts_tag_sequence: List[str]) -> List[Tuple[int, int, str]]:
    """support Tag sequence: ['O', 'B-POS', 'B-NEG', 'B-NEU', 'I-POS', 'I-NEG', 'I-NEU']

    Parameters
    ----------
    ts_tag_sequence : List[str]
        tag sequence like [O, T-POS, O, T-NEG, ...]

    Returns
    -------
    List[Tuple[int, int, str]]
        aspect terms with start index, end index and sentiment polarity
    """
    ts_sequence, sentiments = [], []
    start = -1
    for index, ts_tag in enumerate(ts_tag_sequence):
        if ts_tag == 'O':
            if start != -1:
                # TODO: process sentiment inconsistence
                ts_sequence.append((start, index - 1, sentiments[0]))
                start, sentiments = -1, []
        else:
            cur, pos = ts_tag.split('-')
            if cur == 'B':
                if start != -1:
                    ts_sequence.append((start, index - 1, sentiments[0]))
                start, sentiments = index, [pos]
            else:
                if start != -1:
                    sentiments.append(pos)
    if start != -1:
        ts_sequence.append((start, index, sentiments[0]))
    return ts_sequence


def match(pred: List[Tuple], gold: List[Tuple]):
    true_count = 0
    for t in pred:
        if t in gold:
            true_count += 1
    return true_count



def absa_evaluate_polarity(pred_Y: List[List[str]], gold_Y: List[List[str]], polarity: str) -> Tuple[float, float, float]:
    """evaluate function for end2end aspect based sentiment analysis, with labels: {B,I}-{POS, NEG, NEU} and O

    Parameters
    ----------
    pred_Y : List[List[str]]
        predicted tags
    gold_Y : List[List[str]]
        gold standard tags (i.e., post-processed labels)
    polarity : str
        polarity needed to be evaluated

    Returns
    -------
    Tuple[float, float, str]
        precision, recall, micro f1
    """
    assert len(gold_Y) == len(pred_Y)
    TP, FN, FP = 0, 0, 0
    for i in range(len(gold_Y)):
        gold = gold_Y[i]
        pred = pred_Y[i]
        assert len(gold) == len(pred)
        gold_aspects = tag2aspect_sentiment(gold)
        pred_aspects = tag2aspect_sentiment(pred)
        sifted_pred_aspects = [
            pred_aspect for pred_aspect in pred_aspects if polarity in pred_aspect
        ]
        sifted_gold_aspects = [
            gold_aspect for gold_aspect in gold_aspects if polarity in gold_aspect
        ]
        hits_num = match(pred=sifted_pred_aspects, gold=sifted_gold_aspects)
        TP += hits_num
        FP += (len(sifted_pred_aspects) - hits_num)
        FN += (len(sifted_gold_aspects) - hits_num)

    pre = float(TP) / float(TP + FP + 0.00001)
    rec = float(TP) / float(TP + FN + 0.00001)
    f1 = 2 * pre * rec / (pre + rec + 0.00001)

    return pre, rec, f1


def evaluate(test_Y, pred_Y):
    """evaluate function for aspect term extraction

    Parameters
    ----------
    pred_Y : List[List[str]]
        predicted tags
    gold_Y : List[List[str]]
        gold standard tags (i.e., post-processed labels)
    polarity : str
        polarity needed to be evaluated

    Returns
    -------
    Tuple[float, float, str]
        precision, recall, micro f1
    """
    assert len(test_Y) == len(pred_Y)
    length = len(test_Y)
    TP, FN, FP = 0, 0, 0

    for i in range(length):
        gold = test_Y[i]
        pred = pred_Y[i]
        assert len(gold) == len(pred)
        gold_aspects = tag2aspect(gold)
        pred_aspects = tag2aspect(pred)
        n_hit = match(pred=pred_aspects, gold=gold_aspects)
        TP += n_hit
        FP += (len(pred_aspects) - n_hit)
        FN += (len(gold_aspects) - n_hit)
    p = float(TP) / float(TP + FP + 0.00001)
    r = float(TP) / float(TP + FN + 0.0001)
    f1 = 2 * p * r / (p + r + 0.00001)
    return p, r, f1
