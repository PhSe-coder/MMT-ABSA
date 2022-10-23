from typing import List, Tuple


def absa_evaluate(pred_Y: List[List[str]], gold_Y: List[List[str]]):
    """
    evaluate function for end2end aspect based sentiment analysis, with labels: {B,I}-{POS, NEG, NEU} and O
    :param gold_Y: gold standard tags (i.e., post-processed labels)
    :param pred_Y: predicted tags
    :return:
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

def tag2aspect_sentiment(ts_tag_sequence: List[str]) -> List[Tuple[int, int, str]]:
    '''
    support Tag sequence: ['O', 'B-POS', 'B-NEG', 'B-NEU', 'I-POS', 'I-NEG', 'I-NEU']
    '''
    ts_sequence, sentiments = [], []
    start = -1
    for index, ts_tag in enumerate(ts_tag_sequence):
        if ts_tag == 'O':
            if start != -1:
                # TODO: process sentiment inconsistence
                ts_sequence.append((start, index-1, sentiments[0]))
                start, sentiments = -1, []
        else:
            cur, pos = ts_tag.split('-')
            if cur == 'B':
                if start != -1:
                    ts_sequence.append((start, index-1, sentiments[0]))
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