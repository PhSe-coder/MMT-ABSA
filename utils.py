

from typing import List


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
