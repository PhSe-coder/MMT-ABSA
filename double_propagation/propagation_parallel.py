from rule import Rule
from typing import List
from argparse import ArgumentParser
import nltk
from nltk.corpus import opinion_lexicon
from queue import Queue, Empty
import threading, time
nltk.download('opinion_lexicon')
step = 0

class Producer(threading.Thread):
    def __init__(self, args, name, queue):
        threading.Thread.__init__(self, name=name)
        self.data = queue
        self.args = args

    def run(self):
        global step
        for _ in range(self.args.epoch_nums):
            with open(self.args.dataset, "r") as f:
                text_list = []
                for idx, line in enumerate(f):
                    text = line.split('***')[0]
                    text_list.append(text)
                    if idx % self.args.batch_size == 0:
                        sentences = Rule.annotation_plus(text_list)
                        for sentence in sentences:
                            self.data.put(sentence)
                        text_list.clear()
                if len(text_list) != 0:
                    sentences = Rule.annotation_plus(text_list)
                    for sentence in sentences:
                        self.data.put(sentence)
                    text_list.clear()
            time.sleep(3)
            step += 1
        print("%s finished!" % self.getName())


class Consumer(threading.Thread):
    def __init__(self,args, name, queue):
        threading.Thread.__init__(self,name=name)
        self.data = queue
        self.args = args
        self.positive_words = set(opinion_lexicon.positive())
        self.negtive_words = set(opinion_lexicon.negative())
        with open('./double_propagation/Lessico-Opinion.txt') as f:
            ops = f.read().splitlines()
        with open('./double_propagation/Lessico-Target.txt') as f:
            tars = f.read().splitlines()
        self.target_set = set(tars)
        self.opinion_set = set(ops)
        
    def run(self):
        global step
        with open(self.args.dest, "w") as f:
            while True:
                try:
                    sentence = self.data.get(timeout=8)
                    rule = Rule(sentence, self.positive_words, self.negtive_words)
                    tar_dict, _ = rule.propagation(self.target_set, self.opinion_set)
                    if step != self.args.epoch_nums - 1:
                        continue
                    text = sentence.text
                    tar_senti = sorted(tar_dict.items(), key=lambda d:d[0][0])
                    labels: List[str] = []
                    cur = 0
                    i = 0
                    while i in range(len(text.split())):
                        i += 1
                        try:
                            if cur >= len(tar_senti):
                                labels.append('O')
                                continue
                            if i < tar_senti[cur][0][0]:
                                labels.append('O')
                            elif i == tar_senti[cur][0][0]:
                                labels.append("B-"+tar_senti[cur][1])
                                if len(tar_senti[cur][0]) == 1:
                                    cur += 1
                            elif i > tar_senti[cur][0][0] and i < tar_senti[cur][0][1]:
                                labels.append("I-"+tar_senti[cur][1])
                            elif i == tar_senti[cur][0][1]:
                                labels.append("I-"+tar_senti[cur][1])
                                cur += 1
                        except IndexError as e:
                            # process overlapped aspect terms
                            cur += 1
                            i -= 1
                            print(f"{text} has OVERLAPPED aspect terms")
                    assert len(text.split()) == len(labels)
                    f.write(f"{text}***{' '.join(labels)}\n")
                except Empty:
                    break
        print("%s finished!" % self.getName())

def main(args):
    queue = Queue()
    producer = Producer(args, 'Producer', queue)
    consumer = Consumer(args, 'Consumer', queue)
    producer.start()
    consumer.start()

    producer.join()
    consumer.join()
    print('All threads finished!')

if __name__ == '__main__':
    parser = ArgumentParser(description='Annotate a absa dataset by Double Propagation Algorithm')
    parser.add_argument("--dataset", required=True, help="Dataset to be annotated")
    parser.add_argument("--dest", required=True, help="Path where the annotation result should to be saved")
    parser.add_argument("--epoch_nums", type=int, default=3, help="iteration numbers of the algorithm")
    parser.add_argument("--batch_size", type=int, default=32, help="annotation batch size")
    main(parser.parse_args())