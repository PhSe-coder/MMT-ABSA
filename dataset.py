from typing import List, Dict, Union

from torch import as_tensor, cat
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.generic import PaddingStrategy
from torch.utils.data.dataloader import default_collate
import linecache
from constants import TAGS

def transform(
    text: str,
    text_labels: str,
    wordpiece_tokens: List[str],
    special_tokens: List[str],
):
    text_tokens, text_labels = text.split(), text_labels.split()
    assert len(text_tokens) == len(text_labels)
    token_tuples = list(zip(text_tokens, text_labels))
    i, offset = 0, 0
    labels: List[str] = []
    for token in wordpiece_tokens:
        if token in special_tokens:
            tag = "SPECIAL_TOKEN"
        else:
            tt = token_tuples[i]
            if token.startswith("##"):
                tag = f"I{labels[-1][1:]}" if labels[-1] not in ["O", "SPECIAL_TOKEN"] else "O"
            else:
                tag = tt[1]
                if tag != "O":
                    tag = f"B{tag[1:]}" if labels[-1] in ["O", "SPECIAL_TOKEN"] else f"I{tag[1:]}"
            offset += len(token.replace("##", ""))
            if offset == len(tt[0]):
                i += 1
                offset = 0
        labels.append(tag)
    return labels


class BaseDataset(Dataset):

    def __init__(self, filename: str, tokenizer: PreTrainedTokenizer, device=None, src=True):
        dataset = []
        total_lines = sum(1 for _ in open(filename, "rb"))
        pos_list = [item.split()[0] for item in open("./ann/laptop_pos.txt").read().splitlines()]
        deprel_list = [
            item.split()[0] for item in open("./ann/laptop_deprel.txt").read().splitlines()
        ]
        with open(filename, "r") as f:
            for line in tqdm(f, total=total_lines, desc=filename):
                line = line.strip()
                pos_labels, deprel_labels, hard_labels = None, None, None
                try:
                    text, gold_labels, pos_labels, deprel_labels, hard_labels = line.rsplit("***")
                except ValueError:
                    text, gold_labels = line.rsplit("***")
                tok_dict: Dict[str, List[int]] = tokenizer(text,
                                                           padding=PaddingStrategy.MAX_LENGTH,
                                                           truncation=True)
                wordpiece_tokens = tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
                special_tokens = tokenizer.all_special_tokens
                glod_wordpiece_labels = transform(text, gold_labels, wordpiece_tokens,
                                                  special_tokens)
                _labels = [
                    TAGS.index(label) if label in TAGS else -1 for label in glod_wordpiece_labels
                ]
                data = {
                    "input_ids": as_tensor(tok_dict.input_ids, device=device),
                    "gold_labels": as_tensor(_labels.copy(), device=device),
                    "attention_mask": as_tensor(tok_dict.attention_mask, device=device),
                    "token_type_ids": as_tensor(tok_dict.token_type_ids, device=device),
                    "domains": as_tensor([src], device=device)
                }
                if hard_labels is not None:
                    labels = transform(text, hard_labels, wordpiece_tokens, special_tokens)
                    _labels = [TAGS.index(label) if label in TAGS else -1 for label in labels]
                    data['hard_labels'] = as_tensor(_labels.copy(), device=device)
                if pos_labels is not None:
                    labels = transform(text, pos_labels, wordpiece_tokens, special_tokens)
                    _labels.clear()
                    for idx, label in enumerate(labels):
                        if label in pos_list:
                            _labels.append(pos_list.index(label))
                        else:
                            if glod_wordpiece_labels[idx] not in ["O", "SPECIAL_TOKEN"]:
                                _labels.append(pos_list.index(label[0] + '-[UNK]'))
                            else:
                                _labels.append(-1)
                    data['pos_labels'] = as_tensor(_labels.copy(), device=device)
                if deprel_labels is not None:
                    labels = transform(text, deprel_labels, wordpiece_tokens, special_tokens)
                    _labels.clear()
                    for idx, label in enumerate(labels):
                        if label in deprel_list:
                            _labels.append(deprel_list.index(label))
                        else:
                            if glod_wordpiece_labels[idx] not in ["O", "SPECIAL_TOKEN"]:
                                _labels.append(deprel_list.index(label[0] + '-[UNK]'))
                            else:
                                _labels.append(-1)
                    data['deprel_labels'] = as_tensor(_labels.copy(), device=device)
                dataset.append(data)
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


# class MyDataset(Dataset):

#     def __init__(self, filename: str, tokenizer: PreTrainedTokenizer):
#         self.datafile = filename
#         self.total_lines = sum(1 for _ in open(filename, "rb"))
#         self.tokenizer = tokenizer

#     def __getitem__(self, index):
#         try:
#             line = linecache.getline(self.datafile,
#                                      index + 1)  # `getline` method start from index 1 rather than 0
#             line = line.strip()
#             text, gold_labels, pos_labels, deprel_labels, head_labels = line.rsplit("***")
#             tok_dict: Dict[str, List[int]] = self.tokenizer(text,
#                                                             padding=PaddingStrategy.MAX_LENGTH,
#                                                             truncation=True)
#             wordpiece_tokens = self.tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
#             special_tokens = self.tokenizer.all_special_tokens
#             glod_wordpiece_labels = transform(text, gold_labels, wordpiece_tokens, special_tokens)
#             labels = [-1]
#             subtoken_ids = [0]
#             gold_labels = ot2bio_absa(gold_labels.split(" "))
#             for i, token in enumerate(text.split()):
#                 labels.append(TAGS.index(gold_labels[i]))
#                 length = len(self.tokenizer.tokenize(token))
#                 for _ in range(length):
#                     subtoken_ids.append(i+1)
#                     if len(subtoken_ids) + 1 == self.tokenizer.model_max_length:
#                         break
#                 if len(subtoken_ids) + 1 == self.tokenizer.model_max_length:
#                     break
#             while len(subtoken_ids) < len(glod_wordpiece_labels):
#                 subtoken_ids.append(-1)
#             while len(labels) < len(glod_wordpiece_labels):
#                 labels.append(-1)
#             assert len(subtoken_ids) == len(labels) == self.tokenizer.model_max_length
#             # labels = [TAGS.index(label) if label in TAGS else -1 for label in glod_wordpiece_labels]
#             data = {
#                 "input_ids": as_tensor(tok_dict.input_ids),
#                 "gold_labels": as_tensor(labels),
#                 "subtoken_ids": as_tensor(subtoken_ids),
#                 "attention_mask": as_tensor(tok_dict.attention_mask),
#                 "token_type_ids": as_tensor(tok_dict.token_type_ids),
#             }
#             assert max(subtoken_ids) == (data["gold_labels"] != -1).sum()
#             if pos_labels is not None:
#                 pos_labels = pos_labels.split()
#                 pos_label_ids = []
#                 for i, token in enumerate(text.split()):
#                     length = len(self.tokenizer.tokenize(token))
#                     for _ in range(length):
#                         pos_label_ids.append(POS_DICT.get(pos_labels[i], POS_DICT.get('O')))
#                 pos_label_ids.insert(0, POS_DICT.get('[CLS]'))
#                 pos_label_ids.append(POS_DICT.get('[SEP]'))
#                 while len(pos_label_ids) > len(glod_wordpiece_labels):
#                     pos_label_ids.pop(-2)
#                 while len(pos_label_ids) < len(glod_wordpiece_labels):
#                     pos_label_ids.append(POS_DICT.get('[PAD]'))
#                 assert len(glod_wordpiece_labels) == len(pos_label_ids)
#                 data['pos_label_ids'] = as_tensor(pos_label_ids)
#             if deprel_labels is not None:
#                 deprel_labels = deprel_labels.split()
#                 head_labels = [int(label) for label in head_labels.split()]
#                 head_label_ids = [0]
#                 idx = 0
#                 deprel_label_ids = [DEPREL_DICT.get('[CLS]')]
#                 for i, token in enumerate(text.split()):
#                     head_label_ids.append(head_labels[i])
#                     deprel_label_ids.append(DEPREL_DICT.get(deprel_labels[i], DEPREL_DICT.get('O')))
#                     length = len(self.tokenizer.tokenize(token))
#                     for _ in range(length):
#                         idx += 1
#                         if idx + 1 == self.tokenizer.model_max_length:
#                             break
#                     if idx + 1 == self.tokenizer.model_max_length:
#                         break
#                 edge_src, edge_dst, edge_features = [], [], []
#                 for i, head_label in enumerate(head_label_ids):
#                     if head_label != 0 and head_label <= max(subtoken_ids) and i <= max(subtoken_ids):
#                         edge_src.append(head_label-1)
#                         edge_dst.append(i-1)
#                         edge_features.append(deprel_label_ids[i])
#                 g = dgl.graph((tensor(edge_src), tensor(edge_dst)), num_nodes=max(subtoken_ids))
#                 assert g.num_nodes() == (data['gold_labels'] != -1).sum()
#                 g.edata['ex'] = tensor(edge_features)
#                 data['deprel_graph'] = g
#         except Exception as e:
#             print(e)
#         return data

#     def __len__(self):
#         return self.total_lines

#     @staticmethod
#     def collate_fn(batch):
#         batch_dict = {}
#         try:
#             deprel_graph = [item.pop('deprel_graph') for item in batch]
#             batch_dict: Dict[str, List[Tensor]] = default_collate(batch)
#             batch_dict['deprel_graph'] = dgl.batch(deprel_graph)
#         except Exception as e:
#             print(e)
#         return batch_dict


class MyDataset(Dataset):

    def __init__(self, filenames: Union[List[str], str], tokenizer: PreTrainedTokenizer):
        if isinstance(filenames, str):
            filenames = [filenames]
        self.datafiles = filenames
        self.total_lines = [sum(1 for _ in open(filename, "rb")) for filename in filenames]
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        data = []
        try:
            for idx, datafile in enumerate(self.datafiles):
                # `getline` method start from index 1 rather than 0
                if index >= self.total_lines[idx]:
                    index = index % self.total_lines[idx]
                line = linecache.getline(datafile, index + 1)
                line = line.strip()
                text, gold_labels = line.rsplit("***")
                tok_dict: Dict[str, List[int]] = self.tokenizer(text,
                                                                padding=PaddingStrategy.MAX_LENGTH,
                                                                truncation=True)
                wordpiece_tokens = self.tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
                special_tokens = self.tokenizer.all_special_tokens
                glod_wordpiece_labels = transform(text, gold_labels, wordpiece_tokens, special_tokens)
                labels = [
                    TAGS.index(label) if label in TAGS else -1 for label in glod_wordpiece_labels
                ]
                valid_mask = tok_dict.attention_mask.copy()
                valid_mask[0] = 0
                valid_mask[len(valid_mask) - valid_mask[::-1].index(1) - 1] = 0
                data.append({
                    "input_ids": as_tensor(tok_dict.input_ids),
                    "gold_labels": as_tensor(labels),
                    "attention_mask": as_tensor(tok_dict.attention_mask),
                    "token_type_ids": as_tensor(tok_dict.token_type_ids),
                    "valid_mask": as_tensor(valid_mask),
                })
        except Exception as e:
            print(e)
        return data

    def __len__(self):
        return self.total_lines[0]


# class MMTDataset(Dataset):

#     def __init__(self,
#                  filename: str,
#                  tokenizer: PreTrainedTokenizer,
#                  device=None,
#                  src=True) -> None:
#         dataset = []
#         total_lines = sum(1 for _ in open(filename, "rb"))
#         with open(filename, "r") as f:
#             for line in tqdm(f, total=total_lines, desc=filename):
#                 line = line.strip()
#                 special_tokens = tokenizer.all_special_tokens
#                 input_ids, attention_mask, token_type_ids = [], [], []
#                 gold_label_list, dp_labels, domains = [], [], []
#                 for item in line.split("####"):
#                     text, gold_labels, hard_labels = item.rsplit("***")
#                     tok_dict: Dict[str, List[int]] = tokenizer(text,
#                                                                padding=PaddingStrategy.MAX_LENGTH,
#                                                                truncation=True)
#                     wordpiece_tokens = tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
#                     labels = transform(text, gold_labels, wordpiece_tokens, special_tokens)
#                     input_ids.append(as_tensor(tok_dict.input_ids, device=device))
#                     _labels = [TAGS.index(label) if label in TAGS else -1 for label in labels]
#                     gold_label_list.append(as_tensor(_labels.copy(), device=device))
#                     labels = transform(text, hard_labels, wordpiece_tokens, special_tokens)
#                     _labels = [TAGS.index(label) if label in TAGS else -1 for label in labels]
#                     dp_labels.append(as_tensor(_labels.copy(), device=device))
#                     attention_mask.append(as_tensor(tok_dict.attention_mask, device=device))
#                     token_type_ids.append(as_tensor(tok_dict.token_type_ids, device=device))
#                     # specified as a list for the benefit of tensor broadcasting
#                     domains.append(as_tensor([src], device=device))
#                 dataset.append({
#                     "input_ids": input_ids,
#                     "gold_labels": gold_label_list,
#                     "hard_labels": dp_labels,
#                     "attention_mask": attention_mask,
#                     "token_type_ids": token_type_ids,
#                     "domains": domains
#                 })
#         self.dataset = dataset

#     def __getitem__(self, index):
#         return self.dataset[index]

#     def __len__(self):
#         return len(self.dataset)

#     @staticmethod
#     def collate_fn(batch):
#         batch_dict: Dict[str, List[Tensor]] = default_collate(batch)
#         for k, v in batch_dict.items():
#             batch_dict[k] = cat(v)
#         return batch_dict


class ContrastDataset(Dataset):

    def __init__(self, filename: str, tokenizer: PreTrainedTokenizer, device=None) -> None:
        dataset = []
        total_lines = sum(1 for _ in open(filename, "rb"))
        with open(filename, "r") as f:
            for line in tqdm(f, total=total_lines, desc=filename):
                line = line.strip()
                input_ids, attention_mask, token_type_ids = [], [], []
                for item in line.split("####"):
                    text = item.rsplit("***")[0]
                    tok_dict: Dict[str, List[int]] = tokenizer(text,
                                                               padding=PaddingStrategy.MAX_LENGTH,
                                                               truncation=True)
                    input_ids.append(as_tensor(tok_dict.input_ids, device=device))
                    attention_mask.append(as_tensor(tok_dict.attention_mask, device=device))
                    token_type_ids.append(as_tensor(tok_dict.token_type_ids, device=device))
                dataset.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                })

        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_fn(batch):
        batch_dict: Dict[str, List[Tensor]] = default_collate(batch)
        for k, v in batch_dict.items():
            batch_dict[k] = cat(v)
        return batch_dict


class MMTDataset(Dataset):

    def __init__(self,
                 filename: str,
                 tokenizer: PreTrainedTokenizer,
                 device=None,
                 src=True) -> None:
        dataset = []
        total_lines = sum(1 for _ in open(filename, "rb"))
        pos_list = [item.split()[0] for item in open("./ann/laptop_pos.txt").read().splitlines()]
        deprel_list = [
            item.split()[0] for item in open("./ann/laptop_deprel.txt").read().splitlines()
        ]
        with open(filename, "r") as f:
            for line in tqdm(f, total=total_lines, desc=filename):
                line = line.strip()
                special_tokens = tokenizer.all_special_tokens
                input_ids, attention_mask, token_type_ids = [], [], []
                gold_label_list, pos_label_list, deprel_label_list, domains, hard_label_list = [], [], [], [], []
                for item in line.split("####"):
                    text, gold_labels, pos_labels, deprel_labels, hard_labels = item.rsplit("***")
                    tok_dict: Dict[str, List[int]] = tokenizer(text,
                                                               padding=PaddingStrategy.MAX_LENGTH,
                                                               truncation=True)
                    wordpiece_tokens = tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
                    glod_wordpiece_labels = transform(text, gold_labels, wordpiece_tokens,
                                                      special_tokens)
                    _labels = [
                        TAGS.index(label) if label in TAGS else -1
                        for label in glod_wordpiece_labels
                    ]
                    input_ids.append(as_tensor(tok_dict.input_ids, device=device))
                    gold_label_list.append(as_tensor(_labels.copy(), device=device))
                    labels = transform(text, hard_labels, wordpiece_tokens, special_tokens)
                    _labels = [TAGS.index(label) if label in TAGS else -1 for label in labels]
                    hard_label_list.append(as_tensor(_labels.copy(), device=device))
                    labels = transform(text, pos_labels, wordpiece_tokens, special_tokens)
                    _labels = []
                    for idx, label in enumerate(labels):
                        if label in pos_list:
                            _labels.append(pos_list.index(label))
                        else:
                            if glod_wordpiece_labels[idx] not in ["O", "SPECIAL_TOKEN"]:
                                _labels.append(pos_list.index(label[0] + '-[UNK]'))
                            else:
                                _labels.append(-1)
                    pos_label_list.append(as_tensor(_labels.copy(), device=device))
                    labels = transform(text, deprel_labels, wordpiece_tokens, special_tokens)
                    _labels.clear()
                    for idx, label in enumerate(labels):
                        if label in deprel_list:
                            _labels.append(deprel_list.index(label))
                        else:
                            if glod_wordpiece_labels[idx] not in ["O", "SPECIAL_TOKEN"]:
                                _labels.append(deprel_list.index(label[0] + '-[UNK]'))
                            else:
                                _labels.append(-1)
                    deprel_label_list.append(as_tensor(_labels.copy(), device=device))
                    attention_mask.append(as_tensor(tok_dict.attention_mask, device=device))
                    token_type_ids.append(as_tensor(tok_dict.token_type_ids, device=device))
                    # specified as a list for the benefit of tensor broadcasting
                    domains.append(as_tensor([src], device=device))
                dataset.append({
                    "input_ids": input_ids,
                    "gold_labels": gold_label_list,
                    "pos_labels": pos_label_list,
                    "deprel_labels": deprel_label_list,
                    "hard_labels": hard_label_list,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "domains": domains
                })
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_fn(batch):
        batch_dict: Dict[str, List[Tensor]] = default_collate(batch)
        for k, v in batch_dict.items():
            batch_dict[k] = cat(v)
        return batch_dict


if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=100)
    dataset = MMTDataset("./processed/dataset/rest.train.txt", tokenizer)
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, 16, True, collate_fn=dataset.collate_fn)
    for batch in dataloader:
        print(batch)