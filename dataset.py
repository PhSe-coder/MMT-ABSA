from typing import List, Dict

from torch import as_tensor, cat
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.generic import PaddingStrategy
from torch.utils.data.dataloader import default_collate

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
        data = []
        total_lines = sum(1 for _ in open(filename, "rb"))
        with open(filename, "r") as f:
            for line in tqdm(f, total=total_lines, desc=filename):
                line = line.strip()
                text, text_labels = line.rsplit("***")[0:2]
                tok_dict: Dict[str, List[int]] = tokenizer(text,
                                                           padding=PaddingStrategy.MAX_LENGTH,
                                                           truncation=True)
                wordpiece_tokens = tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
                special_tokens = tokenizer.all_special_tokens
                labels = transform(text, text_labels, wordpiece_tokens, special_tokens)
                data.append({
                    "input_ids": as_tensor(tok_dict.input_ids, device=device),
                    "gold_labels": as_tensor(labels, device=device),
                    "attention_mask": as_tensor(tok_dict.attention_mask, device=device),
                    "token_type_ids": as_tensor(tok_dict.token_type_ids, device=device),
                    "domains": as_tensor([src], device=device)
                })
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MMTDataset(Dataset):

    def __init__(self,
                 filename: str,
                 tokenizer: PreTrainedTokenizer,
                 device=None,
                 src=True) -> None:
        dataset = []
        total_lines = sum(1 for _ in open(filename, "rb"))
        with open(filename, "r") as f:
            for line in tqdm(f, total=total_lines, desc=filename):
                line = line.strip()
                special_tokens = tokenizer.all_special_tokens
                input_ids, attention_mask, token_type_ids = [], [], []
                gold_label_list, dp_labels, domains = [], [], []
                for item in line.split("####"):
                    text, gold_labels, hard_labels = item.rsplit("***")
                    tok_dict: Dict[str, List[int]] = tokenizer(text,
                                                               padding=PaddingStrategy.MAX_LENGTH,
                                                               truncation=True)
                    wordpiece_tokens = tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
                    labels = transform(text, gold_labels, wordpiece_tokens, special_tokens)
                    input_ids.append(as_tensor(tok_dict.input_ids, device=device))
                    _labels = [TAGS.index(label) if label in TAGS else -1 for label in labels]
                    gold_label_list.append(as_tensor(_labels.copy(), device=device))
                    labels = transform(text, hard_labels, wordpiece_tokens, special_tokens)
                    _labels = [TAGS.index(label) if label in TAGS else -1 for label in labels]
                    dp_labels.append(as_tensor(_labels.copy(), device=device))
                    attention_mask.append(as_tensor(tok_dict.attention_mask, device=device))
                    token_type_ids.append(as_tensor(tok_dict.token_type_ids, device=device))
                    # specified as a list for the benefit of tensor broadcasting
                    domains.append(as_tensor([src], device=device))
                dataset.append({
                    "input_ids": input_ids,
                    "gold_labels": gold_label_list,
                    "hard_labels": dp_labels,
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


class MMTDataset1(Dataset):

    def __init__(self,
                 filename: str,
                 tokenizer: PreTrainedTokenizer,
                 device=None,
                 src=True) -> None:
        dataset = []
        total_lines = sum(1 for _ in open(filename, "rb"))
        with open(filename, "r") as f:
            for line in tqdm(f, total=total_lines, desc=filename):
                line = line.strip()
                special_tokens = tokenizer.all_special_tokens
                input_ids, attention_mask, token_type_ids = [], [], []
                gold_label_list, pos_label_list, deprel_label_list, domains = [], [], [], []
                for item in line.split("####"):
                    text, gold_labels, pos_labels, deprel_labels= item.rsplit("***")
                    tok_dict: Dict[str, List[int]] = tokenizer(text,
                                                               padding=PaddingStrategy.MAX_LENGTH,
                                                               truncation=True)
                    wordpiece_tokens = tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
                    labels = transform(text, gold_labels, wordpiece_tokens, special_tokens)
                    input_ids.append(as_tensor(tok_dict.input_ids, device=device))
                    gold_label_list.append(as_tensor(labels.copy(), device=device))
                    labels = transform(text, pos_labels, wordpiece_tokens, special_tokens)
                    _labels = [TAGS.index(label) if label in TAGS else -1 for label in labels]
                    pos_label_list.append(as_tensor(_labels.copy(), device=device))
                    labels = transform(text, deprel_labels, wordpiece_tokens, special_tokens)
                    deprel_label_list.append(as_tensor(labels.copy(), device=device))
                    attention_mask.append(as_tensor(tok_dict.attention_mask, device=device))
                    token_type_ids.append(as_tensor(tok_dict.token_type_ids, device=device))
                    # specified as a list for the benefit of tensor broadcasting
                    domains.append(as_tensor([src], device=device))
                dataset.append({
                    "input_ids": input_ids,
                    "gold_labels": gold_label_list,
                    "pos_labels": pos_label_list,
                    "deprel_labels": deprel_label_list,
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
    dataset = MMTDataset("./processed/dataset/rest.validation.txt", tokenizer)
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, 16, True, collate_fn=dataset.collate_fn)
    for batch in dataloader:
        print(batch)