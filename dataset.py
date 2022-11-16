from typing import List, Dict

from torch import as_tensor
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.generic import PaddingStrategy

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
                tag = (f"I{labels[-1][1:]}" if labels[-1] not in ["O", "SPECIAL_TOKEN"] else "O")
            else:
                tag = tt[1]
                if tag != "O":
                    tag = (f"B{tag[1:]}" if labels[-1] in ["O", "SPECIAL_TOKEN"] else f"I{tag[1:]}")
            offset += len(token.replace("##", ""))
            if offset == len(tt[0]):
                i += 1
                offset = 0
        labels.append(tag)
    _labels = [TAGS.index(label) if label in TAGS else -1 for label in labels]
    return _labels


class BaseDataset(Dataset):

    def __init__(self, filename: str, tokenizer: PreTrainedTokenizer, device=None):
        data = []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                text, text_labels = line.split("***")[0:2]
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
                })
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MMTDataset(Dataset):

    def __init__(self, filename: str, tokenizer: PreTrainedTokenizer, device=None) -> None:
        dataset = []
        with open(filename, "r") as f:
            for line in f:
                data = {
                    "input_ids": [],
                    "gold_labels": [],
                    "dp_labels": [],
                    "attention_mask": [],
                    "token_type_ids": [],
                }
                line = line.strip()
                for item in line.split("####"):
                    text, gold_labels, hard_labels = item.split("***")
                    tok_dict: Dict[str, List[int]] = tokenizer(text,
                                                               padding=PaddingStrategy.MAX_LENGTH,
                                                               truncation=True)
                    wordpiece_tokens = tokenizer.convert_ids_to_tokens(tok_dict.input_ids)
                    special_tokens = tokenizer.all_special_tokens
                    labels = transform(text, gold_labels, wordpiece_tokens, special_tokens)
                    data["input_ids"].append(as_tensor(tok_dict.input_ids, device=device))
                    data["gold_labels"].append(as_tensor(labels.copy(), device=device))
                    labels = transform(text, hard_labels, wordpiece_tokens, special_tokens)
                    data["dp_labels"].append(as_tensor(labels.copy(), device=device))
                    data["attention_mask"].append(as_tensor(tok_dict.attention_mask, device=device))
                    data["token_type_ids"].append(as_tensor(tok_dict.token_type_ids, device=device))
                dataset.append(data)
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=100)
    dataset = MMTDataset("./processed/dataset/rest.train.txt", tokenizer)
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, 16, True)
    for batch in dataloader:
        print(batch)