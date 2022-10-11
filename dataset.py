from typing import List
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from settings import TAGS
from transformers.utils.generic import PaddingStrategy

class MyDataset(Dataset):
    def __init__(self, filename, tokenizer: PreTrainedTokenizer):
        data = []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                text, label_text = line.split("####")
                label_tuplelist = [tuple(_.rsplit("=", maxsplit=1)) for _ in label_text.split()]
                label_tuplelist_idx, inner_offset = 0, 0
                res = tokenizer(text, padding=PaddingStrategy.MAX_LENGTH, truncation=True)
                input_ids = res.input_ids
                labels: List[str] = []
                for token in tokenizer.convert_ids_to_tokens(input_ids):
                    if token in tokenizer.all_special_tokens:
                        tag = 'O'
                    else:
                        label_tuple = label_tuplelist[label_tuplelist_idx]
                        if token.startswith("##"):
                            tag = f'I{labels[-1][1:]}' if labels[-1] != "O" else 'O'
                        else:
                            tag = label_tuple[1]
                            if tag != 'O':
                                tag = f'B{tag[1:]}' if labels[-1] == "O" else f'I{tag[1:]}'
                        inner_offset += len(token.replace("##", ""))
                        if inner_offset == len(label_tuple[0]):
                            label_tuplelist_idx += 1
                            inner_offset = 0
                    labels.append(tag)
                try:
                    data.append(
                        {
                            "input_ids": input_ids, 
                            "label": [TAGS.index(label) for label in labels], 
                            "attention_mask": res.attention_mask
                        })
                except KeyError as e:
                    print(e)
        self.data = data
        


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
