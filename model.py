import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, pretrained_model: str):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)

    
    def forward(self, batch):
        pass
