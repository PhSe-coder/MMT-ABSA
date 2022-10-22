import torch.nn as nn
from transformers import BertModel, BertConfig, BertForTokenClassification

class Model(nn.Module):
    def __init__(self, pretrained_model: str):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.config = BertConfig.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, 7)

    
    def forward(self, batch):
        batch = {k: v.to("cpu") for k, v in batch.items()}
        outputs = self.bert(**batch)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits