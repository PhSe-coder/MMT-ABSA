from dataclasses import dataclass

import torch
import torch.nn as nn
from pytorch_revgrad import RevGrad
from pytorch_transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

__all__ = ['BertForTokenClassification', 'MMTModel', 'DomainModel']

class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # attention_mask size (batch, seq_len)
        return TokenClassifierOutput(
            logits=logits
        )

@dataclass
class MMTModel(nn.Module):
    model_1: BertForTokenClassification
    model_ema_1: BertForTokenClassification
    model_2: BertForTokenClassification
    model_ema_2: BertForTokenClassification
    
    def __post_init__(self):
        super(MMTModel, self).__init__()
        
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs1 = self.model_1(input_ids, token_type_ids, attention_mask)
        outputs1_ema = self.model_ema_1(input_ids, token_type_ids, attention_mask)
        outputs2 = self.model_2(input_ids, token_type_ids, attention_mask)
        outputs2_ema = self.model_ema_2(input_ids, token_type_ids, attention_mask)
        
        
        

class DomainModel(nn.Module):
    def __init__(self, dim, adv):
        super(DomainModel, self).__init__()
        out_dim = 100
        self.project = nn.Sequential(nn.Tanh(),
                                     nn.Linear(dim, out_dim),
                                     nn.Linear(out_dim, 1),
                                     nn.Softmax(dim=1))
        self.full_layer = nn.Linear(dim, 1)
        self.grl = RevGrad(adv)

    def forward(self, inputs):
        temp = self.grl(inputs)
        alphas = self.project(temp)
        x = torch.sum(temp * alphas, dim=1)
        final_out = self.full_layer(x)
        return final_out.squeeze(1), alphas