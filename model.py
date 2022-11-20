import torch
import torch.nn as nn
from torch import Tensor
from pytorch_revgrad import RevGrad
from pytorch_transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

import logging
logger = logging.getLogger(__name__)


class SoftEntropy(nn.Module):

    def __init__(self):
        super(SoftEntropy, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, targets, attention_mask: Tensor = None):
        log_probs = self.logsoftmax(inputs)
        num_labels = inputs.shape[-1]
        total_loss = (-self.softmax(targets) * log_probs).view(-1, num_labels)
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            loss = total_loss[active_loss].mean(0).sum()
        else:
            loss = total_loss.mean(0).sum()
        return loss


__all__ = ['BertForTokenClassification', 'MMTModel', 'DomainModel']


class DomainModel(nn.Module):

    def __init__(self, dim: int, adv: float = 1.0):
        super(DomainModel, self).__init__()
        out_dim = int(dim / 2)
        self.project = nn.Sequential(nn.Linear(dim, out_dim), nn.Tanh(), nn.Linear(out_dim, 1),
                                     nn.Softmax(dim=1))
        self.full_layer = nn.Linear(dim, 1)
        self.grl = RevGrad(adv)

    def forward(self, inputs):
        temp = self.grl(inputs)
        alphas = self.project(temp)
        x = torch.sum(temp * alphas, dim=1)
        final_out = self.full_layer(x)
        return final_out.squeeze(-1), alphas


class BertForTokenClassification(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    def forward(self,
                input_ids: Tensor,
                token_type_ids: Tensor = None,
                attention_mask: Tensor = None,
                **kwargs):
        labels: Tensor = kwargs.get("gold_labels")
        token_weights: Tensor = kwargs.get("token_weights")
        if token_weights is not None:
            assert token_weights.shape == input_ids.shape
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # attention_mask size (batch, seq_len)
        loss = None
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = self.loss_fct(active_logits, active_labels)
                if token_weights is not None:
                    loss = token_weights.view(-1)[active_loss].mul(loss).mean()
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if token_weights is not None:
                    loss = token_weights.view(-1).mul(loss).mean()
        return TokenClassifierOutput(logits=logits, loss=loss, hidden_states=sequence_output)

    def post_operation(self, *args, **kwargs):
        pass


class MMTModel(nn.Module):

    def __init__(self,
                 model_1: BertForTokenClassification,
                 model_1_ema: BertForTokenClassification,
                 model_2: BertForTokenClassification,
                 model_2_ema: BertForTokenClassification,
                 dom_model: DomainModel,
                 alpha: float = 0.999,
                 ce_soft_weight: float = 0.5,
                 domain_loss_weight: float = 0.1,
                 hard_label_loss_weight: float = 0.5,
                 domain_loss=nn.BCEWithLogitsLoss(),
                 ce_loss=nn.CrossEntropyLoss(ignore_index=-1),
                 ce_soft_loss=SoftEntropy()):
        super(MMTModel, self).__init__()
        self.model_1 = model_1
        self.model_ema_1 = model_1_ema
        self.model_2 = model_2
        self.model_ema_2 = model_2_ema
        self.dom_model = dom_model
        self.hard_label_loss_weight = hard_label_loss_weight
        self.alpha = alpha
        self.ce_soft_weight = ce_soft_weight
        self.domain_loss_weight = domain_loss_weight
        self.domain_loss = domain_loss
        self.ce_loss = ce_loss
        self.ce_soft_loss = ce_soft_loss

    def forward(self,
                input_ids: Tensor,
                token_type_ids: Tensor = None,
                attention_mask: Tensor = None,
                **kwargs):
        gold_labels = kwargs.get("gold_labels")
        hard_labels = kwargs.get("hard_labels", gold_labels)
        domains = kwargs.get("domains")
        input_ids_1, input_ids_2 = torch.chunk(input_ids, 2)
        token_type_ids_1, token_type_ids_2 = torch.chunk(token_type_ids, 2)
        attention_mask_1, attention_mask_2 = torch.chunk(attention_mask, 2)
        # for source domain, supervised with gold labels
        # for target domain, supervised with hard labels generated by double propagation
        labels = torch.where(domains, gold_labels, hard_labels)
        labels_1, labels_2 = torch.chunk(labels, 2)
        domains_1, domains_2 = torch.chunk(torch.as_tensor(domains.squeeze(-1), dtype=float), 2)
        token_weights = torch.broadcast_to(
            torch.where(domains_1 == 0, self.hard_label_loss_weight, domains_1).unsqueeze(-1),
            input_ids_1.shape).contiguous()
        outputs1: TokenClassifierOutput = self.model_1(input_ids_1,
                                                       token_type_ids_1,
                                                       attention_mask_1,
                                                       gold_labels=labels_1,
                                                       token_weights=token_weights)
        outputs1_ema: TokenClassifierOutput = self.model_ema_1(input_ids_2, token_type_ids_2,
                                                               attention_mask_2)
        token_weights = torch.broadcast_to(
            torch.where(domains_2 == 0, self.hard_label_loss_weight, domains_2).unsqueeze(-1),
            input_ids_2.shape).contiguous()
        outputs2: TokenClassifierOutput = self.model_2(input_ids_2,
                                                       token_type_ids_2,
                                                       attention_mask_2,
                                                       gold_labels=labels_2,
                                                       token_weights=token_weights)
        outputs2_ema: TokenClassifierOutput = self.model_ema_2(input_ids_1, token_type_ids_1,
                                                               attention_mask_1)
        loss_ce = outputs1.loss + outputs2.loss
        domain_pre, _ = self.dom_model(outputs1.hidden_states)
        d_loss_1 = self.domain_loss(domain_pre, domains_1)
        domain_pre, _ = self.dom_model(outputs2.hidden_states)
        d_loss_2 = self.domain_loss(domain_pre, domains_2)
        d_loss = d_loss_1 + d_loss_2
        loss_ce_soft = self.ce_soft_loss(outputs1.logits, outputs2_ema.logits,
                                         attention_mask_1) + self.ce_soft_loss(
                                             outputs2.logits, outputs1_ema.logits, attention_mask_2)

        loss = loss_ce * (1 - self.ce_soft_weight) + \
             loss_ce_soft * self.ce_soft_weight + d_loss * self.domain_loss_weight
        logger.info("loss_ce: %f, loss_ce_soft: %f, d_loss: %f, loss: %f", loss_ce.item(),
                    loss_ce_soft.item(), d_loss.item(), loss.item())
        return TokenClassifierOutput(logits=torch.cat([outputs1.logits, outputs2.logits]),
                                     loss=loss)

    def post_operation(self, *args, **kwargs):
        global_step = kwargs.get("global_step")
        self.__update_ema_variables(self.model_1, self.model_ema_1, self.alpha, global_step)
        self.__update_ema_variables(self.model_2, self.model_ema_2, self.alpha, global_step)

    def __update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
