import torch
import torch.nn as nn
import torch.nn.functional as f
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


class MSE(nn.Module):

    def __init__(self):
        super(MSE, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.softmax = nn.Softmax(-1)

    def forward(self, inputs, targets, attention_mask: Tensor = None):
        num_labels = inputs.shape[-1]
        input_softmax = self.softmax(inputs)
        target_softmax = self.softmax(targets)
        loss = self.mse_loss(input_softmax, target_softmax).view(-1, num_labels)
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            loss = loss[active_loss].mean(0).sum()
        else:
            loss = loss.mean(0).sum()
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
        config.output_hidden_states = True
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
                gold_labels: Tensor = None,
                hard_label_weight: float = 1.0,
                domain_mask: Tensor = None,
                **kwargs):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        # half_num = int(self.config.num_hidden_layers / 2)
        # sequence_output = 0.6*torch.stack(outputs[2][1:half_num]).mean(0) + 0.4*torch.stack(
        #     outputs[2][half_num:]).mean(0)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # attention_mask size (batch, seq_len)
        loss = None
        if gold_labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = gold_labels.view(-1)[active_loss]
                loss = self.loss_fct(active_logits, active_labels)
                if domain_mask is not None:
                    src_domain_loss = loss[domain_mask.view(-1)[active_loss] ==
                                           True].mean().nan_to_num(0)
                    tar_domain_loss = loss[domain_mask.view(-1)[active_loss] ==
                                           False].mean().nan_to_num(0)
                    loss = src_domain_loss + hard_label_weight * tar_domain_loss
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), gold_labels.view(-1))
                if domain_mask is not None:
                    src_domain_loss = loss[domain_mask.view(-1) == True].mean()
                    tar_domain_loss = loss[domain_mask.view(-1) == False].mean()
                    loss = src_domain_loss + hard_label_weight * tar_domain_loss
        return TokenClassifierOutput(logits=logits, loss=loss, hidden_states=sequence_output)

    def post_operation(self, *args, **kwargs):
        pass


def softmax_mse_loss_weight(input_logits,
                            target_logits,
                            weight,
                            word_seq,
                            attention_mask=None,
                            thr1=0.6,
                            thr2=0.9,
                            thr3=0.4):
    """Compute the weighted squared loss
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = f.softmax(input_logits, dim=-1)
    target_softmax = f.softmax(target_logits, dim=-1)
    num_classes = input_logits.size()[-1]
    # print('number_classes',num_classes)
    pro = torch.max(target_softmax, dim=-1, keepdim=True)

    o_mask = (pro[1] == 0).float()
    # print('before pad o tag:', torch.sum(o_mask))
    other_mask = (pro[1] != 0).float()
    # print('before pad other tag:', torch.sum(other_mask))

    total_word_number = 0
    # print('contain pad word number:', word_seq.shape[0]*word_seq.shape[1])
    if attention_mask is not None:
        pad_mask = (attention_mask != 0).float()
        total_word_number = torch.sum(pad_mask)
        # print('+++word_total number:', torch.sum(pad_mask))
        o_mask = o_mask * pad_mask.unsqueeze(2)
        # print('after pad:', torch.sum(o_mask))
        other_mask = other_mask * pad_mask.unsqueeze(2)
        # print('after pad other:', torch.sum(other_mask))

    o_number = torch.sum(o_mask)
    o_top_number = int(thr1 * o_number)

    # print('o_top_number:', o_top_number)
    # print('o_number:', o_number)
    other_number = torch.sum(other_mask)
    other_top_number = int(thr1 * other_number)
    #o_top_number = 15 * other_top_number
    # print('other_top_number:', other_top_number)
    # print('other_number:', other_number)
    ##print('o_mask:', o_mask)
    ##print('other_mask:', other_mask)
    sort_o, _ = torch.topk((o_mask * pro[0]).view(-1), o_top_number)
    sort_other, _ = torch.topk((other_mask * pro[0]).view(-1), other_top_number)

    if sort_o.shape[0] == 0:
        fix_o_thr = 0
    else:
        fix_o_thr = sort_o[-1]
    if sort_other.shape[0] == 0:
        fix_other_thr = 0
    else:
        fix_other_thr = sort_other[-1]

    #fix_o_thr = torch.sum(o_mask*pro[0]) / torch.sum(o_mask)
    #fix_other_thr = torch.sum(other_mask*pro[0]) / (torch.sum(other_mask)+1e-10)
    if fix_o_thr < thr2:
        fix_o_thr = thr2
    if fix_other_thr < thr3:
        fix_other_thr = thr3

    # print('o_thr:', fix_o_thr)
    # print('other_thr:', fix_other_thr)
    thr_mask = (pro[0] > fix_o_thr).float()
    thr_mask_1 = (pro[0] > fix_other_thr).float()
    #sum_num = torch.sum(thr_mask)
    ##print('thr_mask:', thr_mask)
    ##print('pro[0]:', pro[0]*thr_mask)

    #weight_mask = (o_mask*thr_mask+other_mask*thr_mask_1)
    o_mat = o_mask * thr_mask
    other_mat = other_mask * thr_mask_1
    # print('O tag number:', torch.sum(o_mat))
    # print('Other tag number:', torch.sum(other_mat))
    weight_mask = o_mat + other_mat

    loss = weight.unsqueeze(1).unsqueeze(2) * weight_mask * ((input_softmax - target_softmax)**2)
    loss = torch.mean(loss.view(-1)) * num_classes
    #loss = torch.sum(loss)/total_word_number
    return loss


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
        domains_1, domains_2 = torch.chunk(domains, 2)
        torch.broadcast_to(domains_1, input_ids_1.shape)
        domain_mask = torch.broadcast_to(domains_1, input_ids_1.shape).contiguous()
        outputs1: TokenClassifierOutput = self.model_1(
            input_ids_1,
            token_type_ids_1,
            attention_mask_1,
            gold_labels=labels_1,
            hard_label_weight=self.hard_label_loss_weight,
            domain_mask=domain_mask)
        outputs1_ema: TokenClassifierOutput = self.model_ema_1(input_ids_2, token_type_ids_2,
                                                               attention_mask_2)
        domain_mask = torch.broadcast_to(domains_2, input_ids_2.shape).contiguous()
        outputs2: TokenClassifierOutput = self.model_2(
            input_ids_2,
            token_type_ids_2,
            attention_mask_2,
            gold_labels=labels_2,
            hard_label_weight=self.hard_label_loss_weight,
            domain_mask=domain_mask)
        outputs2_ema: TokenClassifierOutput = self.model_ema_2(input_ids_1, token_type_ids_1,
                                                               attention_mask_1)
        loss_ce = outputs1.loss + outputs2.loss
        domain_pre_1, _ = self.dom_model(outputs1.hidden_states)
        d_loss_1 = self.domain_loss(domain_pre_1, domains_1.squeeze(-1).float())
        domain_pre_2, _ = self.dom_model(outputs2.hidden_states)
        d_loss_2 = self.domain_loss(domain_pre_2, domains_2.squeeze(-1).float())
        d_loss = d_loss_1 + d_loss_2
        loss_ce_soft = self.ce_soft_loss(outputs1.logits, outputs2_ema.logits,
                                         attention_mask_1) + self.ce_soft_loss(
                                             outputs2.logits, outputs1_ema.logits, attention_mask_2)
        # domains = domains.squeeze(-1)
        # print(domains)
        # weights = torch.sigmoid(torch.cat([domain_pre_1, domain_pre_2])[domains==0])
        # loss_ce_soft = softmax_mse_loss_weight(torch.cat([outputs1.logits, outputs2.logits])[domains==0], torch.cat([outputs2_ema.logits, outputs1_ema.logits])[domains==0], weights, input_ids[domains==0], attention_mask[domains==0])

        loss = loss_ce + loss_ce_soft * self.ce_soft_weight + d_loss * self.domain_loss_weight
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
