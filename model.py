import logging

import torch
import torch.nn as nn
import torch.nn.functional as f
from pytorch_revgrad import RevGrad
from pytorch_transformers import BertModel, BertPreTrainedModel
from torch import Tensor
from transformers.modeling_outputs import TokenClassifierOutput

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
            total_loss = total_loss[active_loss]
            loss = total_loss.mean(0).nan_to_num(0).sum()
        else:
            loss = total_loss.mean(0).nan_to_num(0).sum()

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


__all__ = [
    'PretrainedBertForTokenClassification', 'BertForTokenClassification', 'MMTModel', 'DomainModel',
    'ContrastModel'
]


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
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self,
                input_ids: Tensor,
                token_type_ids: Tensor = None,
                attention_mask: Tensor = None,
                gold_labels: Tensor = None,
                **kwargs):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # attention_mask size (batch, seq_len)
        loss = None
        if gold_labels is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = gold_labels.view(-1)[active_loss]
                loss = self.loss_fct(active_logits, active_labels)
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), gold_labels.view(-1))
        return TokenClassifierOutput(logits=logits, loss=loss, hidden_states=sequence_output)


class PretrainedBertForTokenClassification(BertPreTrainedModel):

    def __init__(self, config):
        config.output_hidden_states = True
        super(PretrainedBertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.pos = [(tag, float(weight)) for item in open("./ann/rest_pos.txt").read().splitlines()
                    for tag, weight in (item.split(), )]
        self.deprel = [(tag, float(weight))
                       for item in open("./ann/rest_deprel.txt").read().splitlines()
                       for tag, weight in (item.split(), )]
        self.pos_project = nn.Linear(config.hidden_size, len(self.pos))
        self.deprel_project = nn.Linear(config.hidden_size, len(self.deprel))

    def forward(self,
                input_ids: Tensor,
                token_type_ids: Tensor = None,
                attention_mask: Tensor = None,
                gold_labels: Tensor = None,
                pos_labels: Tensor = None,
                deprel_labels: Tensor = None,
                **kwargs):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        # half_num = int(self.config.num_hidden_layers / 2) + 1
        # sequence_output = 0.5*torch.stack(outputs[2][1:]).mean(0) + 0.5*torch.stack(
        #     outputs[2][-1:]).mean(0)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # attention_mask size (batch, seq_len)
        loss = None
        if gold_labels is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = gold_labels.view(-1)[active_loss]
                main_loss = self.loss_fct(active_logits, active_labels)
                pos_logits = self.pos_project(sequence_output)
                active_logits = pos_logits.view(-1, len(self.pos))[active_loss]
                # active_logits.add_(
                #     torch.log(torch.Tensor([weight
                #                             for _, weight in self.pos]).pow(2.0).add(1e-12)).cuda())
                active_labels = pos_labels.view(-1)[active_loss]
                pos_loss = self.loss_fct(active_logits, active_labels)
                deprel_logits = self.deprel_project(sequence_output)
                active_logits = deprel_logits.view(-1, len(self.deprel))[active_loss]
                # active_logits.add_(
                #     torch.log(
                #         torch.Tensor([weight
                #                       for _, weight in self.deprel]).pow(2.0).add(1e-12)).cuda())
                active_labels = deprel_labels.view(-1)[active_loss]
                deprel_loss = self.loss_fct(active_logits, active_labels)
                aux_loss = pos_loss + deprel_loss
                logger.debug(f"{main_loss}, {pos_loss}, {deprel_loss}")
                loss = 0.0 * aux_loss
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), gold_labels.view(-1))
        return TokenClassifierOutput(logits=logits, loss=loss, hidden_states=sequence_output)


class ContrastModel(nn.Module):

    def __init__(self,
                 model_1: BertForTokenClassification,
                 model_2: BertForTokenClassification,
                 K=256,
                 m: float = 0.999,
                 dim: int = 128):
        super(MMTModel, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.K = K
        self.m = m
        self.T = 0.07
        # create the encoders
        self.encoder_q = nn.Sequential(nn.ReLU(), nn.Linear(self.model_1.config.hidden_size, dim))
        self.encoder_k = nn.Sequential(nn.ReLU(), nn.Linear(self.model_1.config.hidden_size, dim))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor = None,
        attention_mask: Tensor = None,
    ):
        input_ids_1, input_ids_2 = torch.chunk(input_ids, 2)
        token_type_ids_1, token_type_ids_2 = torch.chunk(token_type_ids, 2)
        attention_mask_1, attention_mask_2 = torch.chunk(attention_mask, 2)
        outputs1: TokenClassifierOutput = self.model_1(input_ids_1, token_type_ids_1,
                                                       attention_mask_1)
        outputs2: TokenClassifierOutput = self.model_2(input_ids_2, token_type_ids_2,
                                                       attention_mask_2)
        # encode the sentence representation
        q = self.encoder_q(torch.mean(outputs1.hidden_states, dim=1))
        q = nn.functional.normalize(q, dim=1)  # NxC
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            # encode the disturbance/augmented sencente representation
            k = self.encoder_k(torch.mean(outputs2.hidden_states, dim=1))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        loss = self.ce_loss(logits, labels)
        return TokenClassifierOutput(loss, logits)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


class MMTModel(nn.Module):

    def __init__(self,
                 model_1: BertForTokenClassification,
                 model_1_ema: BertForTokenClassification,
                 model_2: BertForTokenClassification,
                 model_2_ema: BertForTokenClassification,
                 alpha: float = 0.999,
                 soft_loss_weight: float = 1,
                 domain_loss_weight: float = 0.1):
        super(MMTModel, self).__init__()
        self.model_1 = model_1
        self.model_ema_1 = model_1_ema
        self.model_2 = model_2
        self.model_ema_2 = model_2_ema
        self.alpha = alpha
        self.soft_loss_weight = soft_loss_weight
        self.domain_loss_weight = domain_loss_weight
        self.domain_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.ce_soft_loss = SoftEntropy()

    def forward(self,
                input_ids: Tensor,
                token_type_ids: Tensor = None,
                attention_mask: Tensor = None,
                gold_labels: Tensor = None,
                pos_labels: Tensor = None,
                deprel_labels: Tensor = None,
                hard_labels: Tensor = None,
                domains: Tensor = None,
                **kwargs):
        # domain_mask = torch.broadcast_to(domains, input_ids.shape).contiguous()
        if input_ids.shape[0] == 1:
            outputs: TokenClassifierOutput = self.model_1(input_ids,
                                                          token_type_ids,
                                                          attention_mask,
                                                          gold_labels=gold_labels,
                                                          pos_labels=pos_labels,
                                                          deprel_labels=deprel_labels)
            return TokenClassifierOutput(logits=outputs.logits, loss=outputs.loss)
        # domains_mask_1, domains_mask_2 = torch.chunk(domain_mask, 2)
        input_ids_1, input_ids_2 = torch.chunk(input_ids, 2)
        token_type_ids_1, token_type_ids_2 = torch.chunk(token_type_ids, 2)
        attention_mask_1, attention_mask_2 = torch.chunk(attention_mask, 2)

        if self.training:
            # exponential moving average
            self.__update_ema_variables(self.model_1, self.model_ema_1, self.alpha)
            self.__update_ema_variables(self.model_2, self.model_ema_2, self.alpha)
            # Consider target domain only
            # For target domain, supervised with double propagation labels
            # and auxiliary tasks(pos, dep)
            labels = torch.where(domains, gold_labels, hard_labels)
            labels_1, labels_2 = tuple(
                label
                for label, mask in zip(torch.chunk(labels, 2), torch.chunk(domains.squeeze(-1), 2)))
            pos_labels_1, pos_labels_2 = tuple(label for label, mask in zip(
                torch.chunk(pos_labels, 2), torch.chunk(domains.squeeze(-1), 2)))
            deprel_labels_1, deprel_labels_2 = tuple(label for label, mask in zip(
                torch.chunk(deprel_labels, 2), torch.chunk(domains.squeeze(-1), 2)))
        else:
            labels_1, labels_2 = None, None
            pos_labels_1, pos_labels_2 = None, None
            deprel_labels_1, deprel_labels_2 = None, None
        outputs1: TokenClassifierOutput = self.model_1(input_ids_1,
                                                       token_type_ids_1,
                                                       attention_mask_1,
                                                       gold_labels=labels_1,
                                                       pos_labels=pos_labels_1,
                                                       deprel_labels=deprel_labels_1)
        outputs1_ema: TokenClassifierOutput = self.model_ema_1(input_ids_2, token_type_ids_2,
                                                               attention_mask_2)
        outputs2: TokenClassifierOutput = self.model_2(input_ids_2,
                                                       token_type_ids_2,
                                                       attention_mask_2,
                                                       gold_labels=labels_2,
                                                       pos_labels=pos_labels_2,
                                                       deprel_labels=deprel_labels_2)
        outputs2_ema: TokenClassifierOutput = self.model_ema_2(input_ids_1, token_type_ids_1,
                                                               attention_mask_1)
        loss = None
        if self.training:
            # hard label cross entropy loss
            loss_ce = outputs1.loss + outputs2.loss
            # soft label loss
            loss_ce_soft = self.ce_soft_loss(
                outputs1.logits, outputs2_ema.logits, attention_mask_1) + self.ce_soft_loss(
                    outputs2.logits, outputs1_ema.logits, attention_mask_2)
            # total loss
            loss = loss_ce + loss_ce_soft * self.soft_loss_weight  # + d_loss * self.domain_loss_weight
            logger.debug("loss_ce: %f, loss_ce_soft: %f, loss: %f", loss_ce.item(),
                         loss_ce_soft.item(), loss.item())
        return TokenClassifierOutput(logits=torch.cat([outputs1.logits, outputs2.logits]),
                                     loss=loss)

    @torch.no_grad()
    def __update_ema_variables(self, model, ema_model, alpha):
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

    # domain loss for target domain and source domain
    # hidden_states = self.model_1(input_ids_1, token_type_ids_1,
    #                              attention_mask_1).hidden_states
    # domain_pre_1, _ = self.dom_model(hidden_states)
    # d_loss_1 = self.domain_loss(domain_pre_1, domains_mask_1[:, 0].float())
    # hidden_states = self.model_1(input_ids_2, token_type_ids_2,
    #                              attention_mask_2).hidden_states
    # domain_pre_2, _ = self.dom_model(hidden_states)
    # d_loss_2 = self.domain_loss(domain_pre_2, domains_mask_2[:, 0].float())
    # d_loss = (d_loss_1 + d_loss_2) / 2
