import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertModel, BertPreTrainedModel
from torch import Tensor
from transformers.modeling_outputs import TokenClassifierOutput
from mi_estimators import InfoNCE
import torch.utils.checkpoint as cp

logger = logging.getLogger(__name__)

__all__ = ['PretrainedBertForTokenClassification', 'MIBert', 'MMTModel']

class MILoss(nn.Module):

    def __init__(self, threshold=0.5):
        super(MILoss, self).__init__()
        self.mi_threshold = threshold

    def forward(self, p: Tensor, log_p: Tensor):
        condi_entropy = -torch.sum(p * log_p, dim=-1).mean()
        y_dis = torch.mean(p, dim=0)
        y_entropy = -torch.sum(y_dis * torch.log(y_dis), dim=-1)
        return -y_entropy + condi_entropy


class SoftEntropy(nn.Module):

    def __init__(self):
        super(SoftEntropy, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        num_labels = inputs.shape[-1]
        total_loss = (-self.softmax(targets) * log_probs).view(-1, num_labels)
        loss = total_loss.mean()
        return loss


class MIBert(BertPreTrainedModel):

    def __init__(self, config, alpha, tau):
        super(MIBert, self).__init__(config)
        self.bert = BertModel(config)
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("tau", torch.tensor(tau))
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(0.1)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        # self.mi_loss = MILoss()
        self.mi_loss = InfoNCE(config.hidden_size, config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,
                batch_tgt: Dict[str, Tensor],
                batch_src: Dict[str, Tensor] = None,
                student: bool = True):
        loss, ce_loss, _mi_loss = None, None, 0
        if self.training:
            input_ids = torch.cat([batch_src['input_ids'], batch_tgt['input_ids']])
            token_type_ids = torch.cat([batch_src['token_type_ids'], batch_tgt['token_type_ids']])
            attention_mask = torch.cat([batch_src['attention_mask'], batch_tgt['attention_mask']])
            # valid_mask = torch.cat([batch_src['valid_mask'], batch_tgt['valid_mask']])
            sequence_output = self.bert(input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask)[0]
            sequence_output = self.dropout(sequence_output)
            logits: Tensor = self.classifier(sequence_output) / self.tau
            src_logits, tgt_logits = torch.chunk(logits, 2)
            _, tgt_outputs = torch.chunk(sequence_output, 2)
            active_mask = batch_tgt['attention_mask'].view(-1) == 1
            active_tgt_logits = tgt_logits.view(-1, self.num_labels)[active_mask]
            hidden_states = tgt_outputs.view(-1, tgt_outputs.size(-1))[active_mask]
            if student:
                gold_labels = batch_src['gold_labels']
                ce_loss = self.loss_fct(src_logits.view(-1, src_logits.size(-1)),
                                        gold_labels.view(-1))
                active_logits = logits.view(-1, logits.size(-1))[attention_mask.view(-1) == 1]
                # use checkpoint to avoid cuda out of memory although making the running slower.
                _mi_loss = cp.checkpoint(
                    self.mi_loss.learning_loss,
                    sequence_output.view(-1,
                                         sequence_output.size(-1))[attention_mask.view(-1) == 1],
                    f.softmax(active_logits, dim=-1))
                # p = f.softmax(active_logits, dim=-1)
                # log_p = f.log_softmax(active_logits, dim=-1)
                # _mi_loss = self.mi_loss(p, log_p)
                # _mi_loss = self.mi_loss.learning_loss(
                #     sequence_output.view(-1, sequence_output.size(-1))[valid_mask.view(-1) == 1],
                #     f.softmax(active_logits, dim=-1))
                loss = ce_loss + self.alpha * _mi_loss
        else:
            input_ids = batch_tgt['input_ids']
            attention_mask = batch_tgt['attention_mask']
            token_type_ids = batch_tgt['token_type_ids']
            valid_mask = batch_tgt['valid_mask']
            sequence_output = self.bert(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)[0]
            active_mask = valid_mask.view(-1) == 1
            tgt_logits: Tensor = self.classifier(sequence_output)
            active_tgt_logits = tgt_logits.view(-1, self.num_labels)[active_mask]
            hidden_states = sequence_output.view(-1, sequence_output.size(-1))[active_mask]
        return TokenClassifierOutput(logits=active_tgt_logits,
                                     loss=(loss, self.alpha * _mi_loss, ce_loss),
                                     hidden_states=hidden_states)


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
                valid_mask: Tensor = None,
                gold_labels: Tensor = None,
                **kwargs):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if gold_labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), gold_labels.view(-1))
        active_mask = valid_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_mask]
        hidden_states = sequence_output.view(-1, sequence_output.size(-1))[active_mask]
        return TokenClassifierOutput(logits=active_logits, loss=loss, hidden_states=hidden_states)


class ContrastModel(nn.Module):

    def __init__(self, model_1: MIBert, model_2: MIBert, K=65535, m: float = 0.999, dim: int = 128):
        super(ContrastModel, self).__init__()
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
        self.register_buffer("queue", torch.randn(self.model_1.config.hidden_size, K))
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
        q = torch.mean(outputs1.hidden_states, dim=1)
        q = nn.functional.normalize(q, dim=1)  # NxC
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            # encode the disturbance/augmented sencente representation
            k = torch.mean(outputs2.hidden_states, dim=1)  # keys: NxC
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
                 model_1: MIBert,
                 model_1_ema: MIBert,
                 model_2: MIBert,
                 model_2_ema: MIBert,
                 alpha: float = 0.999,
                 soft_loss_weight: float = 0):
        super(MMTModel, self).__init__()
        self.model_1 = model_1
        self.model_ema_1 = model_1_ema
        self.model_2 = model_2
        self.model_ema_2 = model_2_ema
        self.alpha = alpha
        self.soft_loss_weight = soft_loss_weight
        self.ce_soft_loss = SoftEntropy()

    def forward(self, batch):
        loss = None
        if self.training:
            outputs1: TokenClassifierOutput = self.model_1(batch[2], batch[0])
            outputs1_ema: TokenClassifierOutput = self.model_ema_1(batch[3], batch[1], False)
            outputs2: TokenClassifierOutput = self.model_2(batch[3], batch[1])
            outputs2_ema: TokenClassifierOutput = self.model_ema_2(batch[2], batch[0], False)
            # exponential moving average
            self.__update_ema_variables(self.model_1, self.model_ema_1, self.alpha)
            self.__update_ema_variables(self.model_2, self.model_ema_2, self.alpha)
            # hard label cross entropy loss
            loss_ce = outputs1.loss + outputs2.loss
            ins = torch.cat([outputs1.logits, outputs2.logits])
            outs = torch.cat([outputs2_ema.logits, outputs1_ema.logits])
            # soft label loss
            loss_ce_soft = self.ce_soft_loss(ins, outs)
            # total loss
            loss = loss_ce + self.soft_loss_weight * loss_ce_soft
            logger.debug("loss_ce: %f, loss_ce_soft: %f, loss: %f", loss_ce.item(),
                         loss_ce_soft.item(), loss.item())
        else:
            outputs1: TokenClassifierOutput = self.model_1(batch[0])
            # outputs2: TokenClassifierOutput = self.model_2(batch[0])
            # ins = (outputs1.logits + outputs2.logits) / 2
            ins = outputs2.logits
        return TokenClassifierOutput(logits=ins, loss=loss)

    @torch.no_grad()
    def __update_ema_variables(self, model, ema_model, alpha):
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


# utils
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
