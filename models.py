from typing import List
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as f
from transformers import BertModel, BertConfig
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.modeling_outputs import TokenClassifierOutput
import pytorch_lightning as pl
from mi_estimators import InfoNCE
from constants import TAGS
from torch.optim import AdamW, SGD
from eval import absa_evaluate, evaluate
from model import BertForTokenClassification, MIBert
from model import SoftEntropy
from optimization import BertAdam


class MMTModel(pl.LightningModule):

    def __init__(self,
                 model_1: MIBert,
                 model_1_ema: MIBert,
                 model_2: MIBert,
                 model_2_ema: MIBert,
                 theta: float = 0.999,
                 soft_loss_weight: float = 0.01,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model_1', 'model_1_ema', 'model_2', 'model_2_ema'])
        self.automatic_optimization = False
        self.model_1 = model_1
        self.model_ema_1 = model_1_ema
        self.model_2 = model_2
        self.model_ema_2 = model_2_ema
        self.theta = theta
        self.soft_loss_weight = soft_loss_weight
        self.ce_soft_loss = SoftEntropy()
        self.lr = kwargs.get('lr')
        self.tokenizer = kwargs.get('tokenizer')
        hidden_size = model_1.classifier.in_features
        out_size = model_1.classifier.out_features
        self.mi = InfoNCE(hidden_size, out_size)
        self.softmax = nn.Softmax(-1)

    def forward(self, batch):
        loss = None
        if self.training:
            outputs1: TokenClassifierOutput = self.model_1(batch[1], batch[0])
            outputs1_ema: TokenClassifierOutput = self.model_ema_1(batch[3], batch[2], False)
            outputs2: TokenClassifierOutput = self.model_2(batch[3], batch[2])
            outputs2_ema: TokenClassifierOutput = self.model_ema_2(batch[1], batch[0], False)
            # exponential moving average
            self.__update_ema_variables(self.model_1, self.model_ema_1, self.theta)
            self.__update_ema_variables(self.model_2, self.model_ema_2, self.theta)
            loss = outputs1.loss + outputs2.loss
            # soft label loss
            ins = torch.cat([outputs1.logits, outputs2.logits])
            outs = torch.cat([outputs2_ema.logits, outputs1_ema.logits])
            loss_ce_soft = self.ce_soft_loss(ins, outs)
            # total loss
            loss += self.soft_loss_weight * loss_ce_soft
        else:
            outputs1: TokenClassifierOutput = self.model_1(batch[0])
            outputs2: TokenClassifierOutput = self.model_2(batch[0])
            ins = (outputs1.logits + outputs2.logits) / 2
            # ins = outputs1.logits
        return TokenClassifierOutput(logits=ins, loss=loss)

    @torch.no_grad()
    def __update_ema_variables(self, model, ema_model, alpha):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def configure_optimizers(self):
        params = [(k, v) for k, v in self.named_parameters()
                  if v.requires_grad == True and 'pooler' not in k]
        pretrained_param_optimizer = [n for n in params if 'bert' in n[0]]
        custom_param_optimizer = [n for n in params if 'bert' not in n[0] and 'mi_loss' not in n[0]]
        mi_loss_optimizer = [n for n in params if 'mi_loss' in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        pretrained_params = [{
            'params':
            [p for n, p in pretrained_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
            1e-2
        }, {
            'params': [p for n, p in pretrained_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
        # bert_opt = AdamW(pretrained_params, self.lr, amsgrad=True)
        bert_opt = BertAdam(pretrained_params, self.lr)
        params = [{
            'params': [p for _, p in custom_param_optimizer],
            'lr': self.lr
        }, {
            'params': [p for _, p in mi_loss_optimizer],
            'lr': 3e-5
        }]
        # custom_opt = AdamW(params, amsgrad=True, weight_decay=0.1)
        custom_opt = SGD(params, momentum=0.9, weight_decay=0.01)
        return [bert_opt,
                custom_opt]

    def training_step(self, train_batch, batch_idx):
        opts = self.optimizers()
        outputs = self.forward(train_batch)
        loss = outputs.loss
        self.manual_backward(loss)
        for opt in opts:
            opt.step()
            opt.zero_grad()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch[0].pop("gold_labels")
        outputs: TokenClassifierOutput = self.forward(batch)
        logits = outputs.logits
        pred_list, gold_list = id2label(logits.detach().argmax(dim=-1).tolist(), targets.tolist())
        return pred_list, gold_list

    def validation_epoch_end(self, outputs):
        gold_Y, pred_Y = [], []
        for pred_list, gold_list in outputs:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
        val_pre, val_rec, val_f1 = absa_evaluate(pred_Y, gold_Y)
        self.log_dict({"val_pre": val_pre, "val_rec": val_rec, "val_f1": val_f1})

    def test_step(self, batch, batch_idx):
        targets = batch[0].pop("gold_labels")
        outputs: TokenClassifierOutput = self.forward(batch)
        logits = outputs.logits
        pred_list, gold_list = id2label(logits.detach().argmax(dim=-1).tolist(), targets.tolist())
        sentence = self.tokenizer.batch_decode(batch[0].get("input_ids"), skip_special_tokens=True)
        return pred_list, gold_list, sentence

    def test_epoch_end(self, outputs) -> None:
        gold_Y, pred_Y, text = [], [], []
        for pred_list, gold_list, sentence in outputs:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
            text.extend(sentence)
        test_pre, test_rec, test_f1 = absa_evaluate(pred_Y, gold_Y)
        self.log_dict({
            "absa_test_pre": round(test_pre, 4),
            "absa_test_rec": round(test_rec, 4),
            "absa_test_f1": round(test_f1, 4)
        })
        test_pre, test_rec, test_f1 = evaluate(pred_Y, gold_Y)
        self.log_dict({
            "ae_test_pre": round(test_pre, 4),
            "ae_test_rec": round(test_rec, 4),
            "ae_test_f1": round(test_f1, 4)
        })
        #     with open(os.path.join(self.args.output_dir, filename), "w") as f:
        #         f.write(content)
        # with open(os.path.join(self.args.output_dir, "predict.txt"), "w") as f:
        #     for i in range(len(gold_Y)):
        #         f.write(f"{text[i]}***{' '.join(pred_Y[i])}***{' '.join(gold_Y[i])}\n")

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("MMTModel")
        group.add_argument("--alpha",
                           type=float,
                           help='the weight parameter of the Mutual Information loss')
        group.add_argument("--theta", type=float, default=0.999, help='the weight of the ema')
        group.add_argument("--soft_loss_weight", type=float, default=0.01)
        # group.add_argument("--init_1",
        #                    type=str,
        #                    help="pretrained model checkpoint for initializing the mmt model 1")
        # group.add_argument("--init_2",
        #                    type=str,
        #                    help="pretrained model checkpoint for initializing the mmt model 2")
        return parent_parser


class MIBertClassifier(pl.LightningModule):

    def __init__(self, alpha: float, tau: float, pretrained_model: str, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model = MIBert.from_pretrained(pretrained_model, alpha, tau, num_labels=len(TAGS))
        self.lr = kwargs.get('lr')
        self.tokenizer = kwargs.get('tokenizer')

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, batch):
        return self.model(batch[1], batch[0])

    def configure_optimizers(self):
        params = [(k, v) for k, v in self.named_parameters()
                  if v.requires_grad == True and 'pooler' not in k]
        pretrained_param_optimizer = [n for n in params if 'bert' in n[0]]
        custom_param_optimizer = [n for n in params if 'bert' not in n[0] and 'mi_loss' not in n[0]]
        mi_loss_optimizer = [n for n in params if 'mi_loss' in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        pretrained_params = [{
            'params':
            [p for n, p in pretrained_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
            1e-2
        }, {
            'params': [p for n, p in pretrained_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
        # bert_opt = AdamW(pretrained_params, self.lr, amsgrad=True)
        bert_opt = BertAdam(pretrained_params, self.lr)
        params = [{
            'params': [p for n, p in custom_param_optimizer],
            'lr': self.lr
        }, {
            'params': [p for n, p in mi_loss_optimizer],
            'lr': 1e-4
        }]
        custom_opt = AdamW(params, amsgrad=True, weight_decay=0.1)
        return [bert_opt, custom_opt]

    def training_step(self, train_batch, batch_idx):
        opts = self.optimizers()
        for opt in opts:
            opt.zero_grad()
        outputs = self.forward(train_batch)
        loss = outputs.loss
        self.manual_backward(loss)
        for opt in opts:
            opt.step()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch[0].pop("gold_labels")
        outputs: TokenClassifierOutput = self.model(batch[0])
        logits = outputs.logits
        pred_list, gold_list = id2label(logits.detach().argmax(dim=-1).tolist(), targets.tolist())
        return pred_list, gold_list

    def validation_epoch_end(self, outputs):
        gold_Y, pred_Y = [], []
        for pred_list, gold_list in outputs:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
        val_pre, val_rec, val_f1 = absa_evaluate(pred_Y, gold_Y)
        self.log_dict({"val_pre": val_pre, "val_rec": val_rec, "val_f1": val_f1})

    def test_step(self, batch, batch_idx):
        targets = batch[0].pop("gold_labels")
        outputs: TokenClassifierOutput = self.model(batch[0])
        logits = outputs.logits
        pred_list, gold_list = id2label(logits.detach().argmax(dim=-1).tolist(), targets.tolist())
        sentence = self.tokenizer.batch_decode(batch[0].get("input_ids"), skip_special_tokens=True)
        return pred_list, gold_list, sentence

    def test_epoch_end(self, outputs) -> None:
        gold_Y, pred_Y, text = [], [], []
        for pred_list, gold_list, sentence in outputs:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
            text.extend(sentence)
        test_pre, test_rec, test_f1 = absa_evaluate(pred_Y, gold_Y)
        self.log_dict({
            "absa_test_pre": round(test_pre, 4),
            "absa_test_rec": round(test_rec, 4),
            "absa_test_f1": round(test_f1, 4)
        })
        test_pre, test_rec, test_f1 = evaluate(pred_Y, gold_Y)
        self.log_dict({
            "ae_test_pre": round(test_pre, 4),
            "ae_test_rec": round(test_rec, 4),
            "ae_test_f1": round(test_f1, 4)
        })

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MIBert")
        parser.add_argument("--alpha",
                            type=float,
                            help='the weight parameter of the Mutual Information loss')
        return parser


class BertClassifier(pl.LightningModule):

    def __init__(self, pretrained_model: str, num_labels: int, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model = BertForTokenClassification.from_pretrained(pretrained_model,
                                                                num_labels=num_labels)
        self.lr = kwargs.get('lr')
        self.tokenizer = kwargs.get('tokenizer')

    # Using custom or multiple metrics (default_hp_metric=False)
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        pretrained_param_optimizer = [(k, v) for k, v in self.named_parameters()
                                      if 'pooler' not in k]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params = [{
            'params':
            [p for n, p in pretrained_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
            1e-2
        }, {
            'params': [p for n, p in pretrained_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
        return BertAdam(params, self.lr, 0.1, self.trainer.estimated_stepping_batches)

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        outputs = self.forward(**train_batch[0])
        loss = outputs.loss
        self.manual_backward(loss)
        opt.step()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch[0].pop("gold_labels")
        outputs: TokenClassifierOutput = self.forward(**batch[0])
        logits = outputs.logits
        pred_list, gold_list = id2label(logits.argmax(dim=-1).tolist(), targets.tolist())
        return pred_list, gold_list

    def validation_epoch_end(self, outputs):
        gold_Y, pred_Y = [], []
        for pred_list, gold_list in outputs:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
        val_pre, val_rec, val_f1 = absa_evaluate(pred_Y, gold_Y)
        self.log_dict({"val_pre": val_pre, "val_rec": val_rec, "val_f1": val_f1})

    def test_step(self, batch, batch_idx):
        targets = batch[0].pop("gold_labels")
        outputs: TokenClassifierOutput = self.forward(**batch[0])
        logits = outputs.logits
        pred_list, gold_list = id2label(logits.detach().argmax(dim=-1).tolist(), targets.tolist())
        sentence = self.tokenizer.batch_decode(batch[0].get("input_ids"), skip_special_tokens=True)
        return pred_list, gold_list, sentence

    def test_epoch_end(self, outputs) -> None:
        gold_Y, pred_Y, text = [], [], []
        for pred_list, gold_list, sentence in outputs:
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
            text.extend(sentence)
        test_pre, test_rec, test_f1 = absa_evaluate(pred_Y, gold_Y)
        self.log_dict({
            "absa_test_pre": round(test_pre, 4),
            "absa_test_rec": round(test_rec, 4),
            "absa_test_f1": round(test_f1, 4)
        })
        test_pre, test_rec, test_f1 = evaluate(pred_Y, gold_Y)
        self.log_dict({
            "ae_test_pre": round(test_pre, 4),
            "ae_test_rec": round(test_rec, 4),
            "ae_test_f1": round(test_f1, 4)
        })

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BertClassifier")
        return parser


def id2label(predict: List[int], gold: List[List[int]]):
    gold_Y: List[List[str]] = []
    pred_Y: List[List[str]] = []
    for _gold in gold:
        gold_list = [TAGS[_gold[i]] for i in range(len(_gold)) if _gold[i] != -1]
        gold_Y.append(gold_list)
    idx = 0
    for item in gold_Y:
        pred_Y.append([TAGS[pred] for pred in predict[idx:idx + len(item)]])
        idx += len(item)
    assert idx == len(predict)
    return pred_Y, gold_Y
