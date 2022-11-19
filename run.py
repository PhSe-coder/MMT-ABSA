import logging
import math
import os
import shutil
import sys
from time import localtime, strftime
from typing import List, Union

import torch
from pytorch_transformers import BertModel, BertConfig
from torch.optim.optimizer import Optimizer
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, HfArgumentParser, set_seed
from transformers.modeling_outputs import TokenClassifierOutput

from args import ModelArguments
from constants import SUPPORTED_MODELS, TAGS
from dataset import BaseDataset, MMTDataset
from eval import absa_evaluate
from model import *
from optimization import BertAdam

logger = logging.getLogger()
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)


def parse_args():
    parser = HfArgumentParser(ModelArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()

    return args


def init(args: ModelArguments):
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        "adamW": torch.optim.AdamW,
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
        "bertAdam": BertAdam
    }
    args.optimizer = optimizers[args.optimizer]
    args.initializer = initializers[args.initializer]
    os.makedirs("logs", exist_ok=True)
    log_file = './logs/{}-{}.log'.format(args.model_name, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))


class Constructor:

    def __init__(self, args: ModelArguments):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model,
                                                       model_max_length=args.max_seq_length)
        self.dataset_init()
        self.model_init()
        self.running_init()
        self.print_args()

    def model_init(self):
        model_name = self.args.model_name
        num_labels = len(TAGS)
        device = self.args.device
        assert model_name in SUPPORTED_MODELS, f'Model {model_name} is not supported'
        if model_name == 'bert':
            model = BertForTokenClassification.from_pretrained(args.pretrained_model,
                                                               num_labels=num_labels)
        elif model_name == 'mmt':
            model_1 = BertForTokenClassification.from_pretrained(args.pretrained_model,
                                                                 num_labels=num_labels)
            model_ema_1 = BertForTokenClassification.from_pretrained(args.pretrained_model,
                                                                     num_labels=num_labels)
            model_2 = BertForTokenClassification.from_pretrained(args.pretrained_model,
                                                                 num_labels=num_labels)
            model_ema_2 = BertForTokenClassification.from_pretrained(args.pretrained_model,
                                                                     num_labels=num_labels)
            config = BertConfig.from_pretrained(args.pretrained_model)
            for param in model_ema_1.parameters():
                param.detach_()
            for param in model_ema_2.parameters():
                param.detach_()
            domain_model = DomainModel(config.hidden_size)
            model = MMTModel(model_1, model_ema_1, model_2, model_ema_2, domain_model)
        self.model: Union[BertForTokenClassification, MMTModel] = model
        self.model.to(device)

    def dataset_init(self):
        args = self.args
        dataset = MMTDataset if args.model_name == 'mmt' else BaseDataset
        if args.do_train:
            datasets = []
            datasets.append(dataset(args.train_files[0], self.tokenizer, args.device))
            if args.model_name == 'mmt':
                datasets.append(dataset(args.train_files[1], self.tokenizer, args.device, False))
            self.train_set = ConcatDataset(datasets)
        if args.do_eval:
            self.validation_set = dataset(self.args.validation_file, self.tokenizer, args.device,
                                          False)
        if args.do_predict:
            self.test_set = dataset(self.args.test_file, self.tokenizer, args.device, False)

    def running_init(self):
        os.makedirs(self.args.output_dir, exist_ok=True)

    def print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(
            n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.args):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))

    def train(self,
              optimizer: Optimizer,
              train_data_loader: DataLoader,
              val_data_loader: DataLoader = None):
        global_step = 0
        n_total, loss_total = 0, 0
        max_val_f1, max_val_epoch = 0, 0
        gold_Y, pred_Y = [], []
        self.model.train()
        for epoch in range(int(self.args.num_train_epochs)):
            logger.info('>' * 100)
            logger.info('> epoch: {}'.format(epoch))
            for i, batch in tqdm(enumerate(train_data_loader), f'epoch: {epoch}',
                                 len(train_data_loader)):
                optimizer.zero_grad()
                targets = batch.get("gold_labels")
                outputs: TokenClassifierOutput = self.model(**batch)
                logits, loss = outputs.logits, outputs.loss
                assert loss is not None
                loss.backward()
                optimizer.step()
                self.model.post_operation(global_step=global_step)
                pred_list, gold_list = self.id2label(
                    logits.argmax(dim=-1).tolist(), targets.tolist())
                pred_Y.extend(pred_list)
                gold_Y.extend(gold_list)
                l = len(logits)
                n_total += l
                loss_total += loss.item() * l
                global_step += 1
                if global_step % self.args.logging_steps == 0 or i == len(train_data_loader) - 1:
                    p, r, f1 = absa_evaluate(pred_Y, gold_Y)
                    inf = {
                        "steps": global_step,
                        "loss": loss_total / n_total,
                        "precision": f"{p:.4f}",
                        "recall": f"{r:.4f}",
                        "micro_f1": f"{f1:.4f}"
                    }
                    logger.removeHandler(stdout_handler)
                    logger.info(", ".join([f"{k}: {v}" for k, v in inf.items()]))
                    logger.addHandler(stdout_handler)
                    gold_Y.clear()
                    pred_Y.clear()
            if self.args.do_eval:
                val_pre, val_rec, val_f1 = self.evaluate(val_data_loader)
                logger.info(
                    f'> val_pre: {val_pre:.4f}, val_rec: {val_rec:.4f}, val_f1: {val_f1:.4f}')
                assert val_f1 > 0
                if val_f1 > max_val_f1:
                    max_val_f1 = val_f1
                    max_val_epoch = epoch
                    os.makedirs('state_dict', exist_ok=True)
                    path = 'state_dict/{}_{}_{}_val_pre_{}_val_rec_{}_val_f1_{}.pt'.format(
                        self.args.model_name,
                        self.args.train_files.split("/")[-1].split(".")[0],
                        self.args.validation_file.split("/")[-1].split(".")[0], round(val_pre, 4),
                        round(val_rec, 4), round(val_f1, 4))
                    torch.save(self.model.state_dict(), path)
                    logger.info('>> saved: {}'.format(path))

                if epoch - max_val_epoch >= self.args.patience:
                    logger.info(f'>> early stopping at epoch: {epoch}')
                    break
            else:
                path = 'state_dict/{}_{}_{}.pt'.format(
                    self.args.model_name,
                    self.args.train_files.split("/")[-1].split(".")[0],
                    self.args.test_file.split("/")[-1].split(".")[0])
                torch.save(self.model.state_dict(), path)
        return path

    def evaluate(self, data_loader: DataLoader, to_file=False):
        self.model.eval()
        text, gold_Y, pred_Y = [], [], []
        for _, batch in enumerate(data_loader):
            targets = batch.get("gold_labels")
            with torch.no_grad():
                outputs: TokenClassifierOutput = self.model(**batch)
                logits = outputs.logits
            pred_list, gold_list = self.id2label(logits.detach().argmax(dim=-1).tolist(),
                                                 targets.tolist())
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
            if to_file:
                text.extend(self.tokenizer.decode(batch.get("input_ids")))
        self.model.train()
        if to_file:
            with open(os.path.join(self.args.output_dir, "predict.txt"), "w") as f:
                for i in range(len(gold_Y)):
                    f.write(f"{text[i]}***{pred_Y[i]}***{gold_Y[i]}")
        return absa_evaluate(pred_Y, gold_Y)

    def id2label(self, predict: List[List[int]], gold: List[List[int]]):
        gold_Y: List[List[str]] = []
        pred_Y: List[List[str]] = []
        for _pred, _gold in zip(predict, gold):
            assert len(_gold) == len(_pred)
            gold_list = [TAGS[_gold[i]] for i in range(len(_gold)) if _gold[i] != -1]
            pred_list = [TAGS[_pred[i]] for i in range(len(_gold)) if _gold[i] != -1]
            gold_Y.append(gold_list)
            pred_Y.append(pred_list)
        return pred_Y, gold_Y

    def reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.args.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def run(self):
        best_model_path = self.args.best_model_path
        if self.args.do_train:
            train_data_loader = DataLoader(
                dataset=self.train_set,
                batch_size=self.args.batch_size,
                shuffle=True,
                collate_fn=self.train_set.datasets[0].collate_fn if hasattr(
                    self.train_set.datasets[0], "collate_fn") else None)
            param_optimizer = [(k, v) for k, v in self.model.named_parameters()
                               if v.requires_grad == True]
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [{
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay':
                self.args.l2reg
            }, {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay':
                0.0
            }]
            if self.args.optimizer == BertAdam:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=self.args.lr,
                                     warmup=self.args.warmup,
                                     t_total=self.args.num_train_epochs * len(train_data_loader))
            else:
                optimizer = self.args.optimizer(optimizer_grouped_parameters, lr=self.args.lr)
            # self.reset_params()
            val_data_loader = None
            if self.args.do_eval:
                val_data_loader = DataLoader(
                    dataset=self.validation_set,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    collate_fn=self.validation_set.datasets[0].collate_fn if hasattr(
                        self.train_set.datasets[0], "collate_fn") else None)
            best_model_path = self.train(optimizer, train_data_loader, val_data_loader)
        if self.args.do_predict:
            test_data_loader = DataLoader(
                dataset=self.test_set,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=self.test_set.datasets[0].collate_fn if hasattr(
                    self.train_set.datasets[0], "collate_fn") else None)
            logger.info(f">> load best model: {best_model_path.split('/')[-1]}")
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=torch.device(self.args.device)))
            # self.model = torch.load(best_model_path, map_location=torch.device(self.args.device))
            test_pre, test_rec, test_f1 = self.evaluate(test_data_loader)
            content = f'test_pre: {test_pre:.4f}, test_rec: {test_rec:.4f}, test_f1: {test_f1:.4f}'
            logger.info(f'>> {content}')
            with open(os.path.join(self.args.output_dir, "absa_prediction.txt"), "w") as f:
                f.write(content)
        shutil.move(best_model_path,
                    os.path.join(self.args.output_dir,
                                 best_model_path.split("/")[-1]))


if __name__ == '__main__':
    args = parse_args()[0]
    init(args)
    c = Constructor(args)
    c.run()