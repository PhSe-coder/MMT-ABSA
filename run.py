import math
import sys
from time import localtime, strftime
import time
import shutil
from typing import List
from eval import absa_evaluate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from optimization import BertAdam
import os
import logging
from tqdm import tqdm
from transformers import HfArgumentParser, BertTokenizer, set_seed
from pytorch_transformers import BertModel
from args import ModelArguments
from dataset import MyDataset
from model import BertForTokenClassification
from constants import TAGS

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

    dataset_file = {
        "train": args.train_file,
        "validation": args.validation_file,
        "test": args.test_file
    }
    args.dataset_file = dataset_file
    os.makedirs("logs", exist_ok=True)
    log_file = './logs/{}-{}.log'.format(args.model_name, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

class Constructor:
    def __init__(self, args: ModelArguments):
        self.args = args
        self.num_labels = len(TAGS)
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, 
            model_max_length=args.max_seq_length)
        files = args.dataset_file['train'], args.dataset_file['validation'], args.dataset_file['test']
        if args.do_train:
            self.train_set = MyDataset(files[0], tokenizer, args.device)
        if args.do_eval:
            self.validation_set = MyDataset(files[1], tokenizer, args.device)
        if args.do_predict:
            self.test_set = MyDataset(files[2], tokenizer, args.device)
        self.model = BertForTokenClassification.from_pretrained(args.pretrained_model, num_labels=self.num_labels)
        # self.model = Model(args.pretrained_model)
        self.model.to(args.device)
        self.print_args()

    def print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'
                        .format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.args):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))

    def train(self, criterion, optimizer: Optimizer, train_data_loader: DataLoader, 
            val_data_loader: DataLoader=None):
        global_step = 0
        n_total, loss_total = 0, 0
        max_val_f1, max_val_epoch = 0, 0
        gold_Y, pred_Y = [], []
        self.model.train()
        for epoch in range(int(self.args.num_train_epochs)):
            logger.info('>' * 100)
            logger.info('> epoch: {}'.format(epoch))
            for i, batch in tqdm(enumerate(train_data_loader), f'epoch: {epoch}', len(train_data_loader)):
                global_step += 1
                optimizer.zero_grad()
                targets = batch.pop("labels")
                outputs = self.model(**batch)
                outputs = outputs.logits
                loss = criterion(outputs.view(-1, self.num_labels), targets.view(-1))
                loss.backward()
                optimizer.step()
                pred_list, gold_list = self.id2label(outputs.argmax(dim=-1).tolist(), targets.tolist())
                pred_Y.extend(pred_list)
                gold_Y.extend(gold_list)
                l = len(outputs)
                n_total += l
                loss_total += loss.item() * l
                if global_step % self.args.logging_steps == 0 or i == len(train_data_loader) -1:
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
                    gold_Y, pred_Y = [], []
            if self.args.do_eval:
                val_pre, val_rec, val_f1 = self.evaluate(val_data_loader)
                logger.info(f'> val_pre: {val_pre:.4f}, val_rec: {val_rec:.4f}, val_f1: {val_f1:.4f}')

                if val_f1 > max_val_f1:
                    max_val_f1 = val_f1
                    max_val_epoch = epoch
                    os.makedirs('state_dict', exist_ok=True)
                    path = 'state_dict/{}_{}_{}_val_pre_{}_val_rec_{}_val_f1_{}.pt'.format(
                        self.args.model_name, self.args.train_file.split("/")[-1].split(".")[0], 
                        self.args.validation_file.split("/")[-1].split(".")[0], 
                        round(val_pre, 4), round(val_rec, 4), round(val_f1, 4))
                    torch.save(self.model.state_dict(), path)
                    logger.info('>> saved: {}'.format(path))
            
                if epoch - max_val_epoch >= self.args.patience:
                    logger.info(f'>> early stopping at epoch: {epoch}')
                    break
            else:
                path = 'state_dict/{}_{}_{}.pt'.format(
                        self.args.model_name, self.args.train_file.split("/")[-1].split(".")[0],
                        self.args.test_file.split("/")[-1].split(".")[0])
                torch.save(self.model.state_dict(), path)
        return path
    
    def evaluate(self, data_loader: DataLoader):
        self.model.eval()
        gold_Y, pred_Y = [], []
        for _, batch in enumerate(data_loader):
            targets = batch.pop("labels")
            with torch.no_grad():
                outputs = self.model(**batch)
                outputs = outputs.logits
            pred_list, gold_list = self.id2label(outputs.detach().argmax(dim=-1).tolist(), targets.tolist())
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
        self.model.train()
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
        os.makedirs(self.args.output_dir, exist_ok=True)
        best_model_path = self.args.best_model_path
        if self.args.do_train:
            train_data_loader = DataLoader(dataset=self.train_set, batch_size=self.args.batch_size, shuffle=True)
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad == True]
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
                    'weight_decay': self.args.l2reg
                },
                {
                    'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0
                }
            ]
            if self.args.optimizer == BertAdam:
                optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=self.args.lr,
                            warmup=self.args.warmup,
                            t_total=self.args.num_train_epochs*len(train_data_loader))
            else:
                optimizer = self.args.optimizer(optimizer_grouped_parameters, lr=self.args.lr)
            # self.reset_params()
            val_data_loader = None
            if self.args.do_eval:
                val_data_loader = DataLoader(dataset=self.validation_set, batch_size=self.args.batch_size, shuffle=False)
            best_model_path = self.train(criterion, optimizer, train_data_loader, val_data_loader)
        if self.args.do_predict:
            test_data_loader = DataLoader(dataset=self.test_set, batch_size=self.args.batch_size, shuffle=False)
            logger.info(f">> load best model: {best_model_path.split('/')[-1]}")
            self.model.load_state_dict(torch.load(best_model_path, map_location=torch.device(self.args.device)))
            # self.model = torch.load(best_model_path, map_location=torch.device(self.args.device))
            test_pre, test_rec, test_f1 = self.evaluate(test_data_loader)
            content = f'test_pre: {test_pre:.4f}, test_rec: {test_rec:.4f}, test_f1: {test_f1:.4f}'
            logger.info(f'>> {content}')
            with open(os.path.join(self.args.output_dir, "absa_prediction.txt"), "w") as f:
                f.write(content)
        shutil.move(best_model_path, os.path.join(self.args.output_dir, best_model_path.split("/")[-1]))

if __name__ == '__main__':
    args = parse_args()[0]
    init(args)
    c = Constructor(args)
    c.run()