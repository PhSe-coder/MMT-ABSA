import logging
import math
import os
import shutil
import sys
from time import localtime, strftime
from typing import List
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch
import random
import numpy
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, HfArgumentParser, set_seed, get_cosine_schedule_with_warmup
from transformers.modeling_outputs import TokenClassifierOutput
from utils.tag_utils import bio_tags_to_spans

from args import ModelArguments
from constants import SUPPORTED_MODELS, TAGS
from dataset import MMTDataset, MyDataset
from eval import absa_evaluate, evaluate
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
        args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    return args


def init(args: ModelArguments):
    if args.local_rank != 0:
        logging.disable(logging.CRITICAL)
    device_count = torch.cuda.device_count()
    if device_count >= 1:
        args.batch_size = int(args.batch_size / device_count)
    # assert args.batch_size % 2 == 0
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    initializers = {
        'kaiming_normal_': torch.nn.init.kaiming_normal_,
        'kaiming_uniform_': torch.nn.init.kaiming_uniform_,
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
        'sgd': torch.optim.SGD
    }
    if args.optimizer == 'adamW':
        from functools import partial
        args.optimizer = partial(optimizers[args.optimizer], amsgrad=True)
    else:
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
        args = self.args
        model_name = self.args.model_name
        num_labels = len(TAGS)
        assert model_name in SUPPORTED_MODELS, f'Model {model_name} is not supported'
        if model_name == 'bert':
            model = BertForTokenClassification.from_pretrained(args.pretrained_model,
                                                               num_labels=num_labels)
        elif model_name == 'mmt':
            model_1 = BertForTokenClassification.from_pretrained(args.pretrained_model,
                                                                 num_labels=num_labels)
            model_1_ema = BertForTokenClassification.from_pretrained(args.pretrained_model,
                                                                     num_labels=num_labels)
            model_2 = BertForTokenClassification.from_pretrained(args.pretrained_model,
                                                                 num_labels=num_labels)
            model_2_ema = BertForTokenClassification.from_pretrained(args.pretrained_model,
                                                                     num_labels=num_labels)
            model_1.load_state_dict(torch.load(self.args.init_1), strict=False)
            model_1_ema.load_state_dict(torch.load(self.args.init_1), strict=False)
            model_2.load_state_dict(torch.load(self.args.init_2), strict=False)
            model_2_ema.load_state_dict(torch.load(self.args.init_2), strict=False)
            for param in model_1_ema.parameters():
                param.detach_()
            for param in model_2_ema.parameters():
                param.detach_()
            model = MMTModel(model_1, model_1_ema, model_2, model_2_ema)
        device_count = torch.cuda.device_count()
        if device_count >= 1:
            if device_count > 1:
                torch.cuda.set_device(self.args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            model = model.cuda()
        self.model = DistributedDataParallel(model, find_unused_parameters=True)

    def dataset_init(self):
        args = self.args
        if args.do_train:
            if args.model_name == 'bert':
                # self.train_set = BaseDataset(args.train_file, self.tokenizer, True)
                self.train_set0 = MyDataset(args.train_file[0], self.tokenizer)
                self.train_set1 = MyDataset(args.train_file[1], self.tokenizer)
            elif args.model_name == 'mmt':
                self.train_set = MMTDataset(args.train_file, self.tokenizer, False)

        if args.do_eval:
            self.validation_set = MyDataset(self.args.validation_file, self.tokenizer)
        if args.do_predict:
            self.test_set = MyDataset(self.args.test_file, self.tokenizer)

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
              optimizers: List[Optimizer],
              scheduler,
              train_data_loader0: DataLoader,
              train_data_loader1: DataLoader,
              val_data_loader: DataLoader = None):
        from itertools import cycle
        global_step = 0
        n_total, loss_total = 0, 0
        max_val_f1, max_val_epoch = 0, 0
        gold_Y, pred_Y = [], []
        self.model.train()
        for epoch in range(int(self.args.num_train_epochs)):
            logger.info('>' * 100)
            logger.info('> epoch: {}'.format(epoch))
            for i, batch in tqdm(enumerate(zip(train_data_loader0, cycle(train_data_loader1))),
                                 f'epoch: {epoch}', len(train_data_loader0)):
                train_data_loader0.sampler.set_epoch(epoch)
                [optimizer.zero_grad() for optimizer in optimizers]
                batch_src = {k: v.to(self.args.local_rank) for k, v in batch[0].items()}
                batch_tgt = {k: v.to(self.args.local_rank) for k, v in batch[1].items()}
                targets0 = batch_src.get("gold_labels")
                targets1 = batch_tgt.pop("gold_labels")
                # targets = torch.cat([targets0, targets1])
                targets = targets1
                outputs: TokenClassifierOutput = self.model(batch_tgt, batch_src)
                loss = outputs.loss
                # logits = torch.cat([outputs0.logits, outputs1.logits])
                logits = outputs.logits
                assert loss is not None
                loss.backward()
                [optimizer.step() for optimizer in optimizers]
                scheduler.step()
                if targets is not None:  # consider unsupervised learning
                    pred_list, gold_list = self.id2label(
                        logits.argmax(dim=-1).tolist(), targets.tolist())
                    pred_Y.extend(pred_list)
                    gold_Y.extend(gold_list)
                l = len(logits)
                n_total += l
                loss_total += loss.item() * l
                global_step += 1
                if global_step % self.args.logging_steps == 0 or i == len(train_data_loader0) - 1:
                    if targets is not None:
                        p, r, f1 = absa_evaluate(pred_Y, gold_Y)
                    else:
                        p, r, f1 = 0, 0, 0
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
                prediction, gold = self.evaluate(val_data_loader)
                val_pre, val_rec, val_f1 = absa_evaluate(prediction, gold)
                logger.info(
                    f'> val_pre: {val_pre:.4f}, val_rec: {val_rec:.4f}, val_f1: {val_f1:.4f}')
                assert val_f1 > 0
                if val_f1 > max_val_f1:
                    max_val_f1 = val_f1
                    max_val_epoch = epoch
                    os.makedirs('state_dict', exist_ok=True)
                    path = 'state_dict/{}_{}_{}_val_pre_{}_val_rec_{}_val_f1_{}.pt'.format(
                        self.args.model_name, self.args.train_file[0].split("/")[-1].split(".")[0],
                        self.args.validation_file.split("/")[-1].split(".")[0], round(val_pre, 4),
                        round(val_rec, 4), round(val_f1, 4))
                    if self.args.local_rank == 0:
                        torch.save(self.model.state_dict(), path)
                    logger.info('>> saved: {}'.format(path))

                if epoch - max_val_epoch >= self.args.patience:
                    logger.info(f'>> early stopping at epoch: {epoch}')
                    break
            else:
                if epoch == int(self.args.num_train_epochs) - 1:
                    os.makedirs('state_dict', exist_ok=True)
                    if self.args.test_file is None:
                        path = 'state_dict/{}_{}_{}.pt'.format(
                            self.args.model_name,
                            self.args.train_file[0].split("/")[-1].split(".")[0],
                            strftime("%y%m%d_%H%M%S", localtime()))
                    else:
                        path = 'state_dict/{}_{}_{}_{}.pt'.format(
                            self.args.model_name,
                            self.args.train_file[0].split("/")[-1].split(".")[0],
                            self.args.test_file.split("/")[-1].split(".")[0],
                            strftime("%y%m%d_%H%M%S", localtime()))
                    if self.args.local_rank == 0:
                        torch.save(self.model.state_dict(), path)
        return path

    def evaluate(self, data_loader: DataLoader, to_file=False):
        self.model.eval()
        text, gold_Y, pred_Y = [], [], []
        for _, batch in enumerate(data_loader):
            batch = {k: v.to(self.args.local_rank) for k, v in batch.items()}
            targets = batch.pop("gold_labels")
            with torch.no_grad():
                outputs: TokenClassifierOutput = self.model(batch)
                logits = outputs.logits

            pred_list, gold_list = self.id2label(logits.detach().argmax(dim=-1).tolist(),
                                                 targets.tolist())
            pred_Y.extend(pred_list)
            gold_Y.extend(gold_list)
            if to_file:
                text.extend(
                    self.tokenizer.batch_decode(batch.get("input_ids"), skip_special_tokens=True))
        self.model.train()
        if to_file:
            with open(os.path.join(self.args.output_dir, "predict.txt"), "w") as f:
                for i in range(len(gold_Y)):
                    f.write(f"{text[i]}***{' '.join(pred_Y[i])}***{' '.join(gold_Y[i])}\n")
        return pred_Y, gold_Y

    def span2label(self, predict: List[List[int]], gold: List[List[int]]):
        gold_Y: List[List[tuple]] = []
        pred_Y: List[List[tuple]] = []
        for _pred, _gold in zip(predict, gold):
            assert len(_pred) == len(_gold)
            gold_Y.append(bio_tags_to_spans([TAGS[gold] if gold != -1 else 'O' for gold in _gold]))
            pred_Y.append(
                bio_tags_to_spans(
                    [TAGS[_pred[i]] if _gold[i] != -1 else 'O' for i in range(len(_pred))]))
        return pred_Y, gold_Y

    # def id2label(self, predict: List[List[int]], gold: List[List[int]]):
    #     gold_Y: List[List[str]] = []
    #     pred_Y: List[List[str]] = []
    #     for _pred, _gold in zip(predict, gold):
    #         assert len(_gold) == len(_pred)
    #         gold_list = [TAGS[_gold[i]] for i in range(len(_gold)) if _gold[i] != -1]
    #         pred_list = [TAGS[_pred[i]] for i in range(len(_gold)) if _gold[i] != -1]
    #         gold_Y.append(gold_list)
    #         pred_Y.append(pred_list)
    #     return pred_Y, gold_Y
    def id2label(self, predict: List[int], gold: List[List[int]]):
        gold_Y: List[List[str]] = []
        pred_Y: List[List[str]] = []
        for _gold in gold:
            gold_list = [TAGS[_gold[i]] for i in range(len(_gold)) if _gold[i] != -1]
            gold_Y.append(gold_list)
        idx = 0
        for item in gold_Y:
            pred_Y.append([TAGS[pred] for pred in predict[idx:idx + len(item)]])
            idx += len(item)
        return pred_Y, gold_Y

    def reset_params(self):
        for child in self.model.module.children():
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
            if best_model_path is not None:
                self.model.load_state_dict(torch.load(best_model_path))
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                numpy.random.seed(worker_seed)
                random.seed(worker_seed)
            datasampler0 = DistributedSampler(self.train_set0,
                                              num_replicas=dist.get_world_size(),
                                              rank=self.args.local_rank)
            datasampler1 = DistributedSampler(self.train_set1,
                                              num_replicas=dist.get_world_size(),
                                              rank=self.args.local_rank)
            train_data_loader0 = DataLoader(
                dataset=self.train_set0,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=self.train_set0.collate_fn if callable(
                    getattr(self.train_set0, "collate_fn", None)) else None,
                drop_last=True,
                num_workers=self.args.num_workers,
                sampler=datasampler0,
                worker_init_fn=seed_worker)
            train_data_loader1 = DataLoader(
                dataset=self.train_set1,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=self.train_set1.collate_fn if callable(
                    getattr(self.train_set1, "collate_fn", None)) else None,
                drop_last=True,
                num_workers=self.args.num_workers,
                sampler=datasampler1,
                worker_init_fn=seed_worker)
            param_optimizer = [(k, v) for k, v in self.model.named_parameters()
                               if v.requires_grad == True and 'pooler' not in k]
            pretrained_param_optimizer = [n for n in param_optimizer
                                          if 'bert' in n[0]]  #  and 'pos_embeddings' not in n[0]
            custom_param_optimizer = [
                n for n in param_optimizer
                if 'bert' not in n[0]  # and 'domain_model' not in n[0]  or 'pos_embeddings' in n[0]
            ]
            # domain_model_param_optimizer = [n for n in param_optimizer if 'domain_model' in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [{
                'params':
                [p for n, p in pretrained_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay':
                self.args.l2reg
            }, {
                'params':
                [p for n, p in pretrained_param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay':
                0.0
            }]
            total_steps = self.args.num_train_epochs * len(train_data_loader0)
            optimizers = [
                BertAdam(optimizer_grouped_parameters,
                         lr=self.args.bert_lr,
                         warmup=self.args.warmup,
                         t_total=total_steps)
            ]
            optimizers.append(
                self.args.optimizer([{
                    "params":
                    [p for n, p in custom_param_optimizer if not any(nd in n for nd in no_decay)],
                    "lr":
                    self.args.lr,
                    "alpha":
                    0.99,
                    "eps":
                    1e-12,
                    'weight_decay':
                    1e-2
                }, {
                    'params':
                    [p for n, p in custom_param_optimizer if any(nd in n for nd in no_decay)],
                    "lr":
                    self.args.lr,
                    "alpha":
                    0.99,
                    "eps":
                    1e-12,
                    'weight_decay':
                    0.0
                }]))
            scheduler = get_cosine_schedule_with_warmup(optimizers[-1],
                                                        0.1 * total_steps,
                                                        num_training_steps=total_steps)
            if self.args.model_name == 'bert':
                self.reset_params()
            val_data_loader = None
            if self.args.do_eval:
                val_data_loader = DataLoader(
                    dataset=self.validation_set,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    collate_fn=self.validation_set.collate_fn if callable(
                        getattr(self.validation_set, "collate_fn", None)) else None)
            best_model_path = self.train(optimizers, scheduler, train_data_loader0,
                                         train_data_loader1, val_data_loader)
        logger.info(f">> best model path: {best_model_path}")
        if self.args.do_predict:
            test_data_loader = DataLoader(dataset=self.test_set,
                                          batch_size=self.args.batch_size,
                                          shuffle=False,
                                          collate_fn=self.test_set.collate_fn if callable(
                                              getattr(self.test_set, "collate_fn", None)) else None)
            logger.info(f">> load best model: {best_model_path.split('/')[-1]}")
            if not self.args.do_train:
                self.model.load_state_dict(torch.load(best_model_path))
            prediction, gold = self.evaluate(test_data_loader, True)
            items = (absa_evaluate, "absa_prediction.txt"), (evaluate, "ae_prediction.txt")
            for func, filename in (items):
                test_pre, test_rec, test_f1 = func(prediction, gold)
                content = f'test_pre: {test_pre:.4f}, test_rec: {test_rec:.4f}, test_f1: {test_f1:.4f}'
                logger.info(f'>> {content}')
                with open(os.path.join(self.args.output_dir, filename), "w") as f:
                    f.write(content)

        dest_path = os.path.join(self.args.output_dir, best_model_path.split("/")[-1])
        if self.args.local_rank == 0:
            shutil.move(best_model_path, dest_path)
        return dest_path


if __name__ == '__main__':
    args = parse_args()[0]
    init(args)
    c = Constructor(args)
    c.run()