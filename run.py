import math
import sys
from time import localtime, strftime
import torch
import os
import logging
import torch.nn as nn
from tqdm import tqdm
from transformers import HfArgumentParser, BertTokenizer, BertModel, set_seed, TrainingArguments
from torch.utils.data import DataLoader
from args import ModelArguments
from dataset import MyDataset
from model import Model
from constants import TAGS
from torch.optim.optimizer import Optimizer
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def parse_args():
    parser = HfArgumentParser(ModelArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()

    return args

def init(args):
    if args.seed is not None:
        set_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    args.optimizer = optimizers[args.optimizer]
    args.initializer = initializers[args.initializer]

    dataset_file = {
        "train": args.train_file,
        "validation": args.validation_file,
        "test": args.test_file
    }
    args.dataset_file = dataset_file

    log_file = './logs/{}-{}-{}-{}.log'.format(
        args.model_name, args.train_dataset, args.test_dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

class Constructor:
    def __init__(self, args):
        self.args = args
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, model_max_length=args.max_seq_length)
        self.train_set = MyDataset(args.dataset_file['train'], tokenizer)
        self.validation_set = MyDataset(args.dataset_file['validation'], tokenizer)
        self.test_set = MyDataset(args.dataset_file['test'], tokenizer)
        self.model = Model(args.pretrained_model)
        self.__print_args()

    def __print_args(self):
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

    def __train(self, criterion, optimizer: Optimizer, train_data_loader: DataLoader, 
            val_data_loader: DataLoader=None):
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        num_classes = len(TAGS)
        for epoch in range(self.args.num_epoch):
            logger.info('>' * 100)
            logger.info('> epoch: {}'.format(epoch))
            precision = MulticlassPrecision(num_classes, 'macro')
            recall = MulticlassRecall(num_classes, 'macro')
            micro_f1 = MulticlassPrecision(num_classes, 'macro')
            self.model.train()
            logger.removeHandler(logging.StreamHandler(sys.stdout))
            for _, batch in tqdm(enumerate(train_data_loader)):
                global_step += 1
                optimizer.zero_grad()
                outputs = self.model(batch)
                targets = batch['label']
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                precision.update(outputs, targets)
                recall.update(outputs, targets)
                micro_f1.update(outputs, targets)
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.args.logging_steps == 0:
                    train_loss = loss_total / n_total
                    logger.info(f'loss: {train_loss:.4f}, \
                                precision: {precision.compute().item():.4f}, \
                                recall: {recall.compute().item():.4f}, \
                                micro_f1: {micro_f1.compute().item():.4f}')
                    precision.reset()
                    recall.reset()
                    micro_f1.reset()
            val_pre, val_rec, val_f1 = self.__evaluate(val_data_loader)
            logger.addHandler(logging.StreamHandler(sys.stdout))
            logger.info(f'> val_pre: {val_pre:.4f}, val_rec: {val_rec:.4f}, val_f1: {val_f1:.4f}')

            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
                max_val_epoch = epoch
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_{2}_val_pre_{3}_val_rec_{4}_val_f1_{5}.pt'.format(
                    self.args.model_name, self.args.train_dataset, self.args.test_dataset, 
                    round(val_pre, 4), round(val_rec, 4), round(val_f1, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            
            if epoch - max_val_epoch >= self.args.patience:
                logger.info(f'>> early stopping at epoch: {epoch}')
                break
        return path
    
    def __evaluate(self, data_loader: DataLoader):
        self.model.eval()
        num_classes = len(TAGS)
        precision = MulticlassPrecision(num_classes, 'macro')
        recall = MulticlassRecall(num_classes, 'macro')
        micro_f1 = MulticlassPrecision(num_classes, 'macro')
        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                targets = batch['label']
                outputs = self.model(batch)
                precision.update(outputs, targets)
                recall.update(outputs, targets)
                micro_f1.update(outputs, targets)
        p , r, f1 = precision.compute().item(), recall.compute().item(), micro_f1.compute().item()
        precision.reset()
        recall.reset()
        micro_f1.reset()
        return p , r, f1
    
    def __reset_params(self):
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
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parametargsopters())
        optimizer = self.args.optimizer(_params, lr=self.args.lr, weight_decay=self.args.l2reg)
        train_data_loader = DataLoader(dataset=self.train_set, batch_size=self.args.batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=self.validation_set, batch_size=self.args.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.test_set, batch_size=self.args.batch_size, shuffle=False)
        self.__reset_params()
        best_model_path = self.__train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        test_pre, test_rec, test_f1 = self.__evaluate(test_data_loader)
        logger.info(f'>> test_pre: {test_pre:.4f}, test_rec: {test_rec:.4f}, test_f1: {test_f1:.4f}')
        
        

if __name__ == '__main__':
    args = parse_args()
    init(args)
    c = Constructor(args)
    c.run()