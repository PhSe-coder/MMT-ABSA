import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from constants import TAGS
from dataset import MyDataset
from model import MIBert
import pytorch_lightning as pl
from argparse import ArgumentParser
from models import MIBertClassifier, MMTModel, BertClassifier
from pytorch_lightning.callbacks import ModelCheckpoint

parser = ArgumentParser()
# add all the available trainer options to argparse
parser = pl.Trainer.add_argparse_args(parser)
# add PROGRAM level args
parser.add_argument("--model_name", type=str, default="bert", help="bert or mmt")
parser.add_argument("--pretrained_model", type=str, default='bert-base-uncased')
parser.add_argument("--tau", type=float, default=1.0, help='the temperature of the classifier')
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--train_file", nargs='+', help="The input training data files")
parser.add_argument("--validation_file", nargs='+', default=None, help="evaluation data file")
parser.add_argument("--test_file", nargs='+', default=None, help="predict data file ")
parser.add_argument("--output_dir", type=str)
parser.add_argument("--do_train", default=True, action='store_true')
parser.add_argument("--do_eval", default=True, action='store_true')
parser.add_argument("--do_predict", default=True, action='store_true')
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=0)


def dataloader_init(args):
    pretrained_model = args.pretrained_model
    batch_size = args.batch_size
    tokenizer = BertTokenizer.from_pretrained(pretrained_model, model_max_length=100)
    train_set = MyDataset(args.train_file, tokenizer)
    validation_set = MyDataset(args.validation_file, tokenizer)
    test_set = MyDataset(args.test_file, tokenizer)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(validation_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)
    return train_loader, val_loader, test_loader, tokenizer


def mmt_model_trainer(args):
    pm = args.pretrained_model
    alpha = args.alpha
    tau = args.tau
    num_labels = len(TAGS)
    train_loader, val_loader, test_loader, tokenizer = dataloader_init(args)
    # model
    model_1 = MIBert.from_pretrained(pm, alpha, tau, num_labels=num_labels)
    model_1_ema = MIBert.from_pretrained(pm, alpha, tau, num_labels=num_labels)
    model_2 = MIBert.from_pretrained(pm, alpha, tau, num_labels=num_labels)
    model_2_ema = MIBert.from_pretrained(pm, alpha, tau, num_labels=num_labels)
    for param in model_1_ema.parameters():
        param.detach_()
    for param in model_2_ema.parameters():
        param.detach_()
    dict_args = vars(args)
    dict_args['tokenizer'] = tokenizer
    dict_args['model_1'] = model_1
    dict_args['model_1_ema'] = model_1_ema
    dict_args['model_2'] = model_2
    dict_args['model_2_ema'] = model_2_ema
    model = MMTModel(**dict_args)

    # training
    checkpoint_callback = ModelCheckpoint(monitor="val_f1",
                                          mode='max',
                                          filename='mmt-absa-{epoch:02d}-{val_f1:.2f}')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path='best')


def mi_bert_trainer(args):
    train_loader, val_loader, test_loader, tokenizer = dataloader_init(args)
    # model
    dict_args = vars(args)
    dict_args['tokenizer'] = tokenizer
    model = MIBertClassifier(**dict_args)

    # training
    checkpoint_callback = ModelCheckpoint(monitor="val_f1",
                                          mode='max',
                                          filename='mi-absa-{epoch:02d}-{val_f1:.2f}')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path='best')


def bert_trainer(args):
    train_loader, val_loader, test_loader, tokenizer = dataloader_init(args)
    # model
    dict_args = vars(args)
    dict_args['num_labels'] = len(TAGS)
    dict_args['tokenizer'] = tokenizer
    model = BertClassifier(**dict_args)

    # training
    checkpoint_callback = ModelCheckpoint(monitor="val_f1",
                                          mode='max',
                                          filename='bert-absa-{epoch:02d}-{val_f1:.2f}',
                                          save_weights_only=True)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path='best')


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    temp_args, _ = parser.parse_known_args()
    model_name = temp_args.model_name
    trainers = {
        "mmt": mmt_model_trainer,
        "mi_bert": mi_bert_trainer,
        "bert": bert_trainer,
    }
    models = {"mmt": MMTModel, "mi_bert": MIBert, "bert": BertClassifier}
    # add model specific args
    parser = models[model_name].add_model_specific_args(parser)
    trainer = trainers[model_name]
    arguments = parser.parse_args()
    pl.seed_everything(arguments.seed, True)
    trainer(arguments)