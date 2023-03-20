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
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback

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
parser.add_argument("--tune",
                    default=False,
                    action='store_true',
                    help='Whether tune hyperparameters with the Ray.')
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


def mmt_model_trainer(config: dict, args):
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(args.seed, True)
    train_loader, val_loader, test_loader, tokenizer = dataloader_init(args)
    dict_args = vars(args)
    dict_args['tokenizer'] = tokenizer
    dict_args['num_labels'] = len(TAGS)
    dict_args.update(config)
    model = MMTModel(**dict_args)
    callbacks = []
    checkpoint_callback = ModelCheckpoint(monitor="val_f1",
                                          mode='max',
                                          filename='mmt-absa-{epoch:02d}-{val_f1:.2f}')
    callbacks.append(checkpoint_callback)
    if config:
        tune_report_callback = TuneReportCallback(['absa_test_f1', 'ae_test_f1'], on="test_epoch_end")
        callbacks.append(tune_report_callback)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path='best')


def mi_bert_trainer(config: dict, args):
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(args.seed, True)
    train_loader, val_loader, test_loader, tokenizer = dataloader_init(args)
    dict_args = vars(args)
    dict_args['tokenizer'] = tokenizer
    dict_args['num_labels'] = len(TAGS)
    dict_args.update(config)
    model = MIBertClassifier(**dict_args)
    callbacks = []
    checkpoint_callback = ModelCheckpoint(monitor="val_f1",
                                          mode='max',
                                          filename='mi-absa-{epoch:02d}-{val_f1:.2f}')
    callbacks.append(checkpoint_callback)
    if config:
        tune_report_callback = TuneReportCallback(['absa_test_f1', 'ae_test_f1'],
                                                  on="test_epoch_end")
        callbacks.append(tune_report_callback)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path='best')


def bert_trainer(config: dict, args):
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(args.seed, True)
    train_loader, val_loader, test_loader, tokenizer = dataloader_init(args)
    dict_args = vars(args)
    dict_args['num_labels'] = len(TAGS)
    dict_args['tokenizer'] = tokenizer
    dict_args.update(config)
    model = BertClassifier(**dict_args)
    callbacks = []
    checkpoint_callback = ModelCheckpoint(monitor="val_f1",
                                          mode='max',
                                          filename='bert-absa-{epoch:02d}-{val_f1:.2f}',
                                          save_weights_only=True)
    callbacks.append(checkpoint_callback)
    if config:
        tune_report_callback = TuneReportCallback(['absa_test_f1', 'ae_test_f1'],
                                                  on="test_epoch_end")
        callbacks.append(tune_report_callback)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path='best')


if __name__ == '__main__':
    temp_args, _ = parser.parse_known_args()
    model_name = temp_args.model_name
    trainers = {
        "mmt": mmt_model_trainer,
        "mi_bert": mi_bert_trainer,
        "bert": bert_trainer,
    }
    models = {"mmt": MMTModel, "mi_bert": MIBert, "bert": BertClassifier}
    parser = models[model_name].add_model_specific_args(parser)
    trainer = trainers[model_name]
    arguments = parser.parse_args()
    if temp_args.tune:
        train_fn_with_parameters = tune.with_parameters(trainer, args=arguments)
        resources_per_trial = {"gpu": int(arguments.devices)}
        reporter = CLIReporter(metric_columns=["absa_test_f1", "ae_test_f1"])
        tuner = tune.Tuner(
            tune.with_resources(train_fn_with_parameters, resources=resources_per_trial),
            param_space={
                "alpha": tune.grid_search([0.005, 0.01, 0.015, 0.02, 0.025, 0.03]),
                "tau": tune.grid_search([0.2, 0.4, 0.6, 0.8, 1, 1.2])
            },
            run_config=air.RunConfig(
                name=model_name,
                progress_reporter=reporter,
            ),
        )
        results = tuner.fit()
        print("Best hyperparameters found were: ",
              results.get_best_result("absa_test_f1", "max").config)
    else:
        trainer({}, arguments)