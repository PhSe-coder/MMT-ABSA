from dataclasses import dataclass, field
from typing import List, Optional
from transformers import TrainingArguments
from constants import SUPPORTED_MODELS


@dataclass
class ModelArguments(TrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    init_1: str = field(default=None,
                        metadata={"help": "pretrained model used to initialize the mmt model 1"})
    init_2: str = field(default=None,
                        metadata={"help": "pretrained model used to initialize the mmt model 2"})

    model_name: Optional[str] = field(default="bert",
                                      metadata={"help": "The name of the model (ner, pos...)."})
    pretrained_model: Optional[str] = field(
        default='bert-base-uncased',
        metadata={"help": "The name of the dataset to use (via the datasets library)."})
    best_model_path: Optional[str] = field(default=None,
                                           metadata={"help": "path of the best model"})
    num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=100,
        metadata={
            "help":
            ("The maximum total input sequence length after tokenization. If set, sequences longer "
             "than this will be truncated, sequences shorter will be padded.")
        },
    )
    train_file: Optional[List[str]] = field(
        default=None, metadata={"help": "The input training data files (txt file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a txt file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a txt file)."},
    )
    batch_size: Optional[int] = field(default=16, metadata={"help": 'training batch size'})
    dropout: Optional[float] = field(default=0.1)
    alpha: Optional[float] = field(
        default=0.02, metadata={"help": 'the weight parameter of the Mutual Information loss'})
    bert_lr: Optional[float] = field(default=2e-5,
                                     metadata={"help": "learning rate for pretrained model"})
    lr: Optional[float] = field(default=1e-3, metadata={"help": "learning rate for custom modules"})
    l2reg: Optional[float] = field(default=0.01, metadata={"help": "weight decay"})
    warmup: Optional[float] = field(
        default=0.1,
        metadata={"help": "Proportion of training to perform linear learning rate warmup for."})
    max_seq_len: Optional[int] = field(default=100)
    optimizer: Optional[str] = field(default='adam')
    initializer: Optional[str] = field(default='kaiming_uniform_',
                                       metadata={"help": "initializer for customize parameters"})
    logging_steps: Optional[int] = field(
        default=10, metadata={"help": "Number of update steps between two logs"})
    patience: Optional[int] = field(default=5, metadata={"help": 'early stopping rounds'})

    def __post_init__(self):
        # if self.model_name == 'mmt' and (self.init_1 is None or self.init_2 is None):
        #     raise ValueError("init_1 and init_2 arguments are needed.")
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Model name `{self.model_name}` is not supported.")
        if self.do_train and self.train_file is None:
            raise ValueError("Need training file when `do_train` is set.")
        if self.do_eval and self.validation_file is None:
            raise ValueError("Need validation file when `do_eval` is set.")
        if self.do_predict and self.test_file is None:
            raise ValueError("Need test file when `do_eval` is set.")
        if self.train_file is None:
            raise ValueError("Need train file.")
        if self.best_model_path is None and self.do_predict and not self.do_train:
            raise ValueError(
                "Argument `best_model_path` is needed when `do_eval` is True and `do_train` is False"
            )
