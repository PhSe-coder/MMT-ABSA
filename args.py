from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    model_name: Optional[str] = field(default="model", metadata={"help": "The name of the model (ner, pos...)."})
    pretrained_model: Optional[str] = field(
        default='bert-base-uncased', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a txt file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a txt file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a txt file)."},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": 'set seed for reproducibility'}),
    batch_size: Optional[int] = field(
        default=16,
        metadata={"help": 'training batch size'}),
    num_epoch: Optional[int] = field(default=20, metadata={'help': 'try larger number for non-BERT models'}),
    dropout: Optional[float] = field(default=0.1),
    lr: Optional[float] = field(default=2e-5, metadata={"help": "learning rate"}),
    l2reg: Optional[float] = field(default=0.01),
    max_seq_len: Optional[int] = field(default=100),
    optimizer: Optional[str] = field(default='adam'),
    initializer: Optional[str] = field(default='xavier_uniform_', 
        metadata={"help": "initializer for customize parameters"}),
    logging_steps: Optional[int] = field(default=10, metadata={"help": "Number of update steps between two logs"}),
    patience: Optional[int] = field(default=5, metadata={"help": 'early stopping rounds'})

    def __post_init__(self):
        if self.train_file is None or self.validation_file is None:
            raise ValueError("Need  training, validation, test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["txt"], "`train_file` should be a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["txt"], "`validation_file` should be a txt file."