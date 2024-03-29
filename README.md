# MMT-ABSA
This repository contains source code to our paper:
- A Mutual Mean Teacher Framework for Cross-Domain Aspect-Based Sentiment Analysis

In this paper, we propose a mutual mean teacher framework with InfoNCE-based mutual information maximization for both tasks of cross-domain End2End ABSA and cross-domain aspect extraction. To mitigate the error amplification of the mean teacher, we make the student network receive the supervision of pseudo labels given by the peer mean teacher network while its own mean teacher network tracks an exponential moving average (EMA) of the student network, which helps generate more robust and reliable pseudo
labels for target domain data. On the other hand, we estimate the lower bound of the mutual information between representations of input tokens and output labels with InfoNCE, thus the domain-invariant feature can be learned during training. Extensive experiments on ten different transfer dataset pairs show the effectiveness of our framework.

<div align=center><img src="./framework.svg"></div>

Before running our code, please conduct following environment settings, this may take a few minutes.
```shell
conda env create -f env.yaml
```
## 1. Data preprocess
We use the orginal dataset from [BERT-UDA](https://github.com/NUSTM/BERT-UDA).The pos tags or dependency relations are not required in our paper, so we only retain the first two columns in the dataset files, and we have processed them in the `data` directory. 

We need to split the original training datafile into train set and validation set, and generate corresponding contrast data for the peer mean-teacher module. You can run the following shell for above preprocessing:
```shell
./scripts/preprocess.sh
```
For simplicity, we have provided the processed dataset in the `processed` directory, you can use it directly.
## 2. Running
Training on one transfer pair dataset:
```shell
./scripts/temp.sh [source_domain] [target_domain]
# source domain can be rest, laptop, service, device, the same for target domain.
# eg: ./scripts/temp.sh rest laptop
```
To understand the meaning of each parameter in the script, please run:
```shell
python trainers.py --help
```
<i>Please insure the parameters in the scripts are set properly before training.</i> Our code is based on the pytorch-lightning, thus the command line parameters are compatible with all the parameters in pytorch-lightning. For the specific model/training parameters, we write the code as the pattern in [HYPERPARAMETERS via commmand-line](https://lightning.ai/docs/pytorch/1.6.0/common/hyperparameters.html)

Here are some explainations for files:

- [trainers.py](./trainers.py) contains a augment parser and three pytorch lightning(PL) model trainers.
- [models.py](./models.py) contains the implementations for three models in the PL style:
    - BertClassifier: bert base model
    - MIBertClassifier: FMIM model with InfoNCE
    - MMTModel: MMT-MI model
- [model.py](./model.py) contains the implementations for models in the pytorch style
- [hparams.ini](./hparams.ini) contains the hyper-parameter settings for different transfer pairs.
- [mi_estimators.py](./mi_estimators.py) contains the implementations for many MI estimators follow [CLUB](https://github.com/Linear95/CLUB/blob/master/mi_estimators.py).
- [mi_estimation.ipynb](./mi_estimation.ipynb) contains the experiments of the MI estimation quality follow [CLUB](https://github.com/Linear95/CLUB/blob/master/mi_estimation.ipynb), but we change the $f(x, z)$ in InfoNCE with a Bilinear transformation instead of a linear sequential modules.

