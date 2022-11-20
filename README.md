# graduation
## 0. Env setting
```shell
conda env create -f conda_env.yaml
```
## 1. Data preprocess
```shell
./scripts/preprocess.sh
```

## 2. Running models
Bert-base model
```shell
./scripts/run_bert_base.sh
```
MMT model
```shell
./scripts/run_mmt.sh
```