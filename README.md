# Homework 2 - Chinese QA

## Environment
same as environment provided by TA
## setting
download all model
```
bash download.sh
```

## Train context selection
```
python3.8 preprocess_context.py --context_path path/to/context \
--data_path path/to/train_data --split train
python3.8 preprocess_context.py --context_path path/to/context \
--data_path path/to/valid_data --split valid
python3.8 train_context.py --device cuda
```
## Train QA
```
python3.8 preprocess_QA.py --split train
python3.8 preprocess_QA.py --split valid
python3.8 train_QA.py --device cuda
```
## Test QA
```
bash ./run.sh ./path/to/context.json ./path/to/test.json ./path/to/pred.csv
```