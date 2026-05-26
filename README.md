# MODERA

Binary log anomaly detection with ModernBERT-large.

## Backbone

[`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) — 395M params, 8K context.

```bash
huggingface-cli download answerdotai/ModernBERT-large --local-dir models/ModernBERT-large
```

## Datasets

Structured CSVs from [LogHub](https://github.com/logpai/loghub) (BGL, HDFS_v1, Liberty, Thunderbird). Place under `data/{dataset}/{name}.log_structured.csv`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

```bash
python build_dataset.py     # raw structured CSV -> windowed train/test
python train.py             # fine-tune ModernBERT-large
python eval.py              # evaluate the trained checkpoint
```

Hyperparameters and dataset selection live as constants at the top of each script.

## Results

BGL test split (9,427 windows, 8.65% anomaly rate):

| | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| **MODERA** | **0.9887** | **0.9693** | **0.9789** | **0.9964** |
