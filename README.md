# MODERA

Multi-class log anomaly type prediction with ModernBERT.

Work in progress.

## Datasets

We follow the [LogHub](https://github.com/logpai/loghub) data scope. Window size 100 lines, chronological 80/20 train/test split.

|  | BGL | HDFS_v1 | Liberty | Thunderbird |
|---|---:|---:|---:|---:|
| Train sequences | 37,707 | 460,048 | 40,000 | 79,996 |
| Test sequences | 9,427 | 115,013 | 10,000 | 20,000 |
| Anomaly types | 34 | 1 | 10 | 8 |

Download the structured CSVs from LogHub and place them under:

```
data/BGL/BGL.log_structured.csv
data/HDFS_v1/HDFS.log_structured.csv
data/Liberty/liberty2_structured.csv
data/Thunderbird/Thunderbird.log_structured.csv
```

Then run `python build_dataset.py` to generate the windowed train/test CSVs.

## Pretrained model

Backbone: [`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) (395M params, 8K context, Apache 2.0).

Download via `huggingface-cli`:

```bash
huggingface-cli download answerdotai/ModernBERT-large --local-dir models/ModernBERT-large
```

Or any HF snapshot method. Place the snapshot under `models/ModernBERT-large/`.

## File layout

```
MODERA/
  build_dataset.py     # raw structured CSV -> windowed multi-class CSVs
  train.py             # training entrypoint
  eval.py              # evaluation on test split
  data_loader.py       # Dataset, regex mask, balanced sampler, collator
  losses.py            # SupConLoss, FocalLoss, CombinedLoss
  metrics.py           # P/R/F1, report formatter
  model.py             # ModernBERTForLogAD wrapper
  trainer.py           # training loop with resume + early stopping
  docs/                # session handoff notes
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121   # adjust for your CUDA
```

## Usage

### 1. Build the windowed CSVs

```bash
python build_dataset.py
```

Slices each structured log into 100-line windows, splits them 80/20 by time, and saves `train.csv` and `test.csv` next to the source.

### 2. Train

```bash
python train.py
```

Fine-tunes ModernBERT on the train split and saves checkpoints under `checkpoints/{dataset}/run{N}/`. The best epoch and final epoch are kept separately.

### 3. Evaluate

```bash
python eval.py
```

Loads a checkpoint and writes a report (text + JSON) to `results/{dataset}/`.

Each script keeps its knobs as plain constants at the top of the file — open it, change what you want, run it.