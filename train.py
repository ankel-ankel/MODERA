from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parent

from data_loader import LogCollator, LogDataset, balanced_sampler
from model import ModernBertClassifier
from trainer import train



DATASET          = "BGL"
TRAIN_CSV        = "data/BGL/train.csv"
TEST_CSV         = "data/BGL/test.csv"
MODEL_PATH       = "models/ModernBERT-large"

EPOCHS           = 4
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 4
TEST_BATCH_SIZE  = 16
LR               = 5e-5
MAX_TOKEN_LEN    = 2048
SEED             = 42

ALPHA_SUPCON     = 0.5
LABEL_SMOOTHING  = 0.1
POOLING_TYPE     = "attention"
LLRD_DECAY       = 0.9
TARGET_RATIO     = 0.4

BALANCED_SAMPLER = True
STABLE_ADAMW     = True
LLRD             = True
SWA              = True
RESUME_DIR       = None



def next_run_dir(base, prefix="checkpoint"):
    base.mkdir(parents=True, exist_ok=True)
    nums = [int(d.name[len(prefix):]) for d in base.iterdir()
            if d.is_dir() and d.name.startswith(prefix) and d.name[len(prefix):].isdigit()]
    out = base / f"{prefix}{(max(nums) + 1) if nums else 1}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_csv = ROOT / TRAIN_CSV
    test_csv = ROOT / TEST_CSV
    model_path = ROOT / MODEL_PATH

    if RESUME_DIR is not None:
        output_dir = ROOT / RESUME_DIR
        if not output_dir.exists():
            raise FileNotFoundError(f"Cannot resume — {output_dir} does not exist")
        resume = True
    else:
        output_dir = next_run_dir(ROOT / "runs")
        resume = False

    sampler_desc = f"balanced@{TARGET_RATIO}" if BALANCED_SAMPLER else "shuffle"
    print(f"train {DATASET} bs={BATCH_SIZE} ep={EPOCHS} lr={LR} "
          f"alpha_supcon={ALPHA_SUPCON} sampler={sampler_desc}")
    print(f"checkpoints saved to {output_dir}")

    train_ds = LogDataset(str(train_csv))
    test_ds = LogDataset(str(test_csv))

    all_labels = sorted(set(train_ds.labels) | set(test_ds.labels))
    label2id = {lbl: i for i, lbl in enumerate(all_labels)}
    if "normal" not in label2id:
        label2id = {"normal": 0, **{k: v + 1 for k, v in label2id.items()}}
    print(f"  K={len(label2id)}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    collator = LogCollator(tokenizer=tokenizer, label2id=label2id, max_token_len=MAX_TOKEN_LEN)

    if BALANCED_SAMPLER:
        loader_kw = {"sampler": balanced_sampler(train_ds.labels, target_ratio=TARGET_RATIO)}
    else:
        loader_kw = {"shuffle": True}

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, collate_fn=collator,
        num_workers=4, persistent_workers=True, pin_memory=True,
        drop_last=True, **loader_kw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collator,
        num_workers=4, persistent_workers=False, pin_memory=True, drop_last=False,
    )

    model = ModernBertClassifier(str(model_path), num_labels=len(label2id),
                                  pooling=POOLING_TYPE).to(device)

    summary = train(
        model, train_loader, test_loader, label2id, output_dir, device,
        epochs=EPOCHS, lr=LR, alpha_supcon=ALPHA_SUPCON,
        label_smoothing=LABEL_SMOOTHING, resume=resume,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        stable_adamw=STABLE_ADAMW,
        llrd=LLRD, llrd_decay=LLRD_DECAY,
        swa=SWA,
        extra_meta={
            "dataset": DATASET, "batch_size": BATCH_SIZE, "max_token_len": MAX_TOKEN_LEN,
            "model_path": MODEL_PATH, "balanced_sampler": BALANCED_SAMPLER,
            "target_ratio": TARGET_RATIO if BALANCED_SAMPLER else None, "seed": SEED,
            "pooling": POOLING_TYPE,
        },
    )
    print(f"DONE bf1={summary['final_binary_f1']:.4f} dir={output_dir}")


if __name__ == "__main__":
    main()
