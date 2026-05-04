from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parent

from data_loader import LogADCollator, LogADDataset, make_binary_balanced_sampler
from model import ModernBERTForLogAD
from trainer import train


# Inline comments below mark P1 (max hardware) vs P2 (lean cited).
# Lines without P1/P2 comments are the same for both plans.
DATASET            = "BGL"
EPOCHS             = 10        # P2: 10 | P1: 15
BATCH_SIZE         = 8 
LR                 = 3e-5      # P2: 3e-5 | P1: 5e-5
WEIGHT_DECAY       = 0.01
WARMUP_RATIO       = 0.1       # P2: 0.1 | P1: 0.06
GRAD_ACCUM_STEPS   = 1         # P2: 1 | P1: 2 (effective batch = 16)
GRAD_CHECKPOINT    = False

ALPHA_SUPCON       = 0.5       # P2: 0.5 | P1: 1.0
SUPCON_TEMPERATURE = 0.07

USE_FOCAL          = False     # P2: False | P1: True
FOCAL_GAMMA        = 2.0
LABEL_SMOOTHING    = 0.1

USE_CLASS_WEIGHTS  = False     # P2: False | P1: True
USE_BALANCED_SAMPLER = True
TARGET_RATIO       = 0.3

MAX_TOKEN_LEN      = 1024
JOIN_SEP           = " ;; "

NUM_WORKERS        = 4
PIN_MEMORY         = True
PERSISTENT_WORKERS = True
LOG_EVERY          = 50
SAVE_EVERY         = 1

SEED               = 42
SMOKE_N            = None  # number of samples for a quick test run; set to None to use the full dataset

RESUME_RUN         = None  # "run1" to resume from the latest checkpoint
PATIENCE           = 3     # stop if val f1_macro doesn't improve for this many epochs; None to disable

MODEL_PATH         = "models/ModernBERT-large"
ATTN_IMPL          = "sdpa"
DTYPE              = "bfloat16"
TRAIN_CSV_TPL      = "data/{dataset}/train.csv"
TEST_CSV_TPL       = "data/{dataset}/test.csv"


def next_run_dir(base):
    base.mkdir(parents=True, exist_ok=True)
    nums = [int(d.name[3:]) for d in base.iterdir()
            if d.is_dir() and d.name.startswith("run") and d.name[3:].isdigit()]
    out = base / f"run{(max(nums) + 1) if nums else 1}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def stratified_indices(labels, n_total, rng):
    classes = np.unique(labels)
    per = max(1, n_total // len(classes))
    chosen = np.concatenate([
        rng.choice(np.where(labels == c)[0], size=min(per, (labels == c).sum()), replace=False)
        for c in classes
    ])
    if len(chosen) > n_total:
        chosen = rng.choice(chosen, n_total, replace=False)
    return chosen


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    is_smoke = SMOKE_N is not None
    num_workers = 0 if is_smoke else NUM_WORKERS
    persistent_workers = num_workers > 0 and PERSISTENT_WORKERS

    train_csv = ROOT / TRAIN_CSV_TPL.format(dataset=DATASET)
    test_csv = ROOT / TEST_CSV_TPL.format(dataset=DATASET)
    model_path = ROOT / MODEL_PATH
    if RESUME_RUN is not None:
        output_dir = ROOT / "checkpoints" / DATASET / RESUME_RUN
        if not output_dir.exists():
            raise FileNotFoundError(f"Cannot resume — {output_dir} does not exist")
    else:
        output_dir = next_run_dir(ROOT / "checkpoints" / DATASET)

    print(f"train {DATASET} bs={BATCH_SIZE} ep={EPOCHS} alpha_supcon={ALPHA_SUPCON} "
          f"focal={USE_FOCAL} cw={USE_CLASS_WEIGHTS} "
          f"sampler={'binary@' + str(TARGET_RATIO) if USE_BALANCED_SAMPLER else 'shuffle'}")
    print(f"  out: {output_dir}")

    train_ds = LogADDataset(str(train_csv))
    test_ds = LogADDataset(str(test_csv))

    if is_smoke:
        rng = np.random.default_rng(SEED)
        tr_idx = stratified_indices(train_ds.labels, SMOKE_N, rng)
        train_ds.sequences = train_ds.sequences[tr_idx]
        train_ds.labels = train_ds.labels[tr_idx]
        print(f"  smoke: train={len(train_ds)} test={len(test_ds)} (full)")

    all_labels = sorted(set(train_ds.labels) | set(test_ds.labels))
    label2id = {lbl: i for i, lbl in enumerate(all_labels)}
    if "normal" not in label2id:
        label2id = {"normal": 0, **{k: v + 1 for k, v in label2id.items()}}
    print(f"  K={len(label2id)}")

    train_label_ids = np.array([label2id[lbl] for lbl in train_ds.labels])

    class_weights = None
    if USE_CLASS_WEIGHTS:
        present = np.unique(train_label_ids)
        cw = compute_class_weight("balanced", classes=present, y=train_label_ids)
        full = np.ones(len(label2id), dtype=np.float32)
        full[present] = cw
        class_weights = torch.tensor(full, dtype=torch.float32, device=device)
        print(f"  class_weights range [{full.min():.3f}, {full.max():.3f}]")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    collator = LogADCollator(
        tokenizer=tokenizer, label2id=label2id,
        max_token_len=MAX_TOKEN_LEN, join_sep=JOIN_SEP,
    )

    if USE_BALANCED_SAMPLER:
        loader_kw = {"sampler": make_binary_balanced_sampler(train_ds.labels, target_ratio=TARGET_RATIO)}
    else:
        loader_kw = {"shuffle": True}

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, collate_fn=collator,
        num_workers=num_workers, persistent_workers=persistent_workers,
        pin_memory=PIN_MEMORY, drop_last=True, **loader_kw,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator,
        num_workers=num_workers, persistent_workers=False,
        pin_memory=PIN_MEMORY, drop_last=False,
    )

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[DTYPE]
    model = ModernBERTForLogAD(
        str(model_path), num_labels=len(label2id),
        attn_implementation=ATTN_IMPL, dtype=dtype,
    ).to(device)

    summary = train(
        model, train_loader, test_loader, label2id, output_dir, device,
        epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO, grad_accum_steps=GRAD_ACCUM_STEPS,
        grad_checkpoint=GRAD_CHECKPOINT,
        alpha_supcon=ALPHA_SUPCON, supcon_temperature=SUPCON_TEMPERATURE,
        use_focal=USE_FOCAL, focal_gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING, class_weights=class_weights,
        log_every=LOG_EVERY, save_every=SAVE_EVERY,
        patience=PATIENCE, resume=RESUME_RUN is not None,
        extra_meta={
            "dataset": DATASET, "batch_size": BATCH_SIZE,
            "max_token_len": MAX_TOKEN_LEN, "model_path": MODEL_PATH,
            "attn_impl": ATTN_IMPL, "dtype": DTYPE,
            "use_balanced_sampler": USE_BALANCED_SAMPLER,
            "target_ratio": TARGET_RATIO if USE_BALANCED_SAMPLER else None,
            "use_class_weights": USE_CLASS_WEIGHTS,
            "join_sep": JOIN_SEP, "seed": SEED,
        },
    )
    print(f"DONE best_f1m={summary['best_f1_macro']:.4f} dir={output_dir}")


if __name__ == "__main__":
    main()
