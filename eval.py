import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parent

from data_loader import LogADCollator, LogADDataset
from metrics import format_report
from model import ModernBERTForLogAD
from trainer import evaluate


DATASET        = "BGL"
RUN            = "run1"
CHECKPOINT     = "best"
BATCH_SIZE     = 8
MAX_TOKEN_LEN  = 1024
JOIN_SEP       = " ;; "
NUM_WORKERS    = 4
PIN_MEMORY     = True

SEED           = 42
SMOKE_N        = None  # number of test samples for a quick eval; set to None to use the full test set

MODEL_PATH     = "models/ModernBERT-large"
ATTN_IMPL      = "sdpa"
DTYPE          = "bfloat16"
TEST_CSV_TPL   = "data/{dataset}/test.csv"


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
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[DTYPE]

    model_path = ROOT / MODEL_PATH
    ckpt_dir = ROOT / "checkpoints" / DATASET / RUN / CHECKPOINT
    if not ckpt_dir.exists():
        raise FileNotFoundError(ckpt_dir)

    label2id = json.loads((ckpt_dir / "label2id.json").read_text())
    id2label = {v: k for k, v in label2id.items()}
    normal_id = label2id.get("normal", 0)
    cfg_meta = json.loads((ckpt_dir / "config.json").read_text())
    attn = cfg_meta.get("attn_impl", ATTN_IMPL)

    print(f"eval {DATASET} <- {ckpt_dir}")
    print(f"  K={len(label2id)} attn={attn}")

    test_csv = ROOT / TEST_CSV_TPL.format(dataset=DATASET)
    test_ds = LogADDataset(str(test_csv))

    if SMOKE_N is not None:
        rng = np.random.default_rng(SEED)
        idx = stratified_indices(test_ds.labels, SMOKE_N, rng)
        test_ds.sequences = test_ds.sequences[idx]
        test_ds.labels = test_ds.labels[idx]
        print(f"  smoke: test={len(test_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    collator = LogADCollator(
        tokenizer=tokenizer, label2id=label2id,
        max_token_len=MAX_TOKEN_LEN, join_sep=JOIN_SEP,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False,
    )

    model = ModernBERTForLogAD(
        str(model_path), num_labels=len(label2id),
        attn_implementation=attn, dtype=dtype,
    ).to(device)
    state = torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)

    metrics = evaluate(model, test_loader, device, id2label, normal_id)

    out_dir = ROOT / "results" / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{RUN}_{CHECKPOINT}"
    report = "\n".join(format_report(metrics, task_label=f"{DATASET} {tag}"))
    (out_dir / f"{tag}.txt").write_text(report)
    (out_dir / f"{tag}.json").write_text(json.dumps(metrics, indent=2))

    print()
    print(report)
    print(f"\nsaved -> {out_dir}/{tag}.txt , {out_dir}/{tag}.json")


if __name__ == "__main__":
    main()
