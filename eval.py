import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parent

from data_loader import LogCollator, LogDataset
from metrics import compute_metrics, format_report
from model import ModernBertClassifier


DATASET       = "BGL"
CKPT_PATH     = "runs/checkpoint7/checkpoint.pt"
TEST_CSV      = "data/BGL/test.csv"
MODEL_PATH    = "models/ModernBERT-large"

BATCH_SIZE    = 16
MAX_TOKEN_LEN = 2048


def next_run_dir(base, prefix="eval"):
    base.mkdir(parents=True, exist_ok=True)
    nums = [int(d.name[len(prefix):]) for d in base.iterdir()
            if d.is_dir() and d.name.startswith(prefix) and d.name[len(prefix):].isdigit()]
    out = base / f"{prefix}{(max(nums) + 1) if nums else 1}"
    out.mkdir(parents=True, exist_ok=True)
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        labels = batch.pop("labels")
        inputs = {k: v.to(device) for k, v in batch.items()}
        out = model(**inputs)
        all_preds.append(out.logits.argmax(-1).cpu())
        all_labels.append(labels)
    return torch.cat(all_preds), torch.cat(all_labels)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")

    model_path = ROOT / MODEL_PATH
    test_csv = ROOT / TEST_CSV
    ckpt_path = ROOT / CKPT_PATH
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    out_dir = next_run_dir(ROOT / "runs")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    label2id = ckpt["label2id"]
    cfg_meta = ckpt.get("config", {})
    id2label = {v: k for k, v in label2id.items()}
    normal_id = label2id.get("normal", 0)
    pooling = cfg_meta.get("pooling", "mean")

    print(f"eval {DATASET} <- {ckpt_path}")
    print(f"  K={len(label2id)} pooling={pooling} max_token={MAX_TOKEN_LEN}")
    print(f"  out: {out_dir}")

    test_ds = LogDataset(str(test_csv))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    collator = LogCollator(tokenizer=tokenizer, label2id=label2id, max_token_len=MAX_TOKEN_LEN)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator,
        num_workers=4, pin_memory=True, drop_last=False,
    )

    model = ModernBertClassifier(str(model_path), num_labels=len(label2id),
                                pooling=pooling).to(device)
    model.load_state_dict(state)

    preds, labels = evaluate(model, test_loader, device)

    metrics = compute_metrics(preds.numpy(), labels.numpy(), id2label, normal_id)
    metrics["source_ckpt"] = CKPT_PATH

    report = "\n".join(format_report(metrics, task_label=f"{DATASET} <- {CKPT_PATH}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(report)
    (out_dir / "report.json").write_text(json.dumps(metrics, indent=2))

    print()
    print(report)
    print(f"\nsaved -> {out_dir}/report.txt , {out_dir}/report.json")


if __name__ == "__main__":
    main()
