import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parent

from data_loader import LogCollator, LogDataset
from metrics import compute_metrics, format_report
from model import ModernBertClassifier


# đổi checkpoint và dataset ở đây
DATASET        = "BGL"
CKPT_PATH      = "runs/checkpoint1/best.pt"
TEST_CSV       = "data/BGL/test.csv"
VAL_CSV        = "data/BGL/val.csv"
MODEL_PATH     = "models/ModernBERT-large"

BATCH_SIZE     = 8
TUNE_THRESHOLD = True


def _env_bool_eval(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


CKPT_PATH      = os.environ.get("MODERA_CKPT", CKPT_PATH)
DATASET        = os.environ.get("MODERA_DATASET", DATASET)
BATCH_SIZE     = int(os.environ.get("MODERA_EVAL_BATCH", BATCH_SIZE))
if "MODERA_DATASET" in os.environ:
    TEST_CSV = f"data/{DATASET}/test.csv"
    VAL_CSV  = f"data/{DATASET}/val.csv"
TUNE_THRESHOLD = _env_bool_eval("MODERA_TUNE_THRESHOLD", TUNE_THRESHOLD)


# tạo thư mục runs/evalN
def next_run_dir(base, prefix="eval"):
    base.mkdir(parents=True, exist_ok=True)
    nums = [int(d.name[len(prefix):]) for d in base.iterdir()
            if d.is_dir() and d.name.startswith(prefix) and d.name[len(prefix):].isdigit()]
    out = base / f"{prefix}{(max(nums) + 1) if nums else 1}"
    out.mkdir(parents=True, exist_ok=True)
    return out


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        labels = batch.pop("labels")
        inputs = {k: v.to(device) for k, v in batch.items()}
        out = model(**inputs)
        all_logits.append(out.logits.float().cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


# chọn ngưỡng trên val, xong mới chấm test
def tune_threshold(val_probs_anom: np.ndarray, val_is_anom: np.ndarray) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(val_is_anom, val_probs_anom)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    f1_arr = f1[:-1]
    if len(f1_arr) == 0:
        return 0.5, 0.0
    best_idx = int(np.nanargmax(f1_arr))
    return float(thr[best_idx]), float(f1_arr[best_idx])


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")

    model_path = ROOT / MODEL_PATH
    test_csv = ROOT / TEST_CSV
    val_csv = ROOT / VAL_CSV
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
    anomaly_id = label2id.get("anomaly", 1 - normal_id)
    pooling = cfg_meta.get("pooling", "mean")
    max_token_len = int(cfg_meta.get("max_token_len", 2048))
    ckpt_attn = cfg_meta.get("attn_implementation", None)
    use_mlp_head = bool(cfg_meta.get("use_mlp_head", False))
    head_hidden = int(cfg_meta.get("head_hidden", 512))
    head_dropout = float(cfg_meta.get("head_dropout", 0.2))
    use_proj_head = bool(cfg_meta.get("use_proj_head", False))
    proj_hidden = int(cfg_meta.get("proj_hidden", 256))
    proj_out = int(cfg_meta.get("proj_out", 128))

    ckpt_dataset = cfg_meta.get("dataset")
    if ckpt_dataset is not None and ckpt_dataset != DATASET:
        raise ValueError(
            f"ckpt dataset mismatch: trained on '{ckpt_dataset}', requested eval on '{DATASET}'. "
            f"Set MODERA_DATASET={ckpt_dataset} or use a different checkpoint."
        )
    ckpt_model_path = cfg_meta.get("model_path")
    if ckpt_model_path:
        model_path = ROOT / ckpt_model_path

    print(f"eval {DATASET} <- {ckpt_path}")
    print(f"  K={len(label2id)} pooling={pooling} max_token={max_token_len} attn={ckpt_attn}")
    print(f"  use_mlp_head={use_mlp_head} use_proj_head={use_proj_head} tune_thresh={TUNE_THRESHOLD}")
    print(f"  out: {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    test_ds = LogDataset(str(test_csv), tokenizer=tokenizer, max_token_len=max_token_len)
    collator = LogCollator(tokenizer=tokenizer, label2id=label2id)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator,
        num_workers=4, pin_memory=True, drop_last=False,
    )

    model = ModernBertClassifier(
        str(model_path), num_labels=len(label2id), pooling=pooling,
        use_mlp_head=use_mlp_head, head_hidden=head_hidden, head_dropout=head_dropout,
        use_proj_head=use_proj_head, proj_hidden=proj_hidden, proj_out=proj_out,
    ).to(device)
    model.load_state_dict(state)
    if ckpt_attn is not None and model.attn_implementation != ckpt_attn:
        print(f"  WARN attn_implementation mismatch: ckpt={ckpt_attn} runtime={model.attn_implementation}")

    threshold_meta = {"tuned": False, "threshold": 0.5, "val_f1_at_threshold": None}
    val_anom_rate = None
    val_probs_anom = None
    if TUNE_THRESHOLD and val_csv.exists():
        print(f"  tuning threshold on val: {val_csv}")
        val_ds = LogDataset(str(val_csv), tokenizer=tokenizer, max_token_len=max_token_len)
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator,
            num_workers=4, pin_memory=True, drop_last=False,
        )
        val_logits, val_labels = collect_logits(model, val_loader, device)
        val_probs_anom = F.softmax(val_logits, dim=-1)[:, anomaly_id].numpy()
        val_is_anom = (val_labels.numpy() == anomaly_id).astype(np.int64)
        val_anom_rate = float(val_is_anom.mean())
        best_thr, best_val_f1 = tune_threshold(val_probs_anom, val_is_anom)
        threshold_meta = {"tuned": True, "threshold": best_thr, "val_f1_at_threshold": best_val_f1,
                          "val_anomaly_rate": val_anom_rate}
        print(f"  val-tuned threshold = {best_thr:.4f} (val F1 = {best_val_f1:.4f}, val anomaly rate = {val_anom_rate:.4f})")

    test_logits, test_labels = collect_logits(model, test_loader, device)
    test_probs = F.softmax(test_logits, dim=-1).numpy()
    labels_np = test_labels.numpy()
    if TUNE_THRESHOLD and threshold_meta["tuned"]:
        thr = threshold_meta["threshold"]
        preds_np = np.where(test_probs[:, anomaly_id] >= thr, anomaly_id, normal_id).astype(np.int64)
    else:
        preds_np = test_probs.argmax(axis=-1).astype(np.int64)

    metrics = compute_metrics(preds_np, labels_np, id2label, normal_id)
    metrics["source_ckpt"] = CKPT_PATH
    metrics["eval_dataset"] = DATASET
    metrics["max_token_len_used"] = max_token_len
    metrics["normal_id"] = int(normal_id)
    metrics["anomaly_id"] = int(anomaly_id)
    metrics["label2id"] = {k: int(v) for k, v in label2id.items()}
    metrics["threshold"] = threshold_meta

    test_anom_rate = float((labels_np == anomaly_id).mean())
    metrics["test_anomaly_rate"] = test_anom_rate
    if val_anom_rate is not None:
        rate_gap = abs(val_anom_rate - test_anom_rate)
        if rate_gap > 0.10:
            print(f"  WARN val/test anomaly rate gap = {rate_gap:.4f} (val={val_anom_rate:.4f} test={test_anom_rate:.4f}) "
                  f"-- val-tuned threshold may not transfer (distribution shift)")

    report = "\n".join(format_report(metrics, task_label=f"{DATASET} <- {CKPT_PATH}"))
    if threshold_meta["tuned"]:
        report += f"\n\n  Note: threshold tuned on val = {threshold_meta['threshold']:.4f} (val F1 = {threshold_meta['val_f1_at_threshold']:.4f})\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(report)
    (out_dir / "report.json").write_text(json.dumps(metrics, indent=2))
    np.save(out_dir / "preds.npy", preds_np)
    np.save(out_dir / "labels.npy", labels_np)
    np.save(out_dir / "test_probs.npy", test_probs)
    if val_probs_anom is not None:
        np.save(out_dir / "val_probs.npy", val_probs_anom)

    print()
    print(report)
    print(f"\nsaved -> {out_dir}/report.txt , report.json , preds.npy , labels.npy , test_probs.npy")


if __name__ == "__main__":
    main()
