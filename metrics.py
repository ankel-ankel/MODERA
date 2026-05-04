from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    id2label: dict[int, str],
    normal_id: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_samples": len(labels),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(labels, preds, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(labels, preds, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }

    pred_anom = (preds != normal_id).astype(int)
    true_anom = (labels != normal_id).astype(int)
    out["binary_accuracy"] = float(accuracy_score(true_anom, pred_anom))
    out["binary_precision"] = float(precision_score(true_anom, pred_anom, zero_division=0))
    out["binary_recall"] = float(recall_score(true_anom, pred_anom, zero_division=0))
    out["binary_f1"] = float(f1_score(true_anom, pred_anom, zero_division=0))
    out["tp"] = int(((pred_anom == 1) & (true_anom == 1)).sum())
    out["fp"] = int(((pred_anom == 1) & (true_anom == 0)).sum())
    out["fn"] = int(((pred_anom == 0) & (true_anom == 1)).sum())
    out["tn"] = int(((pred_anom == 0) & (true_anom == 0)).sum())
    out["n_normal"] = int((true_anom == 0).sum())
    out["n_anomaly"] = int((true_anom == 1).sum())
    out["pred_normal"] = int((pred_anom == 0).sum())
    out["pred_anomaly"] = int((pred_anom == 1).sum())

    label_ids = sorted(set(labels.tolist()) | set(preds.tolist()))
    p_arr = precision_score(labels, preds, average=None, labels=label_ids, zero_division=0)
    r_arr = recall_score(labels, preds, average=None, labels=label_ids, zero_division=0)
    f_arr = f1_score(labels, preds, average=None, labels=label_ids, zero_division=0)
    out["per_class"] = {
        id2label.get(i, f"class_{i}"): {
            "precision": float(p), "recall": float(r), "f1": float(f),
            "support_true": int((labels == i).sum()),
            "support_pred": int((preds == i).sum()),
        }
        for i, p, r, f in zip(label_ids, p_arr, r_arr, f_arr)
    }
    pred_types = Counter(id2label.get(int(p), f"class_{int(p)}") for p in preds if int(p) != normal_id)
    out["pred_type_counts"] = dict(pred_types.most_common())
    return out


def format_report(metrics: dict[str, Any], task_label: str) -> list[str]:
    n = metrics["n_samples"]
    n_norm = metrics["n_normal"]
    n_anom = metrics["n_anomaly"]
    p_norm = metrics["pred_normal"]
    p_anom = metrics["pred_anomaly"]
    pct_true = 100 * n_anom / max(n, 1)
    pct_pred = 100 * p_anom / max(n, 1)

    lines = [
        "==============================================================",
        f"  Anomaly Detection Report: {task_label}",
        "==============================================================",
        "",
        "  Dataset summary",
        "  --------------------------------------------------------------",
        f"    Sequences scanned : {n:,}",
        f"    Ground truth      : {n_norm:,} normal | {n_anom:,} anomaly | {pct_true:.2f}% anomaly rate",
        f"    Predicted         : {p_norm:,} normal | {p_anom:,} anomaly | {pct_pred:.2f}% predicted as anomaly",
        "",
        "  Binary metrics",
        "  --------------------------------------------------------------",
        f"    precision : {metrics['binary_precision']:.4f}",
        f"    recall    : {metrics['binary_recall']:.4f}",
        f"    f1        : {metrics['binary_f1']:.4f}",
        f"    accuracy  : {metrics['binary_accuracy']:.4f}",
        "",
        f"    TP = {metrics['tp']:>7,}    FN = {metrics['fn']:>7,}",
        f"    FP = {metrics['fp']:>7,}    TN = {metrics['tn']:>7,}",
        "",
        "  Confusion (counts)",
        "  --------------------------------------------------------------",
        f"    {'':<16} {'Normal':>14} {'Anomaly':>14}",
        f"    {'Ground truth':<16} {n_norm:>14,} {n_anom:>14,}",
        f"    {'Prediction':<16} {p_norm:>14,} {p_anom:>14,}",
    ]

    if metrics.get("pred_type_counts"):
        lines += [
            "",
            "  Predicted anomaly types (sorted by count)",
            "  --------------------------------------------------------------",
        ]
        for cls, cnt in metrics["pred_type_counts"].items():
            lines.append(f"    {cls:<28} {cnt:>10,}")

    lines.append("==============================================================")
    return lines
