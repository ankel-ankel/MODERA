import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parent

# hai lần eval đem so, đặt qua env
EVAL_DIR_A    = "runs/eval1"
EVAL_DIR_B    = "runs/eval2"
N_BOOTSTRAP   = 10000
ALPHA         = 0.05
SEED          = 42

EVAL_DIR_A  = os.environ.get("MODERA_EVAL_A", EVAL_DIR_A)
EVAL_DIR_B  = os.environ.get("MODERA_EVAL_B", EVAL_DIR_B)
N_BOOTSTRAP = int(os.environ.get("MODERA_N_BOOTSTRAP", N_BOOTSTRAP))
SEED        = int(os.environ.get("MODERA_SEED", SEED))


# gộp về nhị phân, lấy lớp normal từ nhãn
def binary_collapse(preds: np.ndarray, labels: np.ndarray, normal_id: int):
    return (preds != normal_id).astype(np.int64), (labels != normal_id).astype(np.int64)


# khoảng tin cậy bootstrap
def bootstrap_f1_ci(preds_bin: np.ndarray, labels_bin: np.ndarray,
                    n_boot: int, alpha: float, rng: np.random.Generator):
    n = len(labels_bin)
    f1s = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1s[i] = f1_score(labels_bin[idx], preds_bin[idx], zero_division=0)
    lo = float(np.quantile(f1s, alpha / 2))
    hi = float(np.quantile(f1s, 1 - alpha / 2))
    return float(np.mean(f1s)), lo, hi


def bootstrap_f1_diff_ci(preds_a: np.ndarray, preds_b: np.ndarray, labels_bin: np.ndarray,
                         n_boot: int, alpha: float, rng: np.random.Generator):
    n = len(labels_bin)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        lab = labels_bin[idx]
        diffs[i] = f1_score(lab, preds_b[idx], zero_division=0) - f1_score(lab, preds_a[idx], zero_division=0)
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    return float(np.mean(diffs)), lo, hi


# kiểm định McNemar theo từng cửa sổ
def mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray):
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "stat": 0.0, "p_value": 1.0,
                "test": "mcnemar_continuity_corrected"}
    stat = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
    p = float(chi2.sf(stat, df=1))
    return {"n01": n01, "n10": n10, "stat": float(stat), "p_value": p,
            "test": "mcnemar_continuity_corrected"}


def load_eval(eval_dir: Path):
    if not eval_dir.is_dir():
        preds = np.load(eval_dir.with_name(eval_dir.name + "_preds.npy"))
        labels = np.load(eval_dir.with_name(eval_dir.name + "_labels.npy"))
        values, counts = np.unique(labels, return_counts=True)
        if len(values) != 2:
            raise ValueError(f"{eval_dir.name}: expected two label values, found {values.tolist()}")
        normal_id = int(values[int(np.argmax(counts))])
        return preds, labels, {"normal_id": normal_id}
    preds = np.load(eval_dir / "preds.npy")
    labels = np.load(eval_dir / "labels.npy")
    report = json.loads((eval_dir / "report.json").read_text())
    return preds, labels, report


def normal_id_from_report(report: dict) -> int:
    if "normal_id" in report:
        return int(report["normal_id"])
    src = report.get("source_ckpt")
    if src is None:
        raise ValueError("Cannot determine normal_id: report missing 'normal_id' and 'source_ckpt'")
    ckpt_path = ROOT / src
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Cannot determine normal_id: ckpt {ckpt_path} missing")
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return int(ckpt["label2id"].get("normal", 0))


def main():
    eval_a = ROOT / EVAL_DIR_A
    eval_b = ROOT / EVAL_DIR_B
    print(f"A: {eval_a}")
    print(f"B: {eval_b}")

    preds_a, labels_a, report_a = load_eval(eval_a)
    preds_b, labels_b, report_b = load_eval(eval_b)

    pa_bin, lab_bin = binary_collapse(preds_a, labels_a, normal_id_from_report(report_a))
    pb_bin, lab_bin_b = binary_collapse(preds_b, labels_b, normal_id_from_report(report_b))
    if not np.array_equal(lab_bin, lab_bin_b):
        raise ValueError("Label arrays differ between A and B; not paired evaluation")

    rng = np.random.default_rng(SEED)
    f1_a = f1_score(lab_bin, pa_bin, zero_division=0)
    f1_b = f1_score(lab_bin, pb_bin, zero_division=0)
    p_a = precision_score(lab_bin, pa_bin, zero_division=0)
    p_b = precision_score(lab_bin, pb_bin, zero_division=0)
    r_a = recall_score(lab_bin, pa_bin, zero_division=0)
    r_b = recall_score(lab_bin, pb_bin, zero_division=0)

    mean_a, lo_a, hi_a = bootstrap_f1_ci(pa_bin, lab_bin, N_BOOTSTRAP, ALPHA, rng)
    rng = np.random.default_rng(SEED)
    mean_b, lo_b, hi_b = bootstrap_f1_ci(pb_bin, lab_bin, N_BOOTSTRAP, ALPHA, rng)

    rng = np.random.default_rng(SEED)
    mean_d, lo_d, hi_d = bootstrap_f1_diff_ci(pa_bin, pb_bin, lab_bin, N_BOOTSTRAP, ALPHA, rng)

    mn_pair = mcnemar_test(pa_bin, pb_bin, lab_bin)

    print()
    print(f"  Point F1  A={f1_a:.4f} (P={p_a:.4f} R={r_a:.4f})  B={f1_b:.4f} (P={p_b:.4f} R={r_b:.4f})  delta={f1_b - f1_a:+.4f}")
    print(f"  Bootstrap F1 (B={N_BOOTSTRAP}, {int((1 - ALPHA) * 100)}% CI):")
    print(f"    A: mean={mean_a:.4f}  CI=[{lo_a:.4f}, {hi_a:.4f}]")
    print(f"    B: mean={mean_b:.4f}  CI=[{lo_b:.4f}, {hi_b:.4f}]")
    print(f"    B-A paired difference: mean={mean_d:+.4f}  CI=[{lo_d:.4f}, {hi_d:.4f}]")
    print(f"  McNemar (continuity-corrected): n01={mn_pair['n01']} n10={mn_pair['n10']}  "
          f"chi2={mn_pair['stat']:.4f}  p={mn_pair['p_value']:.4g}")

    out = {
        "eval_a": EVAL_DIR_A, "eval_b": EVAL_DIR_B,
        "n_samples": int(len(lab_bin)),
        "n_anomaly": int(lab_bin.sum()),
        "f1_a": float(f1_a), "f1_b": float(f1_b),
        "precision_a": float(p_a), "precision_b": float(p_b),
        "recall_a": float(r_a), "recall_b": float(r_b),
        "bootstrap": {
            "n_boot": N_BOOTSTRAP, "alpha": ALPHA, "seed": SEED,
            "a": {"mean": mean_a, "ci_lo": lo_a, "ci_hi": hi_a},
            "b": {"mean": mean_b, "ci_lo": lo_b, "ci_hi": hi_b},
            "diff_b_minus_a": {"mean": mean_d, "ci_lo": lo_d, "ci_hi": hi_d, "paired": True},
        },
        "mcnemar": mn_pair,
    }
    out_path = ROOT / "runs" / f"stats_{eval_a.name}_vs_{eval_b.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
