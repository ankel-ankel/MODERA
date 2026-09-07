import csv
import json
import os
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parent

EVAL_PREFIX  = "eval"
# gom các run cùng cấu hình
GROUP_KEYS   = ["dataset", "pooling", "batch_size", "grad_accum_steps", "lr", "max_token_len",
                "alpha_supcon", "label_smoothing", "balanced_sampler", "target_ratio",
                "stable_adamw", "llrd", "llrd_decay", "swa", "model_path",
                "loss_type", "focal_gamma", "focal_alpha", "la_tau",
                "use_mlp_head", "head_hidden", "head_dropout",
                "use_proj_head", "proj_hidden", "proj_out",
                "grad_checkpoint", "epochs", "weight_decay", "warmup_ratio"]
METRICS_KEYS = ["binary_f1", "binary_precision", "binary_recall", "binary_accuracy",
                "tp", "fp", "tn", "fn"]

EVAL_PREFIX = os.environ.get("MODERA_EVAL_PREFIX", EVAL_PREFIX)


def load_ckpt_config(ckpt_rel: str) -> dict:
    p = ROOT / ckpt_rel
    if not p.exists():
        return {}
    import torch
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    return dict(ckpt.get("config", {}))


# quét thư mục runs/
def collect_runs():
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        return []
    rows = []
    pat = re.compile(rf"^{re.escape(EVAL_PREFIX)}(\d+)$")
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or not pat.match(d.name):
            continue
        report_path = d / "report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        src = report.get("source_ckpt")
        cfg = load_ckpt_config(src) if src else {}
        seed = cfg.get("seed", -1)
        threshold_tuned = bool(report.get("threshold", {}).get("tuned", False))
        eval_dataset = report.get("eval_dataset", cfg.get("dataset"))
        if eval_dataset != cfg.get("dataset"):
            print(f"  SKIP {d.name}: eval dataset {eval_dataset} != ckpt dataset {cfg.get('dataset')} (cross-dataset eval)")
            continue
        bool_defaults = {"grad_checkpoint": False, "use_mlp_head": False, "use_proj_head": False}
        def norm(k):
            v = cfg.get(k)
            if v is None and k in bool_defaults:
                return bool_defaults[k]
            return v
        group = tuple((k, norm(k)) for k in GROUP_KEYS) + (("threshold_tuned", threshold_tuned),)
        row = {
            "eval_dir": d.name,
            "source_ckpt": src,
            "seed": seed,
            "threshold_tuned": threshold_tuned,
            "group_key": group,
            **{k: report.get(k) for k in METRICS_KEYS},
        }
        rows.append(row)

    deduped = {}
    for r in rows:
        key = (r["source_ckpt"], r["threshold_tuned"], r["group_key"])
        deduped[key] = r
    return list(deduped.values())


# tính trung bình, độ lệch chuẩn, trung vị
def aggregate(rows):
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault(r["group_key"], []).append(r)

    out = []
    for gk, members in groups.items():
        cfg = dict(gk)
        seeds = sorted({m["seed"] for m in members})
        n = len(members)
        stats = {}
        for mk in METRICS_KEYS:
            vals = [m[mk] for m in members if isinstance(m[mk], (int, float))]
            if not vals:
                stats[f"{mk}_mean"] = None
                stats[f"{mk}_std"] = None
                continue
            stats[f"{mk}_mean"] = mean(vals)
            stats[f"{mk}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        out.append({
            "n_seeds": n,
            "seeds": seeds,
            **cfg,
            **stats,
            "member_eval_dirs": [m["eval_dir"] for m in members],
        })
    return out


def main():
    rows = collect_runs()
    print(f"collected {len(rows)} eval runs")
    if not rows:
        return

    agg = aggregate(rows)
    print(f"grouped into {len(agg)} configs")

    out_dir = ROOT / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "aggregate.json"
    csv_path = out_dir / "aggregate.csv"
    json_path.write_text(json.dumps(agg, indent=2, default=list))

    if agg:
        fields = ["n_seeds", "dataset", "pooling", "max_token_len",
                  "stable_adamw", "llrd", "swa", "alpha_supcon", "label_smoothing",
                  "balanced_sampler", "lr",
                  "binary_f1_mean", "binary_f1_std",
                  "binary_precision_mean", "binary_precision_std",
                  "binary_recall_mean", "binary_recall_std",
                  "seeds", "member_eval_dirs"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in sorted(agg, key=lambda r: -(r.get("binary_f1_mean") or 0)):
                writer.writerow({**row, "seeds": ";".join(str(s) for s in row["seeds"]),
                                 "member_eval_dirs": ";".join(row["member_eval_dirs"])})

    print(f"saved -> {json_path}")
    print(f"saved -> {csv_path}")
    print()
    for row in sorted(agg, key=lambda r: -(r.get("binary_f1_mean") or 0))[:10]:
        f1m = row.get("binary_f1_mean")
        f1s = row.get("binary_f1_std")
        print(f"  n={row['n_seeds']}  F1={f1m:.4f}±{f1s:.4f}  "
              f"pool={row.get('pooling')} maxlen={row.get('max_token_len')} "
              f"stable={row.get('stable_adamw')} llrd={row.get('llrd')} "
              f"alpha={row.get('alpha_supcon')} ds={row.get('dataset')}")


if __name__ == "__main__":
    main()
