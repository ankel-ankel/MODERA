import os
import platform
import random
import socket
import subprocess
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parent

from data_loader import LogCollator, LogDataset, balanced_sampler
from model import ModernBertClassifier
from trainer import train



# đổi dataset và đường dẫn ở đây
DATASET          = "BGL"
TRAIN_CSV        = "data/BGL/train.csv"
VAL_CSV          = "data/BGL/val.csv"
MODEL_PATH       = "models/ModernBERT-large"

# siêu tham số huấn luyện
EPOCHS           = 4
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 4
EVAL_BATCH_SIZE  = 2
LR               = 5e-5
MAX_TOKEN_LEN    = 2048
SEED             = 0

# tham số của recipe
ALPHA_SUPCON     = 0.0
LABEL_SMOOTHING  = 0.1
POOLING_TYPE     = "attention"
LLRD_DECAY       = 0.9
TARGET_RATIO     = 0.4

# bật tắt từng thành phần để chạy ablation
BALANCED_SAMPLER = True
STABLE_ADAMW     = True
LLRD             = True
SWA              = True
LOSS_TYPE        = "ce"
FOCAL_GAMMA      = 2.0
FOCAL_ALPHA      = 0.25
LA_TAU           = 1.0

# head phụ, để False cho cấu hình chính
USE_MLP_HEAD     = False
HEAD_HIDDEN      = 512
HEAD_DROPOUT     = 0.2
USE_PROJ_HEAD    = False
PROJ_HIDDEN      = 256
PROJ_OUT         = 128
GRAD_CHECKPOINT  = False

RESUME_DIR       = None

# env var để chạy nhiều seed hoặc nhiều cấu hình, không phải sửa file
def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None else default


SEED             = int(os.environ.get("MODERA_SEED", SEED))
EPOCHS           = int(os.environ.get("MODERA_EPOCHS", EPOCHS))
MAX_TOKEN_LEN    = int(os.environ.get("MODERA_MAXLEN", MAX_TOKEN_LEN))
BATCH_SIZE       = int(os.environ.get("MODERA_BATCH", BATCH_SIZE))
GRAD_ACCUM_STEPS = int(os.environ.get("MODERA_ACCUM", GRAD_ACCUM_STEPS))
EVAL_BATCH_SIZE  = int(os.environ.get("MODERA_EVAL_BATCH", EVAL_BATCH_SIZE))
NUM_WORKERS      = int(os.environ.get("MODERA_NUM_WORKERS", "4"))
DATASET          = os.environ.get("MODERA_DATASET", DATASET)
if "MODERA_DATASET" in os.environ:
    TRAIN_CSV = f"data/{DATASET}/train.csv"
    VAL_CSV   = f"data/{DATASET}/val.csv"
MODEL_PATH       = os.environ.get("MODERA_MODEL_PATH", MODEL_PATH)
LR               = _env_float("MODERA_LR", LR)

POOLING_TYPE     = os.environ.get("MODERA_POOLING", POOLING_TYPE)
ALPHA_SUPCON     = _env_float("MODERA_ALPHA_SUPCON", ALPHA_SUPCON)
LABEL_SMOOTHING  = _env_float("MODERA_LABEL_SMOOTHING", LABEL_SMOOTHING)
LLRD_DECAY       = _env_float("MODERA_LLRD_DECAY", LLRD_DECAY)
TARGET_RATIO     = _env_float("MODERA_TARGET_RATIO", TARGET_RATIO)
BALANCED_SAMPLER = _env_bool("MODERA_BALANCED_SAMPLER", BALANCED_SAMPLER)
STABLE_ADAMW     = _env_bool("MODERA_STABLE_ADAMW", STABLE_ADAMW)
LLRD             = _env_bool("MODERA_LLRD", LLRD)
SWA              = _env_bool("MODERA_SWA", SWA)
LOSS_TYPE        = os.environ.get("MODERA_LOSS_TYPE", LOSS_TYPE)
FOCAL_GAMMA      = _env_float("MODERA_FOCAL_GAMMA", FOCAL_GAMMA)
FOCAL_ALPHA      = _env_float("MODERA_FOCAL_ALPHA", FOCAL_ALPHA)
LA_TAU           = _env_float("MODERA_LA_TAU", LA_TAU)
USE_MLP_HEAD     = _env_bool("MODERA_USE_MLP_HEAD", USE_MLP_HEAD)
USE_PROJ_HEAD    = _env_bool("MODERA_USE_PROJ_HEAD", USE_PROJ_HEAD)
GRAD_CHECKPOINT  = _env_bool("MODERA_GRAD_CHECKPOINT", GRAD_CHECKPOINT)
RUN_TAG          = os.environ.get("MODERA_RUN_TAG", "")
OPTIM_EPS_ENV    = os.environ.get("MODERA_OPTIM_EPS")
OPTIM_EPS        = float(OPTIM_EPS_ENV) if OPTIM_EPS_ENV else None
OPTIM_VARIANT    = os.environ.get("MODERA_OPTIM", "auto")
OPTIM_KAHAN_ENV  = os.environ.get("MODERA_KAHAN")
OPTIM_KAHAN      = None if OPTIM_KAHAN_ENV is None else OPTIM_KAHAN_ENV.strip() == "1"
OPTIM_BETA2_ENV  = os.environ.get("MODERA_BETA2")
OPTIM_BETA2      = float(OPTIM_BETA2_ENV) if OPTIM_BETA2_ENV else None
MASTER_FP32      = os.environ.get("MODERA_MASTER_FP32", "0") == "1"



def next_run_dir(base, prefix="checkpoint"):
    base.mkdir(parents=True, exist_ok=True)
    nums = [int(d.name[len(prefix):]) for d in base.iterdir()
            if d.is_dir() and d.name.startswith(prefix) and d.name[len(prefix):].isdigit()]
    out = base / f"{prefix}{(max(nums) + 1) if nums else 1}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def init_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "no_git"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "no_git"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    init_seeds(SEED)

    train_csv = ROOT / TRAIN_CSV
    val_csv = ROOT / VAL_CSV
    model_path = ROOT / MODEL_PATH

    if not val_csv.exists():
        raise FileNotFoundError(
            f"{val_csv} not found. Run `python build_dataset.py` to regenerate "
            f"train/val/test split (70/10/20 chronological)."
        )

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

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    train_ds = LogDataset(str(train_csv), tokenizer=tokenizer, max_token_len=MAX_TOKEN_LEN)
    val_ds = LogDataset(str(val_csv), tokenizer=tokenizer, max_token_len=MAX_TOKEN_LEN)

    all_labels = sorted(set(train_ds.labels) | set(val_ds.labels))
    label2id = {lbl: i for i, lbl in enumerate(all_labels)}
    if "normal" not in label2id:
        label2id = {"normal": 0, **{k: v + 1 for k, v in label2id.items()}}
    print(f"  K={len(label2id)}")

    assert len(label2id) == 2, f"binary head expected K=2, got K={len(label2id)} labels={list(label2id)}"
    assert "normal" in label2id and "anomaly" in label2id, f"expected normal+anomaly labels, got {list(label2id)}"
    assert len(train_ds) > 0 and len(val_ds) > 0, f"empty dataset train={len(train_ds)} val={len(val_ds)}"
    assert MAX_TOKEN_LEN > 0 and MAX_TOKEN_LEN <= 8192, f"MAX_TOKEN_LEN={MAX_TOKEN_LEN} out of [1, 8192]"
    assert BATCH_SIZE * GRAD_ACCUM_STEPS > 0, "effective batch size must be > 0"

    collator = LogCollator(tokenizer=tokenizer, label2id=label2id)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(SEED)
    if BALANCED_SAMPLER:
        loader_kw = {"sampler": balanced_sampler(
            train_ds.labels, target_ratio=TARGET_RATIO, generator=loader_gen,
        )}
    else:
        loader_kw = {"shuffle": True, "generator": loader_gen}

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, collate_fn=collator,
        num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0, pin_memory=True,
        drop_last=True, worker_init_fn=_seed_worker, **loader_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator,
        num_workers=NUM_WORKERS, persistent_workers=False, pin_memory=True, drop_last=False,
        worker_init_fn=_seed_worker,
    )

    model = ModernBertClassifier(
        str(model_path), num_labels=len(label2id), pooling=POOLING_TYPE,
        use_mlp_head=USE_MLP_HEAD, head_hidden=HEAD_HIDDEN, head_dropout=HEAD_DROPOUT,
        use_proj_head=USE_PROJ_HEAD, proj_hidden=PROJ_HIDDEN, proj_out=PROJ_OUT,
        grad_checkpoint=GRAD_CHECKPOINT,
        dtype=torch.float32 if MASTER_FP32 else torch.bfloat16,
    ).to(device)

    train_labels_int = np.array([label2id[lbl] for lbl in train_ds.labels])
    counts = np.bincount(train_labels_int, minlength=len(label2id)).astype(np.float64)
    base_probs = (counts / counts.sum()).tolist()

    env_meta = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "git_sha": _git_sha(),
        "attn_implementation": model.attn_implementation,
        "run_tag": RUN_TAG,
    }

    summary = train(
        model, train_loader, val_loader, label2id, output_dir, device,
        epochs=EPOCHS, lr=LR, alpha_supcon=ALPHA_SUPCON,
        label_smoothing=LABEL_SMOOTHING, resume=resume,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        stable_adamw=STABLE_ADAMW,
        llrd=LLRD, llrd_decay=LLRD_DECAY,
        swa=SWA, optim_eps=OPTIM_EPS, optim_variant=OPTIM_VARIANT,
        optim_kahan=OPTIM_KAHAN, optim_beta2=OPTIM_BETA2,
        autocast_dtype=torch.bfloat16 if MASTER_FP32 else None,
        loss_type=LOSS_TYPE, focal_gamma=FOCAL_GAMMA, focal_alpha=FOCAL_ALPHA,
        base_probs=base_probs, la_tau=LA_TAU,
        extra_meta={
            "dataset": DATASET, "batch_size": BATCH_SIZE, "max_token_len": MAX_TOKEN_LEN,
            "model_path": MODEL_PATH, "balanced_sampler": BALANCED_SAMPLER,
            "target_ratio": TARGET_RATIO if BALANCED_SAMPLER else None, "seed": SEED,
            "pooling": POOLING_TYPE,
            "use_mlp_head": USE_MLP_HEAD, "head_hidden": HEAD_HIDDEN, "head_dropout": HEAD_DROPOUT,
            "use_proj_head": USE_PROJ_HEAD, "proj_hidden": PROJ_HIDDEN, "proj_out": PROJ_OUT,
            "grad_checkpoint": GRAD_CHECKPOINT,
            "loss_type": LOSS_TYPE, "focal_gamma": FOCAL_GAMMA, "focal_alpha": FOCAL_ALPHA, "la_tau": LA_TAU,
            "base_probs": base_probs,
            **env_meta,
        },
    )
    print(f"DONE best_val_f1={summary['best_val_binary_f1']:.4f}@ep{summary['best_epoch']} dir={output_dir}")


if __name__ == "__main__":
    main()
