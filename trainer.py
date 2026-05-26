import csv
import math
import re
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from losses import CombinedLoss, rdrop_kl
from metrics import compute_metrics


def llrd_param_groups(model, top_lr, decay, weight_decay):
    layer_ids = set()
    for name, _ in model.named_parameters():
        m = re.search(r"encoder\.layers?\.(\d+)\.", name)
        if m:
            layer_ids.add(int(m.group(1)))
    num_layers = max(layer_ids) + 1 if layer_ids else 0

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r"encoder\.layers?\.(\d+)\.", name)
        if m:
            depth_from_top = num_layers - 1 - int(m.group(1))
        elif name.startswith("classifier") or name.startswith("pool_attn"):
            depth_from_top = 0
        elif "embeddings" in name:
            depth_from_top = num_layers
        elif "encoder.final_norm" in name:
            depth_from_top = 0
        else:
            depth_from_top = num_layers

        layer_lr = top_lr * (decay ** depth_from_top)
        is_no_decay = any(k in name for k in ("bias", "norm", "LayerNorm"))
        wd = 0.0 if is_no_decay else weight_decay

        key = (layer_lr, wd)
        groups.setdefault(key, []).append(p)

    return [{"params": params, "lr": lr, "weight_decay": wd} for (lr, wd), params in groups.items()]


CSV_COLUMNS = [
    "epoch", "time_s", "lr",
    "train/cls", "train/supcon", "train/total",
    "val/binary_f1", "val/binary_precision", "val/binary_recall", "val/binary_accuracy",
    "val/tp", "val/fp", "val/tn", "val/fn",
]


def cosine_warmup(optimizer, warmup, total):
    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, fn)


@torch.no_grad()
def evaluate(model, loader, device, id2label, normal_id):
    model.eval()
    preds, labels_all = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        labels = batch.pop("labels")
        inputs = {k: v.to(device) for k, v in batch.items()}
        out = model(**inputs)
        preds.append(out.logits.argmax(-1).cpu().numpy())
        labels_all.append(labels.numpy())
    return compute_metrics(
        np.concatenate(preds), np.concatenate(labels_all), id2label, normal_id,
    )


def train(
    model, train_loader, val_loader, label2id, output_dir, device, *,
    epochs, lr,
    weight_decay=0.01, warmup_ratio=0.1, grad_accum_steps=1, grad_checkpoint=False,
    alpha_supcon=0.5, supcon_temperature=0.07, label_smoothing=0.1,
    log_every=50, resume=False, extra_meta=None,
    stable_adamw=False, llrd=False, llrd_decay=0.9,
    swa=False,
    r_drop=False, r_drop_alpha=5.0,
):
    swa_last_k = max(1, epochs // 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    id2label = {v: k for k, v in label2id.items()}
    normal_id = label2id.get("normal", 0)

    params = llrd_param_groups(model, lr, llrd_decay, weight_decay) if llrd else model.parameters()
    if stable_adamw:
        from optimi import StableAdamW
        optimizer = StableAdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs // max(1, grad_accum_steps)
    warmup = int(total_steps * warmup_ratio)
    scheduler = cosine_warmup(optimizer, warmup, total_steps)
    loss_fn = CombinedLoss(
        alpha_supcon=alpha_supcon, supcon_temperature=supcon_temperature,
        label_smoothing=label_smoothing,
    )

    if grad_checkpoint and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        print("grad_ckpt=on")

    start_epoch = 0
    results_csv = output_dir / "results.csv"
    state_path = output_dir / "train_state.pt"

    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        print(f"resumed from epoch {start_epoch}")
    else:
        with results_csv.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    print(f"steps={total_steps} warmup={warmup} epochs={epochs} start={start_epoch}")
    meta = {**(extra_meta or {}),
            "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio, "grad_accum_steps": grad_accum_steps,
            "grad_checkpoint": grad_checkpoint, "alpha_supcon": alpha_supcon,
            "supcon_temperature": supcon_temperature,
            "label_smoothing": label_smoothing,
            "stable_adamw": stable_adamw,
            "llrd": llrd,
            "llrd_decay": llrd_decay if llrd else None,
            "swa": swa,
            "swa_last_k": swa_last_k if swa else None,
            "swa_policy": "last_50%_of_epochs" if swa else None,
            "r_drop": r_drop,
            "r_drop_alpha": r_drop_alpha if r_drop else None}

    swa_state = None
    swa_count = 0

    metrics = None
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running = {"cls": 0.0, "supcon": 0.0, "total": 0.0}
        n = 0

        pbar = tqdm(train_loader, desc=f"ep {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            if r_drop:
                out_a = model(**inputs)
                out_b = model(**inputs)
                loss_a, info_a = loss_fn(out_a.logits, out_a.embeddings, labels)
                loss_b, info_b = loss_fn(out_b.logits, out_b.embeddings, labels)
                kl = rdrop_kl(out_a.logits, out_b.logits)
                loss = 0.5 * (loss_a + loss_b) + r_drop_alpha * kl
                info = {k: 0.5 * (info_a[k] + info_b[k]) for k in info_a}
                info["total"] = loss.item()
            else:
                out = model(**inputs)
                loss, info = loss_fn(out.logits, out.embeddings, labels)
            (loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            for k, v in info.items():
                running[k] += v
            n += 1

            if (step + 1) % log_every == 0:
                pbar.set_postfix(lr=scheduler.get_last_lr()[0],
                                 **{k: f"{v/n:.3f}" for k, v in running.items()})

        avg = {k: v / max(1, n) for k, v in running.items()}
        elapsed = time.time() - t0
        print(f"ep {epoch+1}/{epochs} {elapsed:.0f}s cls={avg['cls']:.3f} supcon={avg['supcon']:.3f}")

        row = {
            "epoch": epoch + 1,
            "time_s": round(elapsed, 1),
            "lr": scheduler.get_last_lr()[0],
            "train/cls": round(avg["cls"], 6),
            "train/supcon": round(avg["supcon"], 6),
            "train/total": round(avg["total"], 6),
        }

        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, id2label, normal_id)
            print(f"  binary | P={metrics['binary_precision']:.4f} R={metrics['binary_recall']:.4f} "
                  f"F1={metrics['binary_f1']:.4f} acc={metrics['binary_accuracy']:.4f}  "
                  f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
            row.update({
                "val/binary_f1": round(metrics["binary_f1"], 6),
                "val/binary_precision": round(metrics["binary_precision"], 6),
                "val/binary_recall": round(metrics["binary_recall"], 6),
                "val/binary_accuracy": round(metrics["binary_accuracy"], 6),
                "val/tp": metrics["tp"],
                "val/fp": metrics["fp"],
                "val/tn": metrics["tn"],
                "val/fn": metrics["fn"],
            })

        if swa and epoch >= epochs - swa_last_k:
            cur = {k: v.detach().to("cpu", dtype=torch.float32) for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state = cur
                swa_count = 1
            else:
                for k in swa_state:
                    swa_state[k].mul_(swa_count / (swa_count + 1)).add_(cur[k], alpha=1.0 / (swa_count + 1))
                swa_count += 1

        save_checkpoint(model, label2id, meta, output_dir / "checkpoint.pt", metrics)

        with results_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
        }, state_path)

    if swa and swa_state is not None:
        orig = model.state_dict()
        swa_typed = {k: v.to(dtype=orig[k].dtype, device=orig[k].device) for k, v in swa_state.items()}
        model.load_state_dict(swa_typed)
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, id2label, normal_id)
            print(f"  SWA (avg last {swa_count} epochs) | "
                  f"P={metrics['binary_precision']:.4f} R={metrics['binary_recall']:.4f} F1={metrics['binary_f1']:.4f}")
        save_checkpoint(model, label2id, meta, output_dir / "checkpoint.pt", metrics)

    return {"final_binary_f1": metrics["binary_f1"] if metrics else None}


def save_checkpoint(model, label2id, meta, out_path, metrics=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "label2id": label2id,
        "config": meta,
    }
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, out_path)
