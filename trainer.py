import json
import math
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from losses import CombinedLoss
from metrics import compute_classification_metrics


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
    return compute_classification_metrics(
        np.concatenate(preds), np.concatenate(labels_all), id2label, normal_id,
    )


def train(
    model, train_loader, val_loader, label2id, output_dir, device, *,
    epochs, lr, weight_decay, warmup_ratio, grad_accum_steps, grad_checkpoint,
    alpha_supcon, supcon_temperature, use_focal, focal_gamma, label_smoothing,
    class_weights, log_every, save_every, patience=None, resume=False,
    extra_meta=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    id2label = {v: k for k, v in label2id.items()}
    normal_id = label2id.get("normal", 0)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs // max(1, grad_accum_steps)
    warmup = int(total_steps * warmup_ratio)
    scheduler = cosine_warmup(optimizer, warmup, total_steps)
    loss_fn = CombinedLoss(
        alpha_supcon=alpha_supcon, supcon_temperature=supcon_temperature,
        class_weights=class_weights, use_focal=use_focal,
        focal_gamma=focal_gamma, label_smoothing=label_smoothing,
    )

    if grad_checkpoint and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
        print("grad_ckpt=on")

    history = []
    best_f1 = -1.0
    start_epoch = 0
    patience_counter = 0

    state_path = output_dir / "train_state.pt"
    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_f1 = state["best_f1"]
        start_epoch = state["epoch"]
        patience_counter = state.get("patience_counter", 0)
        print(f"resumed from epoch {start_epoch} (best_f1={best_f1:.4f})")

    print(f"steps={total_steps} warmup={warmup} epochs={epochs} start={start_epoch}")
    meta = {**(extra_meta or {}),
            "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio, "grad_accum_steps": grad_accum_steps,
            "grad_checkpoint": grad_checkpoint, "alpha_supcon": alpha_supcon,
            "supcon_temperature": supcon_temperature, "use_focal": use_focal,
            "focal_gamma": focal_gamma, "label_smoothing": label_smoothing,
            "patience": patience}

    early_stopped = False
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running = {"cls": 0.0, "supcon": 0.0, "total": 0.0}
        n = 0

        pbar = tqdm(train_loader, desc=f"ep {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
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
        history.append({"epoch": epoch + 1, "elapsed_s": elapsed, **avg})

        improved = False
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, id2label, normal_id)
            print(f"  multi  | P={metrics['precision_macro']:.4f} R={metrics['recall_macro']:.4f} "
                  f"F1={metrics['f1_macro']:.4f} acc={metrics['accuracy']:.4f}")
            print(f"  binary | P={metrics['binary_precision']:.4f} R={metrics['binary_recall']:.4f} "
                  f"F1={metrics['binary_f1']:.4f} acc={metrics['binary_accuracy']:.4f}  "
                  f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
            history[-1]["val"] = {k: metrics[k] for k in (
                "precision_macro", "recall_macro", "f1_macro", "accuracy",
                "precision_weighted", "recall_weighted", "f1_weighted",
                "binary_precision", "binary_recall", "binary_f1", "binary_accuracy",
                "tp", "fp", "tn", "fn",
            )}
            if metrics["f1_macro"] > best_f1:
                best_f1 = metrics["f1_macro"]
                improved = True
                save_checkpoint(model, label2id, meta, output_dir / "best", metrics)
                print(f"  [best] f1m={best_f1:.4f}")

        patience_counter = 0 if improved else patience_counter + 1

        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, label2id, meta, output_dir / f"epoch{epoch+1}")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "best_f1": best_f1,
            "history": history,
            "patience_counter": patience_counter,
        }, state_path)

        if patience is not None and patience_counter >= patience:
            print(f"early stop: no improvement for {patience} epoch(s)")
            early_stopped = True
            break

    save_checkpoint(model, label2id, meta, output_dir / "final")
    (output_dir / "log_history.json").write_text(json.dumps(history, indent=2))
    return {"history": history, "best_f1_macro": best_f1, "early_stopped": early_stopped}


def save_checkpoint(model, label2id, meta, out, metrics=None):
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "label2id.json").write_text(json.dumps(label2id, indent=2))
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    if metrics is not None:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
