"""Centralised training engine for the Swin Hybrid model.

Implements the protocol of Section 4.2: AdamW with a warmup-cosine schedule,
per-branch learning-rate multipliers for the ResNet-34 and Swin-Tiny
encoders, gradient accumulation to the effective batch size, gradient
clipping, a staged ramp-in of the classification loss, and checkpoint
selection on the highest validation combined score.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import torch

from ..metrics.metrics import evaluate

logger = logging.getLogger(__name__)

__all__ = ["warmup_cosine", "build_optimizer", "classification_weight", "train_model"]

#: Parameter-name fragments identifying the Swin branch.
_SWIN_KEYS = ("swin",)
#: Parameter-name fragments identifying the pretrained CNN branch.
_CNN_KEYS = ("encoder.cnn",)


def warmup_cosine(
    step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.05
) -> float:
    """Learning-rate multiplier: linear warmup, then cosine decay."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_optimizer(model, config: dict) -> torch.optim.AdamW:
    """Build AdamW with separate learning rates per encoder branch.

    The decoder, task heads and ResNet-34 branch train at the base learning
    rate (``enc_lr_ratio`` = 1.0, as in the recorded runs); the Swin-Tiny
    branch is scaled by ``swin_lr_ratio`` (0.30). The paper text mentions a
    0.15 multiplier for ResNet-34, which the recorded experiments never
    applied; set ``enc_lr_ratio: 0.15`` to use it (see docs/PAPER_CODE_AUDIT.md).
    """
    base_lr = float(config["lr"])
    weight_decay = float(config.get("weight_decay", 1e-4))
    cnn_lr = base_lr * float(config.get("enc_lr_ratio", 1.0))
    swin_lr = base_lr * float(config.get("swin_lr_ratio", 0.30))

    swin_params, cnn_params, head_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in _SWIN_KEYS):
            swin_params.append(param)
        elif any(k in name for k in _CNN_KEYS):
            cnn_params.append(param)
        else:
            head_params.append(param)

    groups = [{"params": head_params, "lr": base_lr, "name": "head"}]
    if cnn_params:
        groups.append({"params": cnn_params, "lr": cnn_lr, "name": "cnn"})
    if swin_params:
        groups.append({"params": swin_params, "lr": swin_lr, "name": "swin"})

    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)


def classification_weight(epoch: int, rampup_epochs: int, target: float) -> float:
    """Staged ramp of the classification loss weight.

    The weight is zero for the first ``rampup_epochs`` epochs, rises linearly
    over the next ``rampup_epochs``, and is held at ``target`` afterwards. With
    the configured value of 10 the classification term reaches its full weight
    at cumulative epoch 20, as described in Section 3.3.
    """
    if epoch < rampup_epochs:
        return 0.0
    if epoch < 2 * rampup_epochs:
        return target * (epoch - rampup_epochs) / float(rampup_epochs)
    return target


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    config: dict,
    exp_name: str,
    save_dir: str | Path,
    device: torch.device | None = None,
) -> dict:
    """Train the model and return the metrics of the selected checkpoint.

    The checkpoint is selected on training loss while the classification loss
    is still ramping in, and on the validation combined score afterwards, so
    that early epochs (whose AUC is uninformative) cannot win selection.

    Returns:
        Dict with the best DSC, AUC and combined score, the checkpoint
        directory and the training history.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(save_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    use_amp = bool(config.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    optimizer = build_optimizer(model, config)

    epochs = int(config.get("epochs", 250))
    accum_steps = int(config.get("accum_steps", 4))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, epochs * steps_per_epoch // accum_steps)
    warmup_steps = (
        int(config.get("warmup_epochs", 10)) * steps_per_epoch // accum_steps
    )
    min_lr_ratio = float(config.get("min_lr_ratio", 0.05))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine(
            step, total_steps, warmup_steps, min_lr_ratio
        ),
    )

    patience = int(config.get("patience", 80))
    eval_every = int(config.get("eval_every", 5))
    grad_clip = float(config.get("grad_clip", 1.0))
    cls_rampup = int(config.get("cls_rampup_epochs", 10))
    target_cls_weight = float(config.get("cls_weight", 0.20))
    selection_starts = cls_rampup * 2 + 5

    best_combined = -1.0
    best_loss = float("inf")
    patience_count = 0
    history: dict = defaultdict(list)

    logger.info(
        "[%s] epochs=%d device=%s amp=%s effective_batch=%d",
        exp_name,
        epochs,
        device,
        use_amp,
        int(config.get("batch_size", 6)) * accum_steps,
    )

    for epoch in range(epochs):
        criterion.cls_weight = classification_weight(
            epoch, cls_rampup, target_cls_weight
        )

        model.train()
        epoch_loss = epoch_seg = epoch_cls = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            cls_mask = batch["use_cls_loss"].float().to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss_dict = criterion(outputs, masks, labels, cls_mask=cls_mask)
                loss = loss_dict["loss"] / accum_steps

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            is_step = (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader)
            if is_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss_dict["loss"].item()
            epoch_seg += loss_dict["seg_loss"].item()
            epoch_cls += loss_dict["cls_loss"].item()

        n_steps = max(1, len(train_loader))
        avg_loss = epoch_loss / n_steps

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)
        history["seg_loss"].append(epoch_seg / n_steps)
        history["cls_loss"].append(epoch_cls / n_steps)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if (epoch + 1) % eval_every != 0 and epoch != epochs - 1:
            continue

        metrics = evaluate(model, val_loader, device, amp=use_amp)
        combined = metrics["combined_score"]

        history["val_dsc"].append(metrics["tumour_dsc"])
        history["val_auc"].append(metrics["auc"])
        history["val_combined"].append(combined)

        if epoch < selection_starts:
            improved = avg_loss < best_loss
            if improved:
                best_loss = avg_loss
        else:
            improved = combined > best_combined
            if improved:
                best_combined = combined

        logger.info(
            "[%s] epoch %d/%d loss=%.4f DSC=%.4f AUC=%.4f combined=%.4f%s",
            exp_name,
            epoch + 1,
            epochs,
            avg_loss,
            metrics["tumour_dsc"],
            metrics["auc"],
            combined,
            " *" if improved else "",
        )

        if improved:
            torch.save(
                {
                    "epoch": epoch,
                    "state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": metrics,
                    "config": config,
                },
                save_dir / "checkpoint_best.pt",
            )
            patience_count = 0
        else:
            patience_count += eval_every
            if patience_count >= patience:
                logger.info("[%s] early stop at epoch %d", exp_name, epoch + 1)
                break

    with (save_dir / "history.json").open("w") as fh:
        json.dump(dict(history), fh, indent=2)

    checkpoint_path = save_dir / "checkpoint_best.pt"
    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"No checkpoint was written for {exp_name}; training produced no "
            "evaluated epoch."
        )

    best = torch.load(checkpoint_path, map_location="cpu")["metrics"]
    logger.info(
        "[%s] done. DSC=%.4f AUC=%.4f combined=%.4f",
        exp_name,
        best["tumour_dsc"],
        best["auc"],
        best["combined_score"],
    )

    return {
        "best_dsc": best["tumour_dsc"],
        "best_auc": best["auc"],
        "combined": best["combined_score"],
        "checkpoint": str(checkpoint_path),
        "history": dict(history),
    }
