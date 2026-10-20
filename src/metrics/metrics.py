"""Evaluation metrics: DSC, patient-level AUC and the combined score.

Following Section 4.3 of the paper:

* DSC is computed only on slices with a non-empty ground-truth tumour mask.
* AUC is computed at the patient level, by averaging the slice-level MIBC
  probabilities of each patient into a single score.
* The combined score used for model selection is

      S_comb = 0.55 * DSC + 0.45 * AUC
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = ["DSC_WEIGHT", "AUC_WEIGHT", "combined_score", "dice_score", "evaluate"]

DSC_WEIGHT = 0.55
AUC_WEIGHT = 0.45


def combined_score(dsc: float, auc: float) -> float:
    """Return ``0.55 * dsc + 0.45 * auc`` (Eq. 9)."""
    return DSC_WEIGHT * dsc + AUC_WEIGHT * auc


@torch.no_grad()
def dice_score(
    pred_logit: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5,
) -> float:
    """Mean Dice over a batch of binarised predictions."""
    pred = (torch.sigmoid(pred_logit) > threshold).float()
    pred = pred.reshape(pred.size(0), -1)
    target = target.float().reshape(target.size(0), -1)
    inter = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1)
    valid = union > 0
    d = torch.ones_like(union)
    d[valid] = (2.0 * inter[valid] + smooth) / (union[valid] + smooth)
    return d.clamp(0, 1).mean().item()


@torch.no_grad()
def evaluate(model, loader, device, amp: bool = False) -> dict:
    """Run a full evaluation pass.

    Args:
        model: a model returning ``{"seg": ..., "cls": ...}``.
        loader: dataloader yielding the ``FedBCaDataset`` batch dict.
        device: evaluation device.
        amp: run the forward pass under autocast. Keep this equal to the
            ``use_amp`` setting of the training configuration. The Swin Hybrid
            model is trained and evaluated in FP32.

    Returns:
        Dict with ``dsc`` (all annotated slices), ``tumour_dsc`` (slices
        flagged as tumour-bearing), patient-level ``auc`` and classification
        statistics, and ``combined_score``.
    """
    model.eval()

    all_dsc: list[float] = []
    tumour_dsc: list[float] = []
    pid_probs: dict[str, list[float]] = defaultdict(list)
    pid_labels: dict[str, int] = {}

    autocast = torch.autocast(device_type=device.type, enabled=amp)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"]
        has_tumour = batch["has_tumor"].cpu().numpy()
        pids = batch["patient_id"]

        with autocast:
            outputs = model(images)

        seg = torch.nan_to_num(outputs["seg"], nan=0.0, posinf=0.0, neginf=0.0)
        cls = torch.nan_to_num(outputs["cls"], nan=0.0, posinf=0.0, neginf=0.0)

        for i in range(images.size(0)):
            if masks[i].sum() <= 0:
                continue
            d = dice_score(seg[i : i + 1], masks[i : i + 1])
            all_dsc.append(d)
            if has_tumour[i]:
                tumour_dsc.append(d)

        probs = torch.softmax(cls, dim=1)[:, 1].detach().cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.5)

        for i, pid in enumerate(pids):
            pid_probs[pid].append(float(probs[i]))
            pid_labels[pid] = int(labels[i].item())

    # Patient-level aggregation: unweighted mean of the slice probabilities.
    pt_ids = sorted(pid_probs)
    pt_prob = np.array([float(np.mean(pid_probs[p])) for p in pt_ids])
    pt_label = np.array([pid_labels[p] for p in pt_ids])
    pt_pred = (pt_prob >= 0.5).astype(np.int32)

    if len(pt_ids) > 0 and len(np.unique(pt_label)) >= 2:
        auc = float(roc_auc_score(pt_label, pt_prob))
    else:
        # A single-class validation fold carries no ranking information.
        auc = 0.5

    mean_dsc = float(np.mean(all_dsc)) if all_dsc else 0.0
    mean_tumour_dsc = float(np.mean(tumour_dsc)) if tumour_dsc else 0.0

    if len(pt_ids) > 0:
        acc = float(accuracy_score(pt_label, pt_pred))
        bal_acc = float(balanced_accuracy_score(pt_label, pt_pred))
        prec = float(precision_score(pt_label, pt_pred, zero_division=0))
        rec = float(recall_score(pt_label, pt_pred, zero_division=0))
        f1 = float(f1_score(pt_label, pt_pred, zero_division=0))
    else:
        acc = bal_acc = prec = rec = f1 = 0.0

    return {
        "dsc": round(mean_dsc, 4),
        "tumour_dsc": round(mean_tumour_dsc, 4),
        "auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "combined_score": round(combined_score(mean_tumour_dsc, auc), 4),
        "n_patients": len(pt_ids),
        "n_tumour_slices": len(tumour_dsc),
    }
