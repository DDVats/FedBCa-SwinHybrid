"""Multi-task objective for the Swin Hybrid model.

The joint loss follows Eq. (7) of the paper::

    L = seg_weight * L_seg + cls_weight * L_cls          (0.80 / 0.20)

``L_seg`` combines Dice, binary cross-entropy, Focal Tversky and a boundary
term, plus auxiliary deep-supervision terms at two decoder depths.
``L_cls`` is a focal cross-entropy evaluated only on the slices selected by
``cls_mask``.

Hyperparameters not fixed by the paper (Tversky alpha/beta/gamma, the
region/boundary weights and the deep-supervision weights) carry the values
used in the experiments; see ``docs/PAPER_CODE_AUDIT.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DiceLoss",
    "DiceCELoss",
    "FocalTverskyLoss",
    "BoundaryLoss",
    "SegmentationLoss",
    "DeepSupervisionLoss",
    "StableFocalLoss",
    "JointLoss",
]


class DiceLoss(nn.Module):
    """Soft Dice loss over sigmoid probabilities."""

    def __init__(self, smooth: float = 1e-5, sigmoid: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.float().contiguous().view(target.size(0), -1)
        inter = (pred * target).sum(1)
        union = pred.sum(1) + target.sum(1)
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return (1.0 - dice.clamp(0, 1)).mean()


class DiceCELoss(nn.Module):
    """Weighted sum of the Dice loss and binary cross-entropy."""

    def __init__(self, lambda_dice: float = 0.70, lambda_ce: float = 0.30) -> None:
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.dice = DiceLoss(sigmoid=True)
        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lambda_dice * self.dice(pred, target) + self.lambda_ce * self.ce(
            pred, target.float()
        )


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss. ``beta > alpha`` penalises false negatives more."""

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.60,
        gamma: float = 0.75,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred).contiguous().view(pred.size(0), -1)
        target = target.float().contiguous().view(target.size(0), -1)

        tp = (pred * target).sum(1)
        fp = (pred * (1 - target)).sum(1)
        fn = ((1 - pred) * target).sum(1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return ((1.0 - tversky.clamp(0, 1)) ** self.gamma).mean()


class BoundaryLoss(nn.Module):
    """L1 distance between the spatial gradients of prediction and target."""

    @staticmethod
    def gradient(x: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return F.pad(dx, (0, 1, 0, 0)) + F.pad(dy, (0, 0, 0, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(
            self.gradient(torch.sigmoid(pred)), self.gradient(target.float())
        )


class SegmentationLoss(nn.Module):
    """Dice + BCE, plus weighted Focal Tversky and boundary terms."""

    def __init__(
        self,
        lambda_dice: float = 0.70,
        lambda_ce: float = 0.30,
        tversky_weight: float = 0.20,
        boundary_weight: float = 0.03,
    ) -> None:
        super().__init__()
        self.region = DiceCELoss(lambda_dice, lambda_ce)
        self.tversky = FocalTverskyLoss(alpha=0.40, beta=0.60, gamma=0.75)
        self.boundary = BoundaryLoss()
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.region(pred, target)
            + self.tversky_weight * self.tversky(pred, target)
            + self.boundary_weight * self.boundary(pred, target)
        )


class DeepSupervisionLoss(nn.Module):
    """Segmentation loss on the main output plus the auxiliary decoder heads."""

    def __init__(
        self, ds_weights: tuple = (0.50, 0.25), tversky_weight: float = 0.20
    ) -> None:
        super().__init__()
        self.ds_weights = ds_weights
        self.seg_loss = SegmentationLoss(tversky_weight=tversky_weight)

    def forward(
        self, main_pred: torch.Tensor, ds_preds, target: torch.Tensor
    ) -> torch.Tensor:
        loss = self.seg_loss(main_pred, target)
        for weight, ds in zip(self.ds_weights, ds_preds):
            if ds.shape[2:] != target.shape[2:]:
                ds = F.interpolate(
                    ds, size=target.shape[2:], mode="bilinear", align_corners=False
                )
            loss = loss + weight * self.seg_loss(ds, target)
        return loss


class StableFocalLoss(nn.Module):
    """Focal cross-entropy with optional MIBC class weighting.

    Args:
        gamma: focusing parameter.
        mibc_alpha: loss weight of the MIBC class (index 1), compensating the
            NMIBC/MIBC imbalance. ``None`` disables class weighting.
        label_smoothing: label smoothing applied to the cross-entropy.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        mibc_alpha: float | None = None,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.mibc_alpha = mibc_alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -15.0, 15.0)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        weight = None
        if self.mibc_alpha is not None:
            weight = torch.tensor(
                [1.0, self.mibc_alpha], dtype=logits.dtype, device=logits.device
            )

        ce = F.cross_entropy(
            logits,
            targets,
            weight=weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class JointLoss(nn.Module):
    """Joint multi-task objective, Eq. (7) of the paper.

    ``cls_weight`` is overwritten each epoch by the training loop, which ramps
    the classification term in over the first cumulative epochs.
    """

    def __init__(
        self,
        seg_weight: float = 0.80,
        cls_weight: float = 0.20,
        focal_gamma: float = 2.0,
        mibc_alpha: float | None = None,
        ds_weights: tuple = (0.50, 0.25),
        tversky_weight: float = 0.20,
    ) -> None:
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_loss_fn = DeepSupervisionLoss(
            ds_weights=ds_weights, tversky_weight=tversky_weight
        )
        self.cls_loss_fn = StableFocalLoss(
            gamma=focal_gamma, mibc_alpha=mibc_alpha, label_smoothing=0.05
        )

    def forward(
        self,
        outputs: dict,
        seg_target: torch.Tensor,
        cls_target: torch.Tensor,
        cls_mask: torch.Tensor | None = None,
    ) -> dict:
        seg_pred = outputs["seg"]
        cls_pred = outputs["cls"]

        seg_loss = self.seg_loss_fn(seg_pred, outputs.get("ds", []), seg_target)
        seg_loss = torch.nan_to_num(seg_loss, nan=1.0, posinf=1.0, neginf=0.0)

        if self.cls_weight > 0:
            if cls_mask is not None:
                valid = cls_mask > 0
                cls_loss = (
                    self.cls_loss_fn(cls_pred[valid], cls_target[valid])
                    if valid.sum() > 0
                    else torch.zeros((), device=seg_pred.device)
                )
            else:
                cls_loss = self.cls_loss_fn(cls_pred, cls_target)
            cls_loss = torch.nan_to_num(cls_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            cls_loss = torch.zeros((), device=seg_pred.device)

        total = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        total = torch.nan_to_num(total, nan=1.0, posinf=1.0, neginf=0.0)

        return {
            "loss": total,
            "seg_loss": seg_loss.detach(),
            "cls_loss": cls_loss.detach(),
        }
