"""Neural network building blocks used by the Swin Hybrid model.

This module contains only the blocks required by the proposed
architecture: the 2.5D input adapter, the attention gate of the
segmentation decoder, the decoder block, the deep-supervision head and
MixStyle feature-statistic mixing applied to the CNN branch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "gn",
    "freeze_batchnorm",
    "InputAdapter",
    "ResidualBlock",
    "AttentionGate",
    "DecoderBlock",
    "DeepSupervisionHead",
    "MixStyle",
    "initialize_weights",
]


def gn(channels: int, groups: int = 8) -> nn.GroupNorm:
    """Return a GroupNorm layer with a group count that divides ``channels``."""
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


def freeze_batchnorm(module: nn.Module) -> None:
    """Put every BatchNorm layer in eval mode and disable its gradients.

    The pretrained CNN branch is trained with frozen BatchNorm statistics.
    ``nn.Module.train()`` resets BatchNorm layers to training mode, so this
    helper must be called both at construction time and from every
    ``train()`` override that wraps the branch.
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


class InputAdapter(nn.Module):
    """Project the N-channel 2.5D stack to the 3 channels the backbones expect.

    The 1x1 convolution is initialised with uniform weights (``1 / in_ch``) so
    that, before any training step, each output channel carries the mean of the
    five input slices. The ImageNet-pretrained encoders therefore receive a
    plausible image on the very first forward pass rather than noise.
    """

    def __init__(self, in_ch: int = 5, out_ch: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.fill_(1.0 / in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ResidualBlock(nn.Module):
    """Two 3x3 GroupNorm convolutions with a projected residual connection."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            gn(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), gn(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


class AttentionGate(nn.Module):
    """Soft attention gate of the Attention U-Net decoder (Oktay et al., 2018).

    Args:
        f_g: channels of the gating signal (the coarser decoder feature).
        f_l: channels of the skip connection being gated.
        f_int: intermediate channel count of the gate.
    """

    def __init__(self, f_g: int, f_l: int, f_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=False), gn(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=False), gn(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=False), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = F.interpolate(
            self.W_g(g), size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return x * self.psi(self.relu(g1 + self.W_x(x)))


class DecoderBlock(nn.Module):
    """Upsample by two, concatenate the gated skip, then two residual blocks.

    Args:
        in_ch: channels arriving from the previous decoder stage.
        skip_ch: channels of the skip connection (``0`` for no skip).
        out_ch: output channels.
    """

    def __init__(
        self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ResidualBlock(in_ch + skip_ch, out_ch, dropout=dropout)
        self.conv2 = ResidualBlock(out_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class DeepSupervisionHead(nn.Module):
    """1x1 convolution producing an auxiliary segmentation logit map."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x: torch.Tensor, out_size: tuple) -> torch.Tensor:
        return F.interpolate(
            self.head(x), size=out_size, mode="bilinear", align_corners=False
        )


class MixStyle(nn.Module):
    """Feature-statistic mixing for domain generalisation (Zhou et al., 2021).

    Channel-wise means and standard deviations are mixed between samples of a
    batch, which pushes the CNN branch towards scanner-invariant features.
    Active during training only.
    """

    def __init__(self, p: float = 0.35, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or np.random.random() > self.p:
            return x
        b = x.size(0)
        if b < 2:
            return x
        mu = x.mean(dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        x_n = (x - mu) / sig
        perm = torch.randperm(b, device=x.device)
        lmda = (
            torch.distributions.Beta(self.alpha, self.alpha)
            .sample((b, 1, 1, 1))
            .to(x.device)
        )
        mu_m = lmda * mu + (1 - lmda) * mu[perm]
        sig_m = lmda * sig + (1 - lmda) * sig[perm]
        return x_n * sig_m + mu_m


def initialize_weights(module: nn.Module) -> None:
    """Kaiming-initialise convolutions and truncated-normal-initialise linears."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
