"""Swin Hybrid: the proposed multi-task CNN-Transformer model.

Pipeline::

    5-channel 2.5D stack
        -> InputAdapter (5 -> 3 channels)
        -> dual-stream encoder
               CNN branch    : ImageNet-pretrained ResNet-34 (+ MixStyle)
               Transformer   : ImageNet-pretrained Swin-Tiny
        -> intermediate feature fusion (concatenation + 1x1 projection)
        -> attention-gated decoder with deep supervision  -> tumour mask
        -> segmentation-guided attention pooling          -> MIBC / NMIBC

The segmentation probability map is detached before it is used to pool the
deepest Transformer feature, so classification gradients never flow back
through the segmentation prediction.

``timm`` returns Swin features in BHWC layout; they are permuted to BCHW
before fusion with the CNN stream.
"""

from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    AttentionGate,
    DecoderBlock,
    DeepSupervisionHead,
    InputAdapter,
    MixStyle,
    freeze_batchnorm,
    gn,
    initialize_weights,
)

logger = logging.getLogger(__name__)

__all__ = ["HybridEncoder", "SwinHybridMultiTask", "build_model"]


class HybridEncoder(nn.Module):
    """Dual-stream encoder: ResNet-34 local features + Swin-Tiny global context.

    Returns a dict with the two CNN feature maps (``l1``, ``l2``) and the three
    Transformer feature maps (``s1``, ``s2``, ``s3``), all in BCHW layout.
    """

    def __init__(
        self,
        use_mixstyle: bool = True,
        pretrained: bool = True,
        img_size: int = 224,
        mixstyle_p: float = 0.35,
        mixstyle_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.img_size = int(img_size)

        # --- CNN branch: local texture and boundary features ---------------
        self.cnn = timm.create_model(
            "resnet34", pretrained=pretrained, features_only=True, out_indices=(1, 2)
        )
        freeze_batchnorm(self.cnn)

        self.mix1 = (
            MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if use_mixstyle else nn.Identity()
        )
        self.mix2 = (
            MixStyle(p=mixstyle_p, alpha=mixstyle_alpha) if use_mixstyle else nn.Identity()
        )

        # --- Transformer branch: global anatomical context -----------------
        # The Swin-Tiny checkpoint is defined at 224x224 and its patch
        # embedding asserts the input size, so ``img_size`` must be passed
        # explicitly for any other input resolution (the released default is
        # 224x224, the resolution of the Swin Hybrid runs; 256x256 also works);
        # timm then adapts the position bias to the new window grid.
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2),
            img_size=self.img_size,
        )

        self.out_ch = {"l1": 64, "l2": 128, "s1": 96, "s2": 192, "s3": 384}

    def train(self, mode: bool = True):  # noqa: D102 - see freeze_batchnorm
        super().train(mode)
        freeze_batchnorm(self.cnn)
        return self

    def forward(self, x: torch.Tensor) -> dict:
        cnn_feats = self.cnn(x)
        l1 = self.mix1(cnn_feats[0])
        l2 = self.mix2(cnn_feats[1])

        # Per-sample scale normalisation keeps the Swin branch numerically
        # stable on z-scored MRI, whose dynamic range differs from ImageNet.
        x_std = x.std(dim=[1, 2, 3], keepdim=True).clamp(min=1e-6)
        x_swin = torch.nan_to_num(x / x_std, nan=0.0, posinf=0.0, neginf=0.0)

        swin_feats = self.swin(x_swin)
        s1, s2, s3 = (f.permute(0, 3, 1, 2).contiguous() for f in swin_feats)
        s1 = torch.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)
        s2 = torch.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)
        s3 = torch.nan_to_num(s3, nan=0.0, posinf=0.0, neginf=0.0)

        return {"l1": l1, "l2": l2, "s1": s1, "s2": s2, "s3": s3}


class SwinHybridMultiTask(nn.Module):
    """Joint tumour segmentation and MIBC/NMIBC classification network.

    Args:
        in_channels: channels of the 2.5D input stack (five in the paper).
        num_classes: classification outputs (two: NMIBC and MIBC).
        use_mixstyle: apply MixStyle to the CNN branch.
        dropout: dropout rate of the decoder blocks.
        pretrained: load ImageNet weights for both encoder branches.
        img_size: side length of the square input. It must match the
            preprocessed data, because the Swin branch is built for this
            resolution and will reject any other.

    Forward returns a dict with ``seg`` (logits, input resolution), ``cls``
    (class logits) and ``ds`` (auxiliary deep-supervision logits, training
    mode only).
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 2,
        use_mixstyle: bool = True,
        dropout: float = 0.15,
        pretrained: bool = True,
        img_size: int = 224,
    ) -> None:
        super().__init__()

        self.img_size = int(img_size)
        self.input_adapter = InputAdapter(in_ch=in_channels, out_ch=3)
        self.encoder = HybridEncoder(
            use_mixstyle=use_mixstyle, pretrained=pretrained, img_size=self.img_size
        )
        ch = self.encoder.out_ch

        # Intermediate fusion of the CNN and Transformer streams.
        self.skip_proj = nn.Sequential(
            nn.Conv2d(ch["s1"] + ch["l2"], 128, 1, bias=False),
            gn(128),
            nn.GELU(),
        )

        # Attention gates filter each skip before it reaches the decoder.
        self.ag3 = AttentionGate(ch["s3"], ch["s2"], 96)
        self.ag2 = AttentionGate(192, 128, 64)
        self.ag1 = AttentionGate(96, ch["l1"], 32)

        self.dec4 = DecoderBlock(ch["s3"], ch["s2"], 192, dropout=dropout)
        self.dec3 = DecoderBlock(192, 128, 96, dropout=dropout)
        self.dec2 = DecoderBlock(96, ch["l1"], 48, dropout=max(0.0, dropout - 0.05))
        self.dec1 = DecoderBlock(48, 0, 24, dropout=0.0)

        self.seg_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(24, 16, 3, padding=1, bias=False),
            gn(16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

        self.ds1 = DeepSupervisionHead(192)
        self.ds2 = DeepSupervisionHead(96)

        # Classifier over the segmentation-guided pooled Transformer feature.
        self.cls_proj = nn.Sequential(
            nn.Linear(ch["s3"], 128, bias=False),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.05),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, num_classes),
        )

        # Small final-layer init keeps early logits near zero while the
        # classification loss is still ramping up.
        with torch.no_grad():
            last = [m for m in self.cls_proj.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.trunc_normal_(last.weight, std=0.01)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

        initialize_weights(self.seg_head)
        initialize_weights(self.dec1)
        initialize_weights(self.skip_proj)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "SwinHybrid built: in_ch=%d mixstyle=%s params=%d trainable=%d",
            in_channels,
            use_mixstyle,
            total,
            trainable,
        )

    def train(self, mode: bool = True):  # noqa: D102 - see freeze_batchnorm
        super().train(mode)
        freeze_batchnorm(self.encoder.cnn)
        return self

    def forward(self, x: torch.Tensor) -> dict:
        _, _, h, w = x.shape

        enc = self.encoder(self.input_adapter(x))
        l1, l2 = enc["l1"], enc["l2"]
        s1, s2, s3 = enc["s1"], enc["s2"], enc["s3"]

        # --- feature fusion -------------------------------------------------
        if l2.shape[2:] != s1.shape[2:]:
            s1 = F.interpolate(
                s1, size=l2.shape[2:], mode="bilinear", align_corners=False
            )
        skip_mid = self.skip_proj(torch.cat([s1, l2], dim=1))

        # --- attention-gated decoder ---------------------------------------
        d4 = self.dec4(s3, self.ag3(s3, s2))
        d3 = self.dec3(d4, self.ag2(d4, skip_mid))

        if l1.shape[2:] != (h // 4, w // 4):
            l1 = F.interpolate(
                l1, size=(h // 4, w // 4), mode="bilinear", align_corners=False
            )

        d2 = self.dec2(d3, self.ag1(d3, l1))
        d1 = self.dec1(d2)

        seg = self.seg_head(d1)
        if seg.shape[2:] != (h, w):
            seg = F.interpolate(seg, size=(h, w), mode="bilinear", align_corners=False)
        seg = torch.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

        ds_outs = [self.ds1(d4, (h, w)), self.ds2(d3, (h, w))] if self.training else []

        # --- segmentation-guided attention pooling --------------------------
        # The map is detached: classification gradients must not propagate
        # through the segmentation prediction.
        seg_attn = torch.sigmoid(seg.detach())
        seg_attn = F.interpolate(
            seg_attn, size=s3.shape[2:], mode="bilinear", align_corners=False
        )
        seg_attn = seg_attn / seg_attn.sum(dim=[2, 3], keepdim=True).clamp(min=1e-6)

        pooled = (s3 * seg_attn).sum(dim=[2, 3])
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        cls = self.cls_proj(pooled)

        return {"seg": seg, "cls": cls, "ds": ds_outs}


def build_model(
    in_channels: int = 5,
    num_classes: int = 2,
    use_mixstyle: bool = True,
    dropout: float = 0.15,
    pretrained: bool = True,
    img_size: int = 224,
    **_: object,
) -> SwinHybridMultiTask:
    """Construct the proposed Swin Hybrid model. Extra kwargs are ignored."""
    return SwinHybridMultiTask(
        in_channels=in_channels,
        num_classes=num_classes,
        use_mixstyle=use_mixstyle,
        dropout=dropout,
        pretrained=pretrained,
        img_size=img_size,
    )
