#!/usr/bin/env python3
"""Run the Swin Hybrid model on one preprocessed slice stack.

With a checkpoint and a preprocessed slice::

    python examples/demo_inference.py \\
        --checkpoint outputs/swin_hybrid_geo_elastic/checkpoint_best.pt \\
        --input data/preprocessed/C1_001_s012_img.npy \\
        --save-overlay overlay.png

Without either, the script builds the model on random weights and a synthetic
input, which is enough to check that the installation works end to end. The
numbers it prints in that mode are meaningless.

This is a research demonstration. It does not provide a clinical diagnosis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make the repository root importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="trained .pt checkpoint")
    parser.add_argument(
        "--input", default=None, help="preprocessed <fname>_img.npy slice stack"
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--in-channels", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-overlay", default=None, help="write a PNG overlay")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_input(args) -> tuple:
    """Return ``(tensor, is_synthetic)`` of shape ``(1, C, H, W)``."""
    if args.input is None:
        print("No --input given: using a synthetic volume (outputs are meaningless).")
        array = np.random.randn(args.in_channels, args.input_size, args.input_size)
        return torch.from_numpy(array.astype(np.float32))[None], True

    path = Path(args.input)
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")

    array = np.load(path).astype(np.float32)
    if array.ndim != 3:
        raise ValueError(
            f"Expected a (C, H, W) slice stack, got shape {array.shape}."
        )
    return torch.from_numpy(array)[None], False


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Without a checkpoint the weights carry no meaning either way, so the
    # model is built untrained and the demo stays runnable offline.
    model = build_model(
        in_channels=args.in_channels, img_size=args.input_size, pretrained=False
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get("state", checkpoint), strict=True)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No --checkpoint given: the model is untrained; outputs are meaningless.")

    model = model.to(device).eval()

    images, synthetic = load_input(args)
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    probability = torch.sigmoid(outputs["seg"])[0, 0].cpu().numpy()
    mask = (probability > args.threshold).astype(np.uint8)
    mibc_probability = torch.softmax(outputs["cls"], dim=1)[0, 1].item()

    print(f"Segmentation map   : {probability.shape}")
    print(f"Predicted tumour px: {int(mask.sum())}")
    print(f"MIBC probability   : {mibc_probability:.4f}")
    if synthetic:
        print("(synthetic input - these values carry no meaning)")

    if args.save_overlay:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        centre_slice = images[0, args.in_channels // 2].cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(centre_slice, cmap="gray")
        axes[0].set_title("Centre slice")
        axes[1].imshow(probability, cmap="magma", vmin=0, vmax=1)
        axes[1].set_title("Tumour probability")
        axes[2].imshow(centre_slice, cmap="gray")
        axes[2].imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.5)
        axes[2].set_title(f"Overlay (MIBC p={mibc_probability:.2f})")
        for ax in axes:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(args.save_overlay, dpi=150)
        print(f"Overlay written to {args.save_overlay}")


if __name__ == "__main__":
    main()
