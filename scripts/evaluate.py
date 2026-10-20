#!/usr/bin/env python3
"""Evaluate a trained checkpoint: DSC, patient-level AUC and combined score.

Works with a centralised checkpoint (``checkpoint_best.pt``) or with a global
federated model (``global_model_best.npz``).

Example::

    python scripts/evaluate.py --config configs/swin_hybrid.yaml \\
        --checkpoint outputs/swin_hybrid_geo_elastic/checkpoint_best.pt \\
        --split test --per-center
"""

from __future__ import annotations

from collections import OrderedDict
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Make the repository root importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import FedBCaDataset, build_dataloader, load_split
from src.metrics import evaluate
from src.models import build_model
from src.utils import load_config, merge_overrides, resolve_path

logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swin_hybrid.yaml")
    parser.add_argument("--checkpoint", required=True, help=".pt or .npz checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--preprocessed-root", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--per-center", action="store_true", help="also report per-centre metrics"
    )
    parser.add_argument("--output", default=None, help="write the metrics to JSON")
    return parser.parse_args()


def load_checkpoint(model, path: Path, device: torch.device) -> None:
    """Load a PyTorch checkpoint or a Flower ``.npz`` parameter dump."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix == ".npz":
        with np.load(path) as data:
            arrays = [data[k] for k in data.files]
        keys = list(model.state_dict().keys())
        if len(keys) != len(arrays):
            raise ValueError(
                f"Checkpoint has {len(arrays)} tensors but the model expects "
                f"{len(keys)}. Was it produced by a different architecture?"
            )
        model.load_state_dict(
            OrderedDict((k, torch.as_tensor(v)) for k, v in zip(keys, arrays)),
            strict=True,
        )
        return

    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("state", checkpoint)
    model.load_state_dict(state, strict=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    config = merge_overrides(
        load_config(args.config),
        preprocessed_root=args.preprocessed_root,
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
    )

    preprocessed_root = resolve_path(config["preprocessed_root"], "preprocessed_root")
    splits_dir = resolve_path(config["splits_dir"], "splits_dir")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = build_model(
        in_channels=int(config.get("in_channels", 5)),
        num_classes=int(config.get("num_classes", 2)),
        img_size=int(config.get("input_size", 224)),
        use_mixstyle=bool(config.get("use_mixstyle", True)),
        dropout=float(config.get("dropout", 0.15)),
        pretrained=False,  # weights come from the checkpoint
    )
    load_checkpoint(model, Path(args.checkpoint), device)
    model = model.to(device).eval()

    split_df = load_split(splits_dir / f"{args.split}.csv")
    batch_size = int(config.get("batch_size", 6))
    num_workers = int(config.get("num_workers", 4))
    use_amp = bool(config.get("use_amp", False))

    dataset = FedBCaDataset(split_df, preprocessed_root, mode="test")
    loader = build_dataloader(dataset, batch_size, num_workers=num_workers)
    metrics = evaluate(model, loader, device, amp=use_amp)

    logger.info("--- %s split ---", args.split)
    logger.info("DSC            : %.4f", metrics["tumour_dsc"])
    logger.info("Patient AUC    : %.4f", metrics["auc"])
    logger.info("Combined score : %.4f", metrics["combined_score"])
    logger.info("Patients       : %d", metrics["n_patients"])

    results = {"overall": metrics}

    if args.per_center:
        results["per_center"] = {}
        for center in sorted(split_df["center"].unique()):
            center_df = split_df[split_df["center"] == center]
            center_loader = build_dataloader(
                FedBCaDataset(center_df, preprocessed_root, mode="test"),
                batch_size,
                num_workers=num_workers,
            )
            center_metrics = evaluate(model, center_loader, device, amp=use_amp)
            results["per_center"][int(center)] = center_metrics
            logger.info(
                "Center %s: DSC=%.4f AUC=%.4f combined=%.4f (%d patients)",
                center,
                center_metrics["tumour_dsc"],
                center_metrics["auc"],
                center_metrics["combined_score"],
                center_metrics["n_patients"],
            )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as fh:
            json.dump(results, fh, indent=2)
        logger.info("Metrics written to %s", output_path)


if __name__ == "__main__":
    main()
