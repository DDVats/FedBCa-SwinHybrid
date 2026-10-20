#!/usr/bin/env python3
"""Centralised training of the Swin Hybrid model.

Example::

    python scripts/train.py --config configs/swin_hybrid.yaml \\
        --augmentation geo_elastic

All data locations come from the configuration or the command line; nothing
machine-specific is hard-coded.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Make the repository root importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augmentation import build_augmentation
from src.datasets import FedBCaDataset, build_dataloader, load_split
from src.losses import JointLoss
from src.models import build_model
from src.training import train_model
from src.utils import load_config, merge_overrides, resolve_path

logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swin_hybrid.yaml")
    parser.add_argument("--preprocessed-root", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--augmentation",
        default=None,
        help="none, gin_only, stack, geo_elastic, geo_bias, geo_gamma, geo_noise",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp-name", default=None, help="checkpoint sub-directory")
    parser.add_argument("--device", default=None, help="cuda, cpu, ...")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    config = merge_overrides(
        load_config(args.config),
        preprocessed_root=args.preprocessed_root,
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        augmentation=args.augmentation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    set_seed(int(config.get("seed", 42)))

    preprocessed_root = resolve_path(config["preprocessed_root"], "preprocessed_root")
    splits_dir = resolve_path(config["splits_dir"], "splits_dir")
    output_dir = resolve_path(config["output_dir"], "output_dir")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    strategy = str(config.get("augmentation", "geo_elastic"))
    exp_name = args.exp_name or f"{config.get('name', 'swin_hybrid')}_{strategy}"

    logger.info("Experiment   : %s", exp_name)
    logger.info("Augmentation : %s", strategy)
    logger.info("Device       : %s", device)

    train_ds = FedBCaDataset(
        load_split(splits_dir / "train.csv"),
        preprocessed_root,
        mode="train",
        augmentation=build_augmentation(strategy),
    )
    val_ds = FedBCaDataset(
        load_split(splits_dir / "val.csv"), preprocessed_root, mode="val"
    )

    num_workers = int(config.get("num_workers", 4))
    batch_size = int(config.get("batch_size", 6))
    train_loader = build_dataloader(train_ds, batch_size, num_workers=num_workers)
    val_loader = build_dataloader(val_ds, batch_size, num_workers=num_workers)

    def n_patients(dataset) -> int:
        return len({r["patient_id"] for r in dataset.records})

    logger.info(
        "train: %d slices / %d patients | val: %d slices / %d patients",
        len(train_ds),
        n_patients(train_ds),
        len(val_ds),
        n_patients(val_ds),
    )

    model = build_model(
        in_channels=int(config.get("in_channels", 5)),
        num_classes=int(config.get("num_classes", 2)),
        img_size=int(config.get("input_size", 224)),
        use_mixstyle=bool(config.get("use_mixstyle", True)),
        dropout=float(config.get("dropout", 0.15)),
        pretrained=bool(config.get("pretrained", True)),
    )

    criterion = JointLoss(
        seg_weight=float(config.get("seg_weight", 0.80)),
        cls_weight=float(config.get("cls_weight", 0.20)),
        focal_gamma=float(config.get("focal_gamma", 2.0)),
        mibc_alpha=config.get("mibc_alpha"),
        ds_weights=tuple(config.get("ds_weights", (0.50, 0.25))),
        tversky_weight=float(config.get("tversky_weight", 0.20)),
    )

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
        exp_name=exp_name,
        save_dir=output_dir,
        device=device,
    )

    logger.info(
        "Best validation: DSC=%.4f AUC=%.4f combined=%.4f",
        result["best_dsc"],
        result["best_auc"],
        result["combined"],
    )
    logger.info("Checkpoint: %s", result["checkpoint"])


if __name__ == "__main__":
    main()
