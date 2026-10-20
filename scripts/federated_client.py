#!/usr/bin/env python3
"""Start one federated client for a single clinical centre.

One script serves all four centres; the centre is given on the command line::

    python scripts/federated_client.py --center 1

The client loads only the slices of its own centre, so no data ever leaves the
site: only model parameters are exchanged with the server.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import flwr as fl
import numpy as np
import torch

# Make the repository root importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augmentation import build_augmentation
from src.datasets import FedBCaDataset, build_dataloader, load_split
from src.federated import FedBCaClient
from src.losses import JointLoss
from src.models import build_model
from src.training import build_optimizer
from src.utils import load_config, merge_overrides, resolve_path

logger = logging.getLogger("federated_client")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--center", type=int, required=True, help="centre index (1-4)"
    )
    parser.add_argument("--config", default="configs/federated.yaml")
    parser.add_argument("--server-address", default=None, help="host:port of the server")
    parser.add_argument("--preprocessed-root", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--augmentation", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    config = merge_overrides(
        load_config(args.config),
        server_address=args.server_address,
        preprocessed_root=args.preprocessed_root,
        splits_dir=args.splits_dir,
        augmentation=args.augmentation,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Give each centre its own stream, while keeping the run reproducible.
    seed = int(config.get("seed", 42)) + args.center
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    preprocessed_root = resolve_path(config["preprocessed_root"], "preprocessed_root")
    splits_dir = resolve_path(config["splits_dir"], "splits_dir")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    strategy = str(config.get("augmentation", "geo_elastic"))
    batch_size = int(config.get("batch_size", 6))
    num_workers = int(config.get("num_workers", 0))

    train_ds = FedBCaDataset(
        load_split(splits_dir / "train.csv", center=args.center),
        preprocessed_root,
        mode="train",
        augmentation=build_augmentation(strategy),
    )
    val_ds = FedBCaDataset(
        load_split(splits_dir / "val.csv", center=args.center),
        preprocessed_root,
        mode="val",
    )

    # Built once and reused every round: rebuilding the loaders each round
    # costs far more than it saves when four clients share one machine.
    train_loader = build_dataloader(train_ds, batch_size, num_workers=num_workers)
    val_loader = build_dataloader(val_ds, batch_size, num_workers=num_workers)

    logger.info("Center       : %d", args.center)
    logger.info("Augmentation : %s", strategy)
    logger.info(
        "train: %d slices / %d patients | val: %d slices / %d patients",
        len(train_ds),
        len({r["patient_id"] for r in train_ds.records}),
        len(val_ds),
        len({r["patient_id"] for r in val_ds.records}),
    )

    model = build_model(
        in_channels=int(config.get("in_channels", 5)),
        num_classes=int(config.get("num_classes", 2)),
        img_size=int(config.get("input_size", 224)),
        use_mixstyle=bool(config.get("use_mixstyle", True)),
        dropout=float(config.get("dropout", 0.15)),
        pretrained=bool(config.get("pretrained", True)),
    ).to(device)

    criterion = JointLoss(
        seg_weight=float(config.get("seg_weight", 0.80)),
        cls_weight=float(config.get("cls_weight", 0.20)),
        focal_gamma=float(config.get("focal_gamma", 2.0)),
        mibc_alpha=config.get("mibc_alpha"),
        ds_weights=tuple(config.get("ds_weights", (0.50, 0.25))),
        tversky_weight=float(config.get("tversky_weight", 0.20)),
    )

    optimizer = build_optimizer(model, config)

    client = FedBCaClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device,
        center_id=args.center,
    )

    address = str(config.get("server_address", "127.0.0.1:8080")).replace(
        "0.0.0.0", "127.0.0.1"
    )
    logger.info("Connecting to %s", address)

    fl.client.start_client(server_address=address, client=client.to_client())
    logger.info("Center %d finished.", args.center)


if __name__ == "__main__":
    main()
