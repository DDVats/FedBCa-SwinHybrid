#!/usr/bin/env python3
"""Start the FedAvg server.

Launch the server first, then one client per centre::

    python scripts/federated_server.py --config configs/federated.yaml
    python scripts/federated_client.py --center 1
    python scripts/federated_client.py --center 2
    python scripts/federated_client.py --center 3
    python scripts/federated_client.py --center 4

The server waits for all four clients, runs 50 communication rounds, logs the
per-round metrics to ``round_metrics.csv`` and keeps the global model with the
highest validation combined score as ``global_model_best.npz``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import flwr as fl

# Make the repository root importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.federated import build_strategy
from src.utils import load_config, merge_overrides, resolve_path

logger = logging.getLogger("federated_server")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/federated.yaml")
    parser.add_argument("--server-address", default=None, help="host:port to bind")
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--run-name",
        default=None,
        help="sub-directory for this run (defaults to the augmentation name)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    config = merge_overrides(
        load_config(args.config),
        server_address=args.server_address,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        local_epochs=args.local_epochs,
        output_dir=args.output_dir,
    )

    run_name = args.run_name or str(config.get("augmentation", "geo_elastic"))
    output_dir = resolve_path(config["output_dir"], "output_dir") / run_name
    num_rounds = int(config.get("num_rounds", 50))
    num_clients = int(config.get("num_clients", 4))

    strategy = build_strategy(
        output_dir=output_dir,
        num_clients=num_clients,
        local_epochs=int(config.get("local_epochs", 3)),
        checkpoint_every=int(config.get("checkpoint_every", 10)),
    )

    address = str(config.get("server_address", "0.0.0.0:8080"))
    logger.info("Run          : %s", run_name)
    logger.info("Output dir   : %s", output_dir)
    logger.info("Rounds       : %d, clients: %d", num_rounds, num_clients)
    logger.info("Waiting for %d clients on %s", num_clients, address)

    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    logger.info(
        "Finished. Best combined score %.4f at round %d.",
        strategy.best_combined,
        strategy.best_round,
    )
    logger.info("Best global model: %s", output_dir / "global_model_best.npz")


if __name__ == "__main__":
    main()
