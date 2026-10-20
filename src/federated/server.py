"""Flower server: sample-size-weighted FedAvg aggregation.

Implements the protocol of Section 3.1: all four centres take part in every
round, parameters are averaged with weights proportional to each centre's
sample count (Eq. 4), and the global model with the highest validation
combined score is retained.

Client-reported DSC and AUC are aggregated the same way, so the round's
combined score is the sample-weighted mean across the four centres.
"""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

import flwr as fl
import numpy as np
from flwr.common import parameters_to_ndarrays

from ..metrics.metrics import AUC_WEIGHT, DSC_WEIGHT

logger = logging.getLogger(__name__)

__all__ = ["weighted_average", "aggregate_train_loss", "FedBCaStrategy", "build_strategy"]


def weighted_average(metrics: list) -> dict:
    """Aggregate client metrics by sample count."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {"dsc": 0.0, "auc": 0.0, "combined": 0.0}

    dsc = sum(n * m.get("dsc", 0.0) for n, m in metrics) / total
    auc = sum(n * m.get("auc", 0.0) for n, m in metrics) / total

    return {
        "dsc": dsc,
        "auc": auc,
        "combined": sum(n * m.get("combined", 0.0) for n, m in metrics) / total,
        "combined_from_means": DSC_WEIGHT * dsc + AUC_WEIGHT * auc,
    }


def aggregate_train_loss(metrics: list) -> dict:
    """Aggregate the clients' training losses by sample count."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {"train_loss": 0.0}
    return {
        "train_loss": sum(n * m.get("train_loss", 0.0) for n, m in metrics) / total
    }


class FedBCaStrategy(fl.server.strategy.FedAvg):
    """FedAvg with per-round logging and best-model retention.

    Args:
        output_dir: directory for the round log and the global checkpoints.
        checkpoint_every: also save a numbered checkpoint every N rounds
            (``0`` disables it).
    """

    def __init__(self, output_dir: str | Path, checkpoint_every: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = int(checkpoint_every)

        self.best_combined = -1.0
        self.best_round = 0
        self.last_train_loss = 0.0
        self.history: list[dict] = []

        self.log_path = self.output_dir / "round_metrics.csv"
        with self.log_path.open("w", newline="") as fh:
            csv.writer(fh).writerow(["round", "dsc", "auc", "combined", "train_loss"])

    def aggregate_fit(self, server_round: int, results, failures):
        """Average the client parameters and persist the global model."""
        parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if failures:
            logger.warning("round %d: %d client failure(s)", server_round, len(failures))
        if metrics:
            self.last_train_loss = float(metrics.get("train_loss", 0.0))

        if parameters is not None:
            arrays = parameters_to_ndarrays(parameters)
            np.savez(self.output_dir / "global_model_latest.npz", *arrays)
            if self.checkpoint_every and server_round % self.checkpoint_every == 0:
                np.savez(
                    self.output_dir / f"global_model_round_{server_round:03d}.npz",
                    *arrays,
                )

        return parameters, metrics

    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate validation metrics and retain the best global model."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        if not (results and metrics):
            return loss, metrics

        dsc = float(metrics.get("dsc", 0.0))
        auc = float(metrics.get("auc", 0.0))
        combined = float(metrics.get("combined", 0.0))

        with self.log_path.open("a", newline="") as fh:
            csv.writer(fh).writerow(
                [server_round, dsc, auc, combined, self.last_train_loss]
            )

        self.history.append(
            {"round": server_round, "dsc": dsc, "auc": auc, "combined": combined}
        )

        if combined > self.best_combined:
            self.best_combined = combined
            self.best_round = server_round
            latest = self.output_dir / "global_model_latest.npz"
            if latest.is_file():
                shutil.copy(latest, self.output_dir / "global_model_best.npz")
            logger.info(
                "round %d: new best DSC=%.4f AUC=%.4f combined=%.4f",
                server_round,
                dsc,
                auc,
                combined,
            )
        else:
            logger.info(
                "round %d: DSC=%.4f AUC=%.4f combined=%.4f",
                server_round,
                dsc,
                auc,
                combined,
            )

        return loss, metrics


def build_strategy(
    output_dir: str | Path,
    num_clients: int = 4,
    local_epochs: int = 3,
    checkpoint_every: int = 10,
) -> FedBCaStrategy:
    """Build the FedAvg strategy of the paper.

    All ``num_clients`` centres are required to fit and to evaluate in every
    round, which reproduces the full-participation protocol of Section 3.1.
    """

    def fit_config(server_round: int) -> dict:
        return {"current_round": server_round, "local_epochs": local_epochs}

    def evaluate_config(server_round: int) -> dict:
        return {"current_round": server_round}

    return FedBCaStrategy(
        output_dir=output_dir,
        checkpoint_every=checkpoint_every,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=aggregate_train_loss,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )
