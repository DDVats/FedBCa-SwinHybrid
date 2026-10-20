"""Flower client: local training at one clinical centre.

One generic client serves all four centres. The centre is selected by the
``--center`` argument of ``scripts/federated_client.py``, so the four clients
differ only in the slices they load and never in their code.

Per communication round the client receives the global parameters, trains for
``local_epochs`` epochs on its own centre's slices, and returns its updated
parameters together with the number of local training samples, which the
server uses as the FedAvg weight.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import flwr as fl
import torch

from ..metrics.metrics import evaluate

logger = logging.getLogger(__name__)

__all__ = ["FedBCaClient"]


class FedBCaClient(fl.client.NumPyClient):
    """Local trainer for one centre.

    Args:
        model: the Swin Hybrid model instance.
        train_loader: dataloader over this centre's training slices.
        val_loader: dataloader over this centre's validation slices.
        criterion: the joint multi-task loss.
        optimizer: optimizer, built once and reused across rounds so that its
            AdamW moment estimates persist between rounds.
        config: client configuration (local epochs, accumulation, clipping).
        device: training device.
        center_id: index of the centre, used for logging only.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        config: dict,
        device: torch.device,
        center_id: int,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.center_id = center_id

        self.use_amp = bool(config.get("use_amp", False)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.use_amp)
        self.accum_steps = int(config.get("accum_steps", 4))
        self.grad_clip = float(config.get("grad_clip", 1.0))

    # -- parameter exchange -------------------------------------------------

    def get_parameters(self, config=None) -> list:
        """Return the model state as a list of NumPy arrays."""
        return [v.detach().cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters) -> None:
        """Load a list of NumPy arrays into the model state."""
        keys = list(self.model.state_dict().keys())
        if len(keys) != len(parameters):
            raise ValueError(
                f"Parameter count mismatch: model has {len(keys)} tensors, "
                f"server sent {len(parameters)}."
            )
        state = OrderedDict(
            (k, torch.as_tensor(v)) for k, v in zip(keys, parameters)
        )
        self.model.load_state_dict(state, strict=True)

    # -- federated rounds ---------------------------------------------------

    def fit(self, parameters, config) -> tuple:
        """Run the local epochs of one communication round."""
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", self.config.get("local_epochs", 3)))
        cls_weight = config.get("cls_weight")
        if cls_weight is not None:
            self.criterion.cls_weight = float(cls_weight)

        self.model.train()
        total_loss, total_steps = 0.0, 0

        for _ in range(local_epochs):
            self.optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(self.train_loader):
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
                cls_mask = (
                    batch["use_cls_loss"].float().to(self.device, non_blocking=True)
                )

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss_dict = self.criterion(
                        outputs, masks, labels, cls_mask=cls_mask
                    )
                    loss = loss_dict["loss"] / self.accum_steps

                if not torch.isfinite(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.scaler.scale(loss).backward()

                is_step = (step + 1) % self.accum_steps == 0 or (step + 1) == len(
                    self.train_loader
                )
                if is_step:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                    if torch.isfinite(grad_norm):
                        self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                total_loss += loss_dict["loss"].item()
                total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        logger.info(
            "[center %d] round %s: %d epochs, %d steps, loss=%.4f",
            self.center_id,
            config.get("current_round", "?"),
            local_epochs,
            total_steps,
            avg_loss,
        )

        # The sample count is the FedAvg aggregation weight.
        return (
            self.get_parameters(),
            len(self.train_loader.dataset),
            {"train_loss": float(avg_loss)},
        )

    def evaluate(self, parameters, config) -> tuple:
        """Evaluate the global model on this centre's validation split."""
        self.set_parameters(parameters)

        metrics = evaluate(self.model, self.val_loader, self.device, amp=self.use_amp)
        logger.info(
            "[center %d] round %s eval: DSC=%.4f AUC=%.4f combined=%.4f",
            self.center_id,
            config.get("current_round", "?"),
            metrics["tumour_dsc"],
            metrics["auc"],
            metrics["combined_score"],
        )

        return (
            float(1.0 - metrics["combined_score"]),
            len(self.val_loader.dataset),
            {
                "dsc": float(metrics["tumour_dsc"]),
                "auc": float(metrics["auc"]),
                "combined": float(metrics["combined_score"]),
                "center": int(self.center_id),
            },
        )
