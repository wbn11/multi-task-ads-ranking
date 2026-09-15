"""Trainer configuration for the Shared Bottom multi-task baseline."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.losses.multitask_loss import MaskedCVRMultiTaskLoss

from .multitask_trainer_base import MultiTaskTrainerBase


class SharedBottomTrainer(MultiTaskTrainerBase):
    """Train Shared Bottom with CTR BCE plus clicked-space CVR BCE."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        amp: bool,
        gradient_clip_norm: float | None,
        logger: logging.Logger,
        loss_config: Mapping[str, Any],
    ) -> None:
        criterion = MaskedCVRMultiTaskLoss(
            ctr_weight=float(loss_config["ctr_weight"]),
            cvr_weight=float(loss_config["cvr_weight"]),
        )
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            loss_component_scopes={
                "ctr_loss": "all",
                "cvr_loss": "clicked",
            },
            loss_component_weights={
                "ctr_loss": criterion.ctr_weight,
                "cvr_loss": criterion.cvr_weight,
            },
            device=device,
            amp=amp,
            gradient_clip_norm=gradient_clip_norm,
            logger=logger,
        )
