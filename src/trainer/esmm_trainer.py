"""Trainer configuration for the ESMM entire-space objective."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.losses.multitask_loss import (
    ESMMWithAuxiliaryCVRLoss,
    build_esmm_loss,
)

from .multitask_trainer_base import MultiTaskTrainerBase


class ESMMTrainer(MultiTaskTrainerBase):
    """Train ESMM with exposure-space CTR and CTCVR objectives."""

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
        criterion = build_esmm_loss(loss_config)
        auxiliary_cvr_enabled = isinstance(
            criterion,
            ESMMWithAuxiliaryCVRLoss,
        )
        component_scopes = {
            "ctr_loss": "all",
            "ctcvr_loss": "all",
        }
        component_weights = {
            "ctr_loss": criterion.ctr_weight,
            "ctcvr_loss": criterion.ctcvr_weight,
        }
        count_metrics: tuple[str, ...] = ()
        component_count_metrics: dict[str, str] = {}
        if auxiliary_cvr_enabled:
            component_scopes["auxiliary_cvr_loss"] = "clicked"
            component_weights["auxiliary_cvr_loss"] = (
                criterion.auxiliary_cvr_weight
            )
            count_metrics = (
                "auxiliary_cvr_positive_count",
                "auxiliary_cvr_negative_count",
                "auxiliary_cvr_sampled_count",
            )
            component_count_metrics["auxiliary_cvr_loss"] = (
                "auxiliary_cvr_sampled_count"
            )
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            loss_component_scopes=component_scopes,
            loss_component_weights=component_weights,
            device=device,
            amp=amp,
            gradient_clip_norm=gradient_clip_norm,
            logger=logger,
            loss_count_metrics=count_metrics,
            loss_component_count_metrics=component_count_metrics,
        )
