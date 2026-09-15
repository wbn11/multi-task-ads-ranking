"""Shared trainer diagnostics for task-gated expert models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from torch import Tensor

from src.metrics.gate import compute_gate_metrics

from .shared_bottom_trainer import SharedBottomTrainer


class ExpertGateDiagnosticsMixin:
    """Collect task-gate routing metrics without choosing a loss function."""

    def _evaluation_forward(
        self,
        batch: Mapping[str, Any],
    ) -> Mapping[str, Tensor]:
        return self.model(batch, return_gate_weights=True)

    def _create_evaluation_diagnostics(
        self,
    ) -> dict[str, list[np.ndarray]]:
        return {"ctr": [], "cvr": []}

    def _update_evaluation_diagnostics(
        self,
        state: dict[str, list[np.ndarray]],
        outputs: Mapping[str, Tensor],
    ) -> None:
        for task in ("ctr", "cvr"):
            key = f"{task}_gate_weights"
            weights = outputs.get(key)
            if not isinstance(weights, Tensor) or weights.ndim != 2:
                raise RuntimeError(
                    f"expert-gated evaluation requires {key} with shape "
                    "[batch, experts]"
                )
            state[task].append(weights.float().cpu().numpy())

    def _finalize_evaluation_diagnostics(
        self,
        state: dict[str, list[np.ndarray]],
    ) -> dict[str, Any]:
        if any(not parts for parts in state.values()):
            raise RuntimeError(
                "expert-gated evaluation produced no gate weights"
            )
        return {
            "gates": {
                task: compute_gate_metrics(np.concatenate(parts, axis=0))
                for task, parts in state.items()
            }
        }


class ExpertGateTrainer(ExpertGateDiagnosticsMixin, SharedBottomTrainer):
    """Train with masked CVR loss and summarize task gate routing."""
