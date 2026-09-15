"""Trainer for the DCN-PLE-ESMM final candidate."""

from __future__ import annotations

from typing import Any

import numpy as np

from .esmm_trainer import ESMMTrainer
from .expert_gate_trainer import ExpertGateDiagnosticsMixin


class DCNPLEESMMTrainer(ExpertGateDiagnosticsMixin, ESMMTrainer):
    """Use ESMM loss while retaining PLE gate-routing diagnostics."""

    def _finalize_evaluation_diagnostics(
        self,
        state: dict[str, list[np.ndarray]],
    ) -> dict[str, Any]:
        diagnostics = super()._finalize_evaluation_diagnostics(state)
        num_shared_experts = int(self.model.num_shared_experts)
        num_task_experts = int(self.model.num_task_experts)
        shared_names = [
            f"shared_{index}" for index in range(1, num_shared_experts + 1)
        ]
        diagnostics["gate_expert_order"] = {
            "ctr": [
                *shared_names,
                *(
                    f"ctr_specific_{index}"
                    for index in range(1, num_task_experts + 1)
                ),
            ],
            "cvr": [
                *shared_names,
                *(
                    f"cvr_specific_{index}"
                    for index in range(1, num_task_experts + 1)
                ),
            ],
        }
        return diagnostics
