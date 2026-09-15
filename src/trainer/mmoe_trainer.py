"""Trainer configuration for the MMoE model."""

from __future__ import annotations

from .expert_gate_trainer import ExpertGateTrainer


class MMoETrainer(ExpertGateTrainer):
    """Train MMoE with masked CVR loss and task gate diagnostics."""
