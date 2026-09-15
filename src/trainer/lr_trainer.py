"""Trainer name and interface for the LR CTR baseline."""

from __future__ import annotations

from .ctr_trainer_base import CTRTrainerBase


class LRTrainer(CTRTrainerBase):
    """Train and evaluate :class:`LogisticRegressionCTR`."""
