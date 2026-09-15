"""Trainer name and interface for the DeepFM CTR model."""

from __future__ import annotations

from .ctr_trainer_base import CTRTrainerBase


class DeepFMTrainer(CTRTrainerBase):
    """Train and evaluate :class:`DeepFM`."""
