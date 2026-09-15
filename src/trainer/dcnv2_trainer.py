"""Trainer name and interface for the DCNv2 CTR model."""

from __future__ import annotations

from .ctr_trainer_base import CTRTrainerBase


class DCNv2Trainer(CTRTrainerBase):
    """Train and evaluate :class:`DCNv2`."""
