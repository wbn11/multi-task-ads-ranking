"""Entire Space Multi-Task Model for CTR, CVR and CTCVR."""

from __future__ import annotations

from src.models.shared_bottom import SharedBottom


class ESMM(SharedBottom):
    """Use Shared Bottom towers with the ESMM entire-space objective.

    The network intentionally matches the Shared Bottom baseline. ESMM's
    defining behavior is supplied by its CTR + CTCVR loss, allowing the
    experiment to isolate the effect of the entire-space objective.
    """
