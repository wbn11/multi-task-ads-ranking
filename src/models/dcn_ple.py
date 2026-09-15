"""DCNv2-enhanced PLE model for the architecture ablation."""

from __future__ import annotations

from .dcn_ple_esmm import DCNPLEESMM


class DCNPLE(DCNPLEESMM):
    """Combine DCNv2 feature crosses with PLE task routing.

    The network is intentionally identical to :class:`DCNPLEESMM`.  The
    ablation differs only in its trainer: ``DCNPLE`` uses exposure-space CTR
    BCE plus clicked-space masked CVR BCE, while ``DCNPLEESMM`` uses the
    entire-space CTR + CTCVR objective.  Keeping the architecture fixed makes
    the effect of the ESMM objective independently measurable.
    """
