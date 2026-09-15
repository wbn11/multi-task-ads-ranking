"""Analytical prior correction after uniform negative downsampling."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _validate_keep_probability(negative_keep_probability: float) -> float:
    keep_probability = float(negative_keep_probability)
    if not math.isfinite(keep_probability) or not 0.0 < keep_probability <= 1.0:
        raise ValueError("negative_keep_probability must be inside (0, 1]")
    return keep_probability


def negative_sampling_logit_offset(
    negative_keep_probability: float,
) -> float:
    """Return the constant added to sampled-distribution CTR logits."""

    return math.log(_validate_keep_probability(negative_keep_probability))


def correct_probabilities_for_negative_sampling(
    probabilities: Iterable[float] | NDArray[np.generic],
    *,
    negative_keep_probability: float,
) -> FloatArray:
    """Map sampled-distribution CTR probabilities to the original prior."""

    sampled = np.asarray(probabilities, dtype=np.float64)
    if sampled.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if sampled.size == 0:
        raise ValueError("probabilities cannot be empty")
    if not np.isfinite(sampled).all():
        raise ValueError("probabilities must be finite")
    if np.any((sampled < 0.0) | (sampled > 1.0)):
        raise ValueError("probabilities must be inside [0, 1]")

    alpha = _validate_keep_probability(negative_keep_probability)
    numerator = alpha * sampled
    denominator = (1.0 - sampled) + numerator
    return numerator / denominator
