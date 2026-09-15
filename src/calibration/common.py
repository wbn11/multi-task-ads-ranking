"""Shared validation and numerical helpers for probability calibrators."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def as_probability_array(
    probabilities: Iterable[float] | NDArray[np.generic],
) -> FloatArray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be inside [0, 1]")
    return values


def validate_calibration_data(
    probabilities: Iterable[float] | NDArray[np.generic],
    labels: Iterable[float] | NDArray[np.generic],
) -> tuple[FloatArray, FloatArray]:
    probability_array = as_probability_array(probabilities)
    label_array = np.asarray(labels, dtype=np.float64)
    if label_array.ndim != 1 or label_array.shape != probability_array.shape:
        raise ValueError("labels and probabilities must have equal 1D shapes")
    if not np.isfinite(label_array).all():
        raise ValueError("labels must be finite")
    if not np.isin(label_array, (0.0, 1.0)).all():
        raise ValueError("labels must contain only 0 and 1")
    if np.unique(label_array).size != 2:
        raise ValueError("calibrator fitting requires both label classes")
    return probability_array, label_array


def probability_to_logit(
    probabilities: Iterable[float] | NDArray[np.generic],
    *,
    epsilon: float,
) -> FloatArray:
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between zero and 0.5")
    values = as_probability_array(probabilities)
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def stable_sigmoid(logits: FloatArray) -> FloatArray:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponentials = np.exp(values[~positive])
    output[~positive] = exponentials / (1.0 + exponentials)
    return output
