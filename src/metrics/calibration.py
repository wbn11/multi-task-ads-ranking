"""Probability calibration metrics and reliability-curve bins."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _validate_binary_probabilities(
    labels: Iterable[float] | NDArray[np.generic],
    probabilities: Iterable[float] | NDArray[np.generic],
) -> tuple[FloatArray, FloatArray]:
    y_true = np.asarray(labels, dtype=np.float64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional")
    if y_true.size == 0 or y_true.shape != y_prob.shape:
        raise ValueError("labels and probabilities require equal non-empty shapes")
    if not np.isfinite(y_true).all() or not np.isfinite(y_prob).all():
        raise ValueError("labels and probabilities must be finite")
    if not np.isin(y_true, (0.0, 1.0)).all():
        raise ValueError("labels must contain only 0 and 1")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("probabilities must be inside [0, 1]")
    return y_true, y_prob


def binary_brier_score(
    labels: Iterable[float] | NDArray[np.generic],
    probabilities: Iterable[float] | NDArray[np.generic],
) -> float:
    """Return the mean squared error between probabilities and labels."""

    y_true, y_prob = _validate_binary_probabilities(labels, probabilities)
    return float(np.mean(np.square(y_prob - y_true)))


def _summarize_bin(
    *,
    bin_index: int,
    labels: FloatArray,
    probabilities: FloatArray,
    lower_bound: float,
    upper_bound: float,
) -> dict[str, Any]:
    prediction_mean = float(probabilities.mean())
    positive_rate = float(labels.mean())
    return {
        "bin_index": bin_index,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "samples": int(labels.size),
        "positives": int(labels.sum()),
        "prediction_mean": prediction_mean,
        "positive_rate": positive_rate,
        "absolute_gap": abs(prediction_mean - positive_rate),
    }


def build_calibration_bins(
    labels: Iterable[float] | NDArray[np.generic],
    probabilities: Iterable[float] | NDArray[np.generic],
    *,
    num_bins: int = 10,
    strategy: str = "equal_frequency",
) -> list[dict[str, Any]]:
    """Build non-empty reliability bins for a binary probability model."""

    y_true, y_prob = _validate_binary_probabilities(labels, probabilities)
    if num_bins <= 1:
        raise ValueError("num_bins must be greater than one")
    if strategy not in ("equal_frequency", "equal_width"):
        raise ValueError("strategy must be equal_frequency or equal_width")

    bins: list[dict[str, Any]] = []
    if strategy == "equal_width":
        assignments = np.minimum(
            np.floor(y_prob * num_bins).astype(np.int64),
            num_bins - 1,
        )
        for bin_index in range(num_bins):
            mask = assignments == bin_index
            if not mask.any():
                continue
            bins.append(
                _summarize_bin(
                    bin_index=bin_index,
                    labels=y_true[mask],
                    probabilities=y_prob[mask],
                    lower_bound=bin_index / num_bins,
                    upper_bound=(bin_index + 1) / num_bins,
                )
            )
        return bins

    ordered_indices = np.argsort(y_prob, kind="mergesort")
    for bin_index, indices in enumerate(
        np.array_split(ordered_indices, min(num_bins, y_true.size))
    ):
        if indices.size == 0:
            continue
        bin_probabilities = y_prob[indices]
        bins.append(
            _summarize_bin(
                bin_index=bin_index,
                labels=y_true[indices],
                probabilities=bin_probabilities,
                lower_bound=float(bin_probabilities.min()),
                upper_bound=float(bin_probabilities.max()),
            )
        )
    return bins


def compute_calibration_metrics(
    labels: Iterable[float] | NDArray[np.generic],
    probabilities: Iterable[float] | NDArray[np.generic],
    *,
    num_bins: int = 10,
    strategy: str = "equal_frequency",
) -> dict[str, Any]:
    """Compute Brier score, ECE, mean bias and curve-ready bins."""

    y_true, y_prob = _validate_binary_probabilities(labels, probabilities)
    bins = build_calibration_bins(
        y_true,
        y_prob,
        num_bins=num_bins,
        strategy=strategy,
    )
    expected_calibration_error = sum(
        (item["samples"] / y_true.size) * item["absolute_gap"]
        for item in bins
    )
    prediction_mean = float(y_prob.mean())
    positive_rate = float(y_true.mean())
    return {
        "brier_score": binary_brier_score(y_true, y_prob),
        "expected_calibration_error": float(expected_calibration_error),
        "prediction_mean": prediction_mean,
        "positive_rate": positive_rate,
        "mean_bias": prediction_mean - positive_rate,
        "binning_strategy": strategy,
        "requested_bins": int(num_bins),
        "non_empty_bins": len(bins),
        "bins": bins,
    }
