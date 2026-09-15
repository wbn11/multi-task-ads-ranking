"""Exact binary classification metrics used by the CTR baseline."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


NumericArray = NDArray[np.generic]


def _validate_inputs(
    labels: Iterable[float] | NDArray[np.generic],
    probabilities: Iterable[float] | NDArray[np.generic],
) -> tuple[NumericArray, NumericArray]:
    y_true = np.asarray(labels)
    y_prob = np.asarray(probabilities)
    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional")
    if y_true.size == 0:
        raise ValueError("metrics require at least one sample")
    if y_true.shape != y_prob.shape:
        raise ValueError("labels and probabilities must have identical shapes")
    if not np.isfinite(y_true).all() or not np.isfinite(y_prob).all():
        raise ValueError("labels and probabilities must be finite")
    if not np.isin(y_true, (0.0, 1.0)).all():
        raise ValueError("labels must be binary values 0 or 1")
    if ((y_prob < 0.0) | (y_prob > 1.0)).any():
        raise ValueError("probabilities must be inside [0, 1]")
    return y_true, y_prob


def _require_both_classes(y_true: NumericArray) -> tuple[int, int]:
    positives = int(y_true.sum())
    negatives = int(y_true.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC metrics require both positive and negative labels")
    return positives, negatives


def _ranking_metrics_from_validated(
    y_true: NumericArray,
    y_prob: NumericArray,
) -> tuple[float, float]:
    """Compute exact tie-aware ROC-AUC and AP with one vectorized sort."""

    positives, negatives = _require_both_classes(y_true)
    order = np.argsort(y_prob, kind="stable")[::-1]
    sorted_probabilities = y_prob[order]
    sorted_labels = y_true[order]

    group_starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(
                sorted_probabilities[1:] != sorted_probabilities[:-1]
            ).astype(np.int64, copy=False)
            + 1,
        )
    )
    group_ends = np.concatenate(
        (group_starts[1:], np.array([y_true.size], dtype=np.int64))
    )
    group_sizes = group_ends - group_starts
    group_positives = np.add.reduceat(sorted_labels, group_starts).astype(
        np.float64, copy=False
    )
    group_negatives = group_sizes.astype(np.float64) - group_positives

    cumulative_negatives = np.cumsum(group_negatives, dtype=np.float64)
    lower_scored_negatives = float(negatives) - cumulative_negatives
    concordant_pairs = np.sum(
        group_positives
        * (lower_scored_negatives + 0.5 * group_negatives),
        dtype=np.float64,
    )
    auc = concordant_pairs / (float(positives) * float(negatives))

    cumulative_positives = np.cumsum(group_positives, dtype=np.float64)
    precision_at_group_end = cumulative_positives / group_ends
    average_precision = np.sum(
        precision_at_group_end * (group_positives / float(positives)),
        dtype=np.float64,
    )
    return float(auc), float(average_precision)


def binary_auc(
    labels: Iterable[float] | NDArray[np.generic],
    probabilities: Iterable[float] | NDArray[np.generic],
) -> float:
    """Compute exact ROC-AUC with vectorized tie handling."""

    y_true, y_prob = _validate_inputs(labels, probabilities)
    return _ranking_metrics_from_validated(y_true, y_prob)[0]


def binary_log_loss(
    labels: Iterable[float] | NDArray[np.generic],
    probabilities: Iterable[float] | NDArray[np.generic],
    *,
    epsilon: float = 1e-7,
) -> float:
    """Compute mean binary cross entropy after numerical clipping."""

    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between zero and 0.5")
    y_true, y_prob = _validate_inputs(labels, probabilities)
    clipped = np.clip(y_prob, epsilon, 1.0 - epsilon)
    losses = -(
        y_true * np.log(clipped) + (1.0 - y_true) * np.log1p(-clipped)
    )
    return float(losses.mean(dtype=np.float64))


def binary_pr_auc(
    labels: Iterable[float] | NDArray[np.generic],
    probabilities: Iterable[float] | NDArray[np.generic],
) -> float:
    """Compute tie-aware Average Precision with vectorized score groups."""

    y_true, y_prob = _validate_inputs(labels, probabilities)
    return _ranking_metrics_from_validated(y_true, y_prob)[1]


def compute_binary_metrics(
    labels: Iterable[float] | NDArray[np.generic],
    probabilities: Iterable[float] | NDArray[np.generic],
) -> dict[str, float]:
    """Compute the standard CTR evaluation bundle."""

    y_true, y_prob = _validate_inputs(labels, probabilities)
    auc, pr_auc = _ranking_metrics_from_validated(y_true, y_prob)
    return {
        "auc": auc,
        "log_loss": binary_log_loss(y_true, y_prob),
        "pr_auc": pr_auc,
        "prediction_mean": float(y_prob.mean(dtype=np.float64)),
    }
