"""Diagnostics for task-specific mixture-of-experts gate weights."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray


def compute_gate_metrics(
    gate_weights: Iterable[Iterable[float]] | NDArray[np.generic],
    *,
    row_sum_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Summarize per-example gate concentration and expert utilization."""

    weights = np.asarray(gate_weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("gate_weights must have shape [samples, experts]")
    sample_count, num_experts = weights.shape
    if sample_count == 0 or num_experts <= 1:
        raise ValueError("gate_weights require samples and at least two experts")
    if row_sum_tolerance <= 0.0:
        raise ValueError("row_sum_tolerance must be positive")
    if not np.isfinite(weights).all():
        raise ValueError("gate_weights must contain only finite values")
    if np.any(weights < 0.0):
        raise ValueError("gate_weights cannot be negative")

    row_sum_max_error = float(np.max(np.abs(weights.sum(axis=1) - 1.0)))
    if row_sum_max_error > row_sum_tolerance:
        raise ValueError(
            "gate weight rows must sum to one within tolerance; "
            f"max_error={row_sum_max_error}"
        )

    safe_weights = np.clip(weights, np.finfo(np.float64).tiny, 1.0)
    per_sample_entropy = -np.sum(weights * np.log(safe_weights), axis=1)
    normalized_entropy = per_sample_entropy / math.log(num_experts)
    top1_experts = np.argmax(weights, axis=1)
    top1_counts = np.bincount(top1_experts, minlength=num_experts)
    return {
        "samples": float(sample_count),
        "num_experts": int(num_experts),
        "mean_weights": [
            float(value) for value in weights.mean(axis=0).tolist()
        ],
        "mean_normalized_entropy": float(normalized_entropy.mean()),
        "mean_max_weight": float(weights.max(axis=1).mean()),
        "top1_fractions": [
            float(value) for value in (top1_counts / sample_count).tolist()
        ],
        "row_sum_max_error": row_sum_max_error,
    }
