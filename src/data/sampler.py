"""Sampling configuration for the streaming Ali-CCP training dataset."""

from __future__ import annotations

import math
from typing import Any, Mapping


def configure_negative_downsampling(
    dataset: Any,
    *,
    negative_to_positive_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Configure epoch-resampled uniform Bernoulli non-click sampling."""

    if (
        not math.isfinite(negative_to_positive_ratio)
        or negative_to_positive_ratio <= 0.0
    ):
        raise ValueError("negative_to_positive_ratio must be positive")
    if seed < 0:
        raise ValueError("seed cannot be negative")
    metadata = getattr(dataset, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("dataset must expose processed metadata")
    counts = metadata.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("dataset metadata must contain counts")

    sample_count = int(counts["samples"])
    positive_count = int(counts["clicks"])
    negative_count = sample_count - positive_count
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("negative downsampling requires both click classes")
    requested_negative_count = int(
        round(negative_to_positive_ratio * positive_count)
    )
    expected_negative_count = min(negative_count, requested_negative_count)
    keep_probability = expected_negative_count / float(negative_count)
    dataset.set_negative_sampling(keep_probability=keep_probability, seed=seed)

    return {
        "enabled": True,
        "strategy": "uniform_non_click_bernoulli_resampled_each_epoch",
        "seed": int(seed),
        "requested_negative_to_positive_ratio": float(
            negative_to_positive_ratio
        ),
        "original_samples": sample_count,
        "original_positive_samples": positive_count,
        "original_negative_samples": negative_count,
        "expected_samples_per_epoch": positive_count + expected_negative_count,
        "expected_positive_samples_per_epoch": positive_count,
        "expected_negative_samples_per_epoch": expected_negative_count,
        "expected_negative_to_positive_ratio": (
            expected_negative_count / float(positive_count)
        ),
        "negative_keep_probability": keep_probability,
    }
