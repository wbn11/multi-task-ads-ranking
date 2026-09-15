"""Deterministic real-data batches used by multi-task overfit checks."""

from __future__ import annotations

import random
from typing import Any, Iterator, Protocol


class SampleDataset(Protocol):
    def __iter__(self) -> Iterator[dict[str, Any]]: ...


def select_funnel_overfit_samples(
    dataset: SampleDataset,
    *,
    batch_size: int,
    conversion_positives: int,
    clicked_non_conversions: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Collect a fixed batch with all valid click/conversion funnel states."""

    non_click_target = batch_size - conversion_positives - clicked_non_conversions
    if min(conversion_positives, clicked_non_conversions, non_click_target) <= 0:
        raise ValueError(
            "batch quotas must leave at least one conversion, clicked "
            "non-conversion and non-click"
        )

    buckets: dict[str, list[dict[str, Any]]] = {
        "conversion": [],
        "clicked_non_conversion": [],
        "non_click": [],
    }
    targets = {
        "conversion": conversion_positives,
        "clicked_non_conversion": clicked_non_conversions,
        "non_click": non_click_target,
    }
    for sample in dataset:
        click = int(sample["click"])
        conversion = int(sample["conversion"])
        if conversion == 1 and click != 1:
            raise RuntimeError("found conversion=1 with click=0 in processed data")
        if conversion == 1:
            bucket_name = "conversion"
        elif click == 1:
            bucket_name = "clicked_non_conversion"
        else:
            bucket_name = "non_click"
        if len(buckets[bucket_name]) < targets[bucket_name]:
            buckets[bucket_name].append(sample)
        if all(len(buckets[name]) == target for name, target in targets.items()):
            break

    if any(len(buckets[name]) != target for name, target in targets.items()):
        raise RuntimeError(
            "could not build the requested fixed batch: "
            f"conversion_positives={len(buckets['conversion'])}/"
            f"{targets['conversion']}, clicked_non_conversions="
            f"{len(buckets['clicked_non_conversion'])}/"
            f"{targets['clicked_non_conversion']}, non_clicks="
            f"{len(buckets['non_click'])}/{targets['non_click']}"
        )

    selected = [
        sample
        for bucket in buckets.values()
        for sample in bucket
    ]
    random.Random(seed).shuffle(selected)
    return selected
