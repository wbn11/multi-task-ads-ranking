"""CTR, clicked-space CVR and exposure-space CTCVR evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .binary import compute_binary_metrics
from .gauc import compute_gauc


NumericArray = NDArray[np.generic]


def _as_vector(
    name: str,
    values: Iterable[float] | NDArray[np.generic],
) -> NumericArray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _metric_bundle(
    labels: NumericArray,
    probabilities: NumericArray,
    user_ids: NDArray[np.generic],
) -> dict[str, Any]:
    metrics = compute_binary_metrics(labels, probabilities)
    gauc = compute_gauc(labels, probabilities, user_ids)
    metrics.update(
        {
            "samples": float(labels.size),
            "positives": float(labels.sum()),
            "positive_rate": float(labels.mean()),
            "gauc": gauc["value"],
            "gauc_details": {
                key: value for key, value in gauc.items() if key != "value"
            },
        }
    )
    return metrics


def compute_multitask_metrics(
    *,
    click_labels: Iterable[float] | NDArray[np.generic],
    conversion_labels: Iterable[float] | NDArray[np.generic],
    ctcvr_labels: Iterable[float] | NDArray[np.generic],
    ctr_probabilities: Iterable[float] | NDArray[np.generic],
    cvr_probabilities: Iterable[float] | NDArray[np.generic],
    ctcvr_probabilities: Iterable[float] | NDArray[np.generic],
    user_ids: Iterable[object] | NDArray[np.object_],
) -> dict[str, Any]:
    """Compute metrics on the correct sample space for each funnel task."""

    arrays = {
        "click_labels": _as_vector("click_labels", click_labels),
        "conversion_labels": _as_vector("conversion_labels", conversion_labels),
        "ctcvr_labels": _as_vector("ctcvr_labels", ctcvr_labels),
        "ctr_probabilities": _as_vector("ctr_probabilities", ctr_probabilities),
        "cvr_probabilities": _as_vector("cvr_probabilities", cvr_probabilities),
        "ctcvr_probabilities": _as_vector(
            "ctcvr_probabilities", ctcvr_probabilities
        ),
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("all labels and probabilities must have identical shapes")
    if isinstance(user_ids, np.ndarray):
        user_array = np.asarray(user_ids)
    else:
        user_array = np.asarray(list(user_ids), dtype=object)
    if user_array.ndim != 1 or user_array.shape != arrays["click_labels"].shape:
        raise ValueError("user_ids must have the same 1D shape as labels")

    click = arrays["click_labels"]
    conversion = arrays["conversion_labels"]
    ctcvr = arrays["ctcvr_labels"]
    if not np.isin(click, (0.0, 1.0)).all():
        raise ValueError("click labels must be binary")
    if not np.isin(conversion, (0.0, 1.0)).all():
        raise ValueError("conversion labels must be binary")
    if not np.isin(ctcvr, (0.0, 1.0)).all():
        raise ValueError("ctcvr labels must be binary")
    if np.any(conversion > click):
        raise ValueError("conversion=1 requires click=1")
    if not np.array_equal(ctcvr, click * conversion):
        raise ValueError("ctcvr labels must equal click * conversion")

    clicked_mask = click == 1.0
    if not clicked_mask.any():
        raise ValueError("CVR metrics require at least one clicked sample")
    return {
        "ctr": _metric_bundle(click, arrays["ctr_probabilities"], user_array),
        "cvr": _metric_bundle(
            conversion[clicked_mask],
            arrays["cvr_probabilities"][clicked_mask],
            user_array[clicked_mask],
        ),
        "ctcvr": _metric_bundle(
            ctcvr,
            arrays["ctcvr_probabilities"],
            user_array,
        ),
    }
