"""User-grouped AUC for recommendation and advertising ranking."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _normalize_user_id(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    normalized = str(value)
    return normalized if normalized else None


def _as_user_codes(
    user_ids: Iterable[object] | NDArray[np.generic],
) -> NDArray[np.int64]:
    """Use fixed-width IDs directly and compact object IDs without index lists."""

    values = (
        np.asarray(user_ids)
        if isinstance(user_ids, np.ndarray)
        else np.asarray(list(user_ids), dtype=object)
    )
    if values.ndim != 1:
        raise ValueError("user_ids must be one-dimensional")
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64, copy=False)

    mapping: dict[str, int] = {}
    codes = np.full(values.size, -1, dtype=np.int64)
    for index, raw_value in enumerate(values):
        normalized = _normalize_user_id(raw_value)
        if normalized is None:
            continue
        code = mapping.get(normalized)
        if code is None:
            code = len(mapping)
            mapping[normalized] = code
        codes[index] = code
    return codes


def compute_gauc(
    labels: Iterable[float] | NDArray[np.generic],
    probabilities: Iterable[float] | NDArray[np.generic],
    user_ids: Iterable[object] | NDArray[np.generic],
    *,
    weighting: str = "impressions",
) -> dict[str, Any]:
    """Compute per-user AUC and aggregate it with auditable coverage stats."""

    y_true = np.asarray(labels)
    y_prob = np.asarray(probabilities)
    users = _as_user_codes(user_ids)
    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional")
    if y_true.size == 0 or y_true.shape != y_prob.shape:
        raise ValueError("labels and probabilities require equal non-empty shapes")
    if users.shape != y_true.shape:
        raise ValueError("user_ids must have the same shape as labels")
    if not np.isfinite(y_true).all() or not np.isfinite(y_prob).all():
        raise ValueError("labels and probabilities must be finite")
    if not np.isin(y_true, (0.0, 1.0)).all():
        raise ValueError("labels must contain only 0 and 1")
    if np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("probabilities must be inside [0, 1]")
    if weighting not in ("impressions", "uniform_users"):
        raise ValueError("weighting must be impressions or uniform_users")

    known_mask = users >= 0
    missing_user_samples = int((~known_mask).sum())
    known_user_samples = int(known_mask.sum())
    if known_user_samples == 0:
        return {
            "value": None,
            "weighting": weighting,
            "total_samples": int(y_true.size),
            "known_user_samples": 0,
            "missing_user_samples": missing_user_samples,
            "known_user_sample_fraction": 0.0,
            "total_users": 0,
            "eligible_users": 0,
            "skipped_single_class_users": 0,
            "skipped_all_negative_users": 0,
            "skipped_all_positive_users": 0,
            "eligible_impressions": 0,
            "eligible_impression_fraction": 0.0,
        }

    known_users = users[known_mask]
    known_labels = y_true[known_mask]
    known_probabilities = y_prob[known_mask]
    order = np.lexsort((known_probabilities, known_users))
    sorted_users = known_users[order]
    sorted_probabilities = known_probabilities[order]
    sorted_labels = known_labels[order]

    score_group_starts = np.flatnonzero(
        np.concatenate(
            (
                np.array([True]),
                (sorted_users[1:] != sorted_users[:-1])
                | (sorted_probabilities[1:] != sorted_probabilities[:-1]),
            )
        )
    ).astype(np.int64, copy=False)
    score_group_ends = np.concatenate(
        (
            score_group_starts[1:],
            np.array([known_user_samples], dtype=np.int64),
        )
    )
    score_group_sizes = score_group_ends - score_group_starts
    score_group_positives = np.add.reduceat(
        sorted_labels, score_group_starts
    ).astype(np.float64, copy=False)
    score_group_negatives = (
        score_group_sizes.astype(np.float64) - score_group_positives
    )
    score_group_users = sorted_users[score_group_starts]

    user_group_starts = np.flatnonzero(
        np.concatenate(
            (
                np.array([True]),
                score_group_users[1:] != score_group_users[:-1],
            )
        )
    ).astype(np.int64, copy=False)
    user_group_ends = np.concatenate(
        (
            user_group_starts[1:],
            np.array([score_group_starts.size], dtype=np.int64),
        )
    )
    score_groups_per_user = user_group_ends - user_group_starts
    user_impressions = np.add.reduceat(score_group_sizes, user_group_starts)
    user_positives = np.add.reduceat(score_group_positives, user_group_starts)
    user_negatives = user_impressions.astype(np.float64) - user_positives

    global_negatives_before = (
        np.cumsum(score_group_negatives, dtype=np.float64)
        - score_group_negatives
    )
    user_negative_baselines = np.repeat(
        global_negatives_before[user_group_starts], score_groups_per_user
    )
    negatives_before_within_user = (
        global_negatives_before - user_negative_baselines
    )
    score_group_concordant_pairs = score_group_positives * (
        negatives_before_within_user + 0.5 * score_group_negatives
    )
    user_concordant_pairs = np.add.reduceat(
        score_group_concordant_pairs, user_group_starts
    )

    all_negative = user_positives == 0.0
    all_positive = user_negatives == 0.0
    eligible = ~(all_negative | all_positive)
    eligible_users = int(eligible.sum())
    skipped_all_negative_users = int(all_negative.sum())
    skipped_all_positive_users = int(all_positive.sum())
    eligible_impressions = int(user_impressions[eligible].sum())
    user_auc = user_concordant_pairs[eligible] / (
        user_positives[eligible] * user_negatives[eligible]
    )
    weights = (
        user_impressions[eligible].astype(np.float64)
        if weighting == "impressions"
        else np.ones(eligible_users, dtype=np.float64)
    )
    total_weight = float(weights.sum())
    weighted_auc_sum = float(np.sum(user_auc * weights, dtype=np.float64))

    skipped_single_class_users = (
        skipped_all_negative_users + skipped_all_positive_users
    )
    return {
        "value": (
            float(weighted_auc_sum / total_weight)
            if total_weight > 0.0
            else None
        ),
        "weighting": weighting,
        "total_samples": int(y_true.size),
        "known_user_samples": known_user_samples,
        "missing_user_samples": missing_user_samples,
        "known_user_sample_fraction": known_user_samples / y_true.size,
        "total_users": int(user_group_starts.size),
        "eligible_users": eligible_users,
        "skipped_single_class_users": skipped_single_class_users,
        "skipped_all_negative_users": skipped_all_negative_users,
        "skipped_all_positive_users": skipped_all_positive_users,
        "eligible_impressions": eligible_impressions,
        "eligible_impression_fraction": eligible_impressions / y_true.size,
    }
