"""Non-parametric monotonic probability calibration using PAVA."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .common import FloatArray, as_probability_array, validate_calibration_data


class IsotonicCalibrator:
    """Learn a non-decreasing mapping with the pool-adjacent-violators algorithm."""

    def __init__(self) -> None:
        self.x_thresholds_: FloatArray | None = None
        self.y_thresholds_: FloatArray | None = None
        self.num_blocks_: int | None = None
        self.training_samples_: int | None = None

    def fit(
        self,
        probabilities: Iterable[float] | NDArray[np.generic],
        labels: Iterable[float] | NDArray[np.generic],
    ) -> IsotonicCalibrator:
        probability_array, label_array = validate_calibration_data(
            probabilities,
            labels,
        )
        order = np.argsort(probability_array, kind="mergesort")
        sorted_probabilities = probability_array[order]
        sorted_labels = label_array[order]
        unique_probabilities, first_indices, counts = np.unique(
            sorted_probabilities,
            return_index=True,
            return_counts=True,
        )
        positive_sums = np.add.reduceat(sorted_labels, first_indices)

        block_lowers: list[float] = []
        block_uppers: list[float] = []
        block_weights: list[float] = []
        block_positive_sums: list[float] = []
        for probability, count, positive_sum in zip(
            unique_probabilities,
            counts,
            positive_sums,
            strict=True,
        ):
            block_lowers.append(float(probability))
            block_uppers.append(float(probability))
            block_weights.append(float(count))
            block_positive_sums.append(float(positive_sum))
            while len(block_weights) >= 2:
                previous_mean = (
                    block_positive_sums[-2] / block_weights[-2]
                )
                current_mean = block_positive_sums[-1] / block_weights[-1]
                if previous_mean <= current_mean:
                    break
                block_uppers[-2] = block_uppers[-1]
                block_weights[-2] += block_weights[-1]
                block_positive_sums[-2] += block_positive_sums[-1]
                block_lowers.pop()
                block_uppers.pop()
                block_weights.pop()
                block_positive_sums.pop()

        threshold_x: list[float] = []
        threshold_y: list[float] = []
        for lower, upper, weight, positive_sum in zip(
            block_lowers,
            block_uppers,
            block_weights,
            block_positive_sums,
            strict=True,
        ):
            fitted_probability = positive_sum / weight
            threshold_x.append(lower)
            threshold_y.append(fitted_probability)
            if upper > lower:
                threshold_x.append(upper)
                threshold_y.append(fitted_probability)

        self.x_thresholds_ = np.asarray(threshold_x, dtype=np.float64)
        self.y_thresholds_ = np.asarray(threshold_y, dtype=np.float64)
        self.num_blocks_ = len(block_weights)
        self.training_samples_ = int(label_array.size)
        return self

    def transform(
        self,
        probabilities: Iterable[float] | NDArray[np.generic],
    ) -> FloatArray:
        if self.x_thresholds_ is None or self.y_thresholds_ is None:
            raise RuntimeError("IsotonicCalibrator must be fitted before transform")
        values = as_probability_array(probabilities)
        return np.interp(
            values,
            self.x_thresholds_,
            self.y_thresholds_,
            left=float(self.y_thresholds_[0]),
            right=float(self.y_thresholds_[-1]),
        )

    def state_dict(self) -> dict[str, Any]:
        if (
            self.x_thresholds_ is None
            or self.y_thresholds_ is None
            or self.num_blocks_ is None
            or self.training_samples_ is None
        ):
            raise RuntimeError(
                "IsotonicCalibrator must be fitted before serialization"
            )
        return {
            "type": "isotonic",
            "algorithm": "pool_adjacent_violators",
            "interpolation": "linear_between_pooled_blocks",
            "x_thresholds": self.x_thresholds_.tolist(),
            "y_thresholds": self.y_thresholds_.tolist(),
            "num_blocks": self.num_blocks_,
            "num_thresholds": int(self.x_thresholds_.size),
            "training_samples": self.training_samples_,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> IsotonicCalibrator:
        if state.get("type") != "isotonic":
            raise ValueError("state is not an Isotonic calibrator")
        x_thresholds = np.asarray(state["x_thresholds"], dtype=np.float64)
        y_thresholds = np.asarray(state["y_thresholds"], dtype=np.float64)
        if (
            x_thresholds.ndim != 1
            or x_thresholds.size == 0
            or x_thresholds.shape != y_thresholds.shape
        ):
            raise ValueError("serialized isotonic thresholds are invalid")
        if not np.isfinite(x_thresholds).all() or not np.isfinite(
            y_thresholds
        ).all():
            raise ValueError("serialized isotonic thresholds must be finite")
        if np.any(np.diff(x_thresholds) <= 0.0):
            raise ValueError("serialized isotonic x thresholds must increase")
        if np.any(np.diff(y_thresholds) < 0.0) or np.any(
            (y_thresholds < 0.0) | (y_thresholds > 1.0)
        ):
            raise ValueError(
                "serialized isotonic y thresholds must be monotonic probabilities"
            )
        calibrator = cls()
        calibrator.x_thresholds_ = x_thresholds
        calibrator.y_thresholds_ = y_thresholds
        calibrator.num_blocks_ = int(state["num_blocks"])
        calibrator.training_samples_ = int(state["training_samples"])
        return calibrator
