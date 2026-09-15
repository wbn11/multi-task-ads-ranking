"""Two-parameter sigmoid calibration fitted with damped Newton updates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .common import (
    FloatArray,
    probability_to_logit,
    stable_sigmoid,
    validate_calibration_data,
)


class PlattCalibrator:
    """Fit a monotonic ``sigmoid(slope * logit(p) + intercept)`` map."""

    def __init__(
        self,
        *,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        l2_regularization: float = 1e-6,
        epsilon: float = 1e-7,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")
        if not math.isfinite(l2_regularization) or l2_regularization < 0.0:
            raise ValueError("l2_regularization must be finite and non-negative")
        if not 0.0 < epsilon < 0.5:
            raise ValueError("epsilon must be between zero and 0.5")
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.l2_regularization = float(l2_regularization)
        self.epsilon = float(epsilon)
        self.slope_: float | None = None
        self.intercept_: float | None = None
        self.iterations_: int | None = None
        self.converged_: bool | None = None
        self.training_samples_: int | None = None
        self.training_objective_: float | None = None

    def _objective(
        self,
        logits: FloatArray,
        labels: FloatArray,
        slope: float,
        intercept: float,
    ) -> float:
        calibrated_logits = slope * logits + intercept
        data_loss = np.mean(
            np.logaddexp(0.0, calibrated_logits)
            - labels * calibrated_logits
        )
        penalty = 0.5 * self.l2_regularization * slope * slope
        return float(data_loss + penalty)

    def fit(
        self,
        probabilities: Iterable[float] | NDArray[np.generic],
        labels: Iterable[float] | NDArray[np.generic],
    ) -> PlattCalibrator:
        probability_array, label_array = validate_calibration_data(
            probabilities,
            labels,
        )
        logits = probability_to_logit(
            probability_array,
            epsilon=self.epsilon,
        )
        slope = 1.0
        intercept = 0.0
        converged = False
        iterations = 0

        for iterations in range(1, self.max_iterations + 1):
            calibrated_logits = slope * logits + intercept
            calibrated = stable_sigmoid(calibrated_logits)
            residual = calibrated - label_array
            weights = np.maximum(calibrated * (1.0 - calibrated), 1e-12)
            gradient = np.asarray(
                [
                    np.mean(residual * logits)
                    + self.l2_regularization * slope,
                    np.mean(residual),
                ],
                dtype=np.float64,
            )
            if float(np.max(np.abs(gradient))) <= self.tolerance:
                converged = True
                break

            hessian = np.asarray(
                [
                    [
                        np.mean(weights * logits * logits)
                        + self.l2_regularization,
                        np.mean(weights * logits),
                    ],
                    [np.mean(weights * logits), np.mean(weights)],
                ],
                dtype=np.float64,
            )
            hessian.flat[::3] += 1e-12
            try:
                newton_step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError as error:
                raise RuntimeError("Platt optimization Hessian is singular") from error

            objective = self._objective(logits, label_array, slope, intercept)
            directional_decrease = float(np.dot(gradient, newton_step))
            step_size = 1.0
            accepted = False
            while step_size >= 1e-8:
                candidate_slope = slope - step_size * float(newton_step[0])
                candidate_intercept = (
                    intercept - step_size * float(newton_step[1])
                )
                if candidate_slope <= 0.0:
                    step_size *= 0.5
                    continue
                candidate_objective = self._objective(
                    logits,
                    label_array,
                    candidate_slope,
                    candidate_intercept,
                )
                if candidate_objective <= (
                    objective - 1e-4 * step_size * directional_decrease
                ):
                    slope = candidate_slope
                    intercept = candidate_intercept
                    accepted = True
                    break
                step_size *= 0.5

            if not accepted:
                break
            if float(np.max(np.abs(step_size * newton_step))) <= self.tolerance:
                converged = True
                break

        self.slope_ = float(slope)
        self.intercept_ = float(intercept)
        self.iterations_ = int(iterations)
        self.converged_ = bool(converged)
        self.training_samples_ = int(label_array.size)
        self.training_objective_ = self._objective(
            logits,
            label_array,
            slope,
            intercept,
        )
        return self

    def transform(
        self,
        probabilities: Iterable[float] | NDArray[np.generic],
    ) -> FloatArray:
        if self.slope_ is None or self.intercept_ is None:
            raise RuntimeError("PlattCalibrator must be fitted before transform")
        logits = probability_to_logit(probabilities, epsilon=self.epsilon)
        return stable_sigmoid(self.slope_ * logits + self.intercept_)

    def state_dict(self) -> dict[str, Any]:
        if (
            self.slope_ is None
            or self.intercept_ is None
            or self.iterations_ is None
            or self.converged_ is None
            or self.training_samples_ is None
            or self.training_objective_ is None
        ):
            raise RuntimeError("PlattCalibrator must be fitted before serialization")
        return {
            "type": "platt",
            "formula": "sigmoid(slope * logit(probability) + intercept)",
            "slope": self.slope_,
            "intercept": self.intercept_,
            "epsilon": self.epsilon,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "l2_regularization": self.l2_regularization,
            "iterations": self.iterations_,
            "converged": self.converged_,
            "training_samples": self.training_samples_,
            "training_objective": self.training_objective_,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> PlattCalibrator:
        if state.get("type") != "platt":
            raise ValueError("state is not a Platt calibrator")
        calibrator = cls(
            max_iterations=int(state["max_iterations"]),
            tolerance=float(state["tolerance"]),
            l2_regularization=float(state["l2_regularization"]),
            epsilon=float(state["epsilon"]),
        )
        slope = float(state["slope"])
        intercept = float(state["intercept"])
        if not math.isfinite(slope) or slope <= 0.0:
            raise ValueError("serialized Platt slope must be positive and finite")
        if not math.isfinite(intercept):
            raise ValueError("serialized Platt intercept must be finite")
        calibrator.slope_ = slope
        calibrator.intercept_ = intercept
        calibrator.iterations_ = int(state["iterations"])
        calibrator.converged_ = bool(state["converged"])
        calibrator.training_samples_ = int(state["training_samples"])
        calibrator.training_objective_ = float(state["training_objective"])
        return calibrator
