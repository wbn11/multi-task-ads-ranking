from __future__ import annotations

import unittest

import numpy as np

from src.metrics.calibration import (
    binary_brier_score,
    build_calibration_bins,
    compute_calibration_metrics,
)


class CalibrationMetricsTest(unittest.TestCase):
    def test_groupwise_calibrated_probabilities_have_zero_ece(self) -> None:
        labels = np.asarray(
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
            dtype=np.float64,
        )
        probabilities = np.asarray([0.2] * 5 + [0.8] * 5)

        metrics = compute_calibration_metrics(
            labels,
            probabilities,
            num_bins=5,
            strategy="equal_width",
        )

        self.assertAlmostEqual(metrics["expected_calibration_error"], 0.0)
        self.assertAlmostEqual(metrics["brier_score"], 0.16)
        self.assertAlmostEqual(metrics["mean_bias"], 0.0)
        self.assertEqual(metrics["non_empty_bins"], 2)

    def test_severely_reversed_probabilities_have_large_ece(self) -> None:
        labels = np.asarray([0] * 5 + [1] * 5)
        probabilities = np.asarray([0.9] * 5 + [0.1] * 5)

        metrics = compute_calibration_metrics(
            labels,
            probabilities,
            num_bins=2,
            strategy="equal_frequency",
        )

        self.assertAlmostEqual(metrics["expected_calibration_error"], 0.9)
        self.assertAlmostEqual(metrics["brier_score"], 0.81)

    def test_equal_frequency_bins_cover_every_sample(self) -> None:
        labels = np.asarray([0, 1] * 5)
        probabilities = np.linspace(0.05, 0.95, 10)

        bins = build_calibration_bins(
            labels,
            probabilities,
            num_bins=3,
            strategy="equal_frequency",
        )

        self.assertEqual([item["samples"] for item in bins], [4, 3, 3])
        self.assertEqual(sum(item["samples"] for item in bins), 10)

    def test_equal_width_assigns_probability_one_to_last_bin(self) -> None:
        bins = build_calibration_bins(
            [0, 1],
            [0.0, 1.0],
            num_bins=2,
            strategy="equal_width",
        )

        self.assertEqual([item["bin_index"] for item in bins], [0, 1])
        self.assertEqual([item["samples"] for item in bins], [1, 1])

    def test_brier_score_matches_manual_mean_squared_error(self) -> None:
        score = binary_brier_score([0, 1], [0.25, 0.75])
        self.assertAlmostEqual(score, 0.0625)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            compute_calibration_metrics([], [], num_bins=10)
        with self.assertRaises(ValueError):
            compute_calibration_metrics([0, 1], [0.2], num_bins=10)
        with self.assertRaises(ValueError):
            compute_calibration_metrics([0, 1], [0.2, 1.2], num_bins=10)
        with self.assertRaises(ValueError):
            compute_calibration_metrics([0, 1], [0.2, 0.8], num_bins=1)
        with self.assertRaises(ValueError):
            compute_calibration_metrics(
                [0, 1],
                [0.2, 0.8],
                num_bins=10,
                strategy="unknown",
            )


if __name__ == "__main__":
    unittest.main()
