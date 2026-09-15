from __future__ import annotations

import math
import unittest

import numpy as np

from src.metrics.gate import compute_gate_metrics


class GateMetricsTest(unittest.TestCase):
    def test_reports_per_sample_entropy_and_top1_utilization(self) -> None:
        weights = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
            ]
        )

        metrics = compute_gate_metrics(weights)

        expected_entropy = float(
            np.mean(-np.sum(weights * np.log(weights), axis=1)) / math.log(3)
        )
        np.testing.assert_allclose(metrics["mean_weights"], [0.4, 0.5, 0.1])
        np.testing.assert_allclose(metrics["top1_fractions"], [0.5, 0.5, 0.0])
        self.assertAlmostEqual(metrics["mean_max_weight"], 0.75)
        self.assertAlmostEqual(
            metrics["mean_normalized_entropy"],
            expected_entropy,
        )
        self.assertAlmostEqual(metrics["row_sum_max_error"], 0.0)

    def test_uniform_and_collapsed_gate_entropy_limits(self) -> None:
        uniform = compute_gate_metrics(np.full((4, 4), 0.25))
        collapsed = compute_gate_metrics(
            np.array([[1.0, 0.0, 0.0, 0.0]] * 4)
        )

        self.assertAlmostEqual(uniform["mean_normalized_entropy"], 1.0)
        self.assertAlmostEqual(collapsed["mean_normalized_entropy"], 0.0)
        self.assertEqual(collapsed["top1_fractions"], [1.0, 0.0, 0.0, 0.0])

    def test_invalid_gate_weights_raise(self) -> None:
        with self.assertRaises(ValueError):
            compute_gate_metrics([0.5, 0.5])
        with self.assertRaises(ValueError):
            compute_gate_metrics([[0.4, 0.4]])
        with self.assertRaises(ValueError):
            compute_gate_metrics([[1.1, -0.1]])


if __name__ == "__main__":
    unittest.main()
