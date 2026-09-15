from __future__ import annotations

import math
import unittest

import numpy as np

from src.metrics.binary import (
    binary_auc,
    binary_log_loss,
    binary_pr_auc,
    compute_binary_metrics,
)


class BinaryMetricsTest(unittest.TestCase):
    def test_vectorized_ranking_metrics_match_pairwise_auc(self) -> None:
        labels = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
        probabilities = np.array([0.2, 0.8, 0.5, 0.5, 0.9, 0.2], dtype=np.float32)
        positive_scores = probabilities[labels == 1]
        negative_scores = probabilities[labels == 0]
        expected_auc = np.mean(
            (positive_scores[:, None] > negative_scores[None, :])
            + 0.5 * (positive_scores[:, None] == negative_scores[None, :])
        )

        self.assertAlmostEqual(binary_auc(labels, probabilities), expected_auc)

    def test_perfect_ranking(self) -> None:
        labels = np.array([0, 1, 0, 1])
        probabilities = np.array([0.1, 0.8, 0.2, 0.9])
        metrics = compute_binary_metrics(labels, probabilities)

        self.assertAlmostEqual(metrics["auc"], 1.0)
        self.assertAlmostEqual(metrics["pr_auc"], 1.0)
        expected_log_loss = -sum(
            math.log(value) for value in (0.9, 0.8, 0.8, 0.9)
        ) / 4
        self.assertAlmostEqual(metrics["log_loss"], expected_log_loss)

    def test_tied_predictions_are_order_independent(self) -> None:
        labels = np.array([0, 1])
        probabilities = np.array([0.5, 0.5])

        self.assertAlmostEqual(binary_auc(labels, probabilities), 0.5)
        self.assertAlmostEqual(binary_pr_auc(labels, probabilities), 0.5)
        self.assertAlmostEqual(binary_log_loss(labels, probabilities), math.log(2))

    def test_reversed_ranking_has_zero_auc(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(binary_auc(labels, probabilities), 0.0)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            binary_auc([1, 1], [0.5, 0.6])
        with self.assertRaises(ValueError):
            binary_log_loss([0, 1], [0.2, 1.2])
        with self.assertRaises(ValueError):
            binary_pr_auc([0, 1], [0.5])


if __name__ == "__main__":
    unittest.main()
