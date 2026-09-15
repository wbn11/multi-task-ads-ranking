from __future__ import annotations

import math
import unittest

import numpy as np

from src.calibration.prior import (
    correct_probabilities_for_negative_sampling,
    negative_sampling_logit_offset,
)
from src.metrics.binary import binary_auc


class NegativeSamplingPriorCorrectionTest(unittest.TestCase):
    def test_recovers_original_class_prior(self) -> None:
        original_positive_count = 100
        original_negative_count = 9_900
        alpha = 0.1
        sampled_probability = original_positive_count / (
            original_positive_count + alpha * original_negative_count
        )

        corrected = correct_probabilities_for_negative_sampling(
            [sampled_probability],
            negative_keep_probability=alpha,
        )

        self.assertAlmostEqual(corrected[0], 0.01)

    def test_matches_logit_offset_equation(self) -> None:
        sampled = np.asarray([0.1, 0.5, 0.9])
        alpha = 0.25
        corrected = correct_probabilities_for_negative_sampling(
            sampled,
            negative_keep_probability=alpha,
        )
        sampled_logits = np.log(sampled / (1.0 - sampled))
        expected = 1.0 / (
            1.0
            + np.exp(
                -(sampled_logits + negative_sampling_logit_offset(alpha))
            )
        )

        np.testing.assert_allclose(corrected, expected)

    def test_preserves_ranking_and_auc(self) -> None:
        labels = np.asarray([0, 1, 0, 1])
        sampled = np.asarray([0.2, 0.8, 0.3, 0.9])
        corrected = correct_probabilities_for_negative_sampling(
            sampled,
            negative_keep_probability=0.2,
        )

        self.assertEqual(
            np.argsort(sampled).tolist(),
            np.argsort(corrected).tolist(),
        )
        self.assertAlmostEqual(
            binary_auc(labels, sampled),
            binary_auc(labels, corrected),
        )

    def test_keep_probability_one_is_identity(self) -> None:
        probabilities = np.asarray([0.0, 0.25, 1.0])
        corrected = correct_probabilities_for_negative_sampling(
            probabilities,
            negative_keep_probability=1.0,
        )
        np.testing.assert_array_equal(corrected, probabilities)
        self.assertEqual(negative_sampling_logit_offset(1.0), 0.0)

    def test_rejects_invalid_inputs(self) -> None:
        for alpha in (0.0, -0.1, 1.1, math.inf):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    correct_probabilities_for_negative_sampling(
                        [0.5],
                        negative_keep_probability=alpha,
                    )
        with self.assertRaises(ValueError):
            correct_probabilities_for_negative_sampling(
                [1.2],
                negative_keep_probability=0.5,
            )


if __name__ == "__main__":
    unittest.main()
