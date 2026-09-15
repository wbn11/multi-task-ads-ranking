from __future__ import annotations

import unittest

from src.metrics.gauc import compute_gauc


class GAUCTest(unittest.TestCase):
    def test_impression_weighted_gauc_and_coverage(self) -> None:
        report = compute_gauc(
            labels=[0, 1, 0, 0, 1, 1, 0, 0, 0, 1],
            probabilities=[0.1, 0.9, 0.9, 0.8, 0.2, 0.1, 0.2, 0.3, 0.4, 0.7],
            user_ids=["u1", "u1", "u2", "u2", "u2", "u2", "u3", "u3", None, ""],
        )

        # u1 has AUC 1 over 2 impressions; u2 has AUC 0 over 4.
        self.assertAlmostEqual(report["value"], 1.0 / 3.0)
        self.assertEqual(report["eligible_users"], 2)
        self.assertEqual(report["eligible_impressions"], 6)
        self.assertEqual(report["skipped_all_negative_users"], 1)
        self.assertEqual(report["missing_user_samples"], 2)
        self.assertAlmostEqual(report["eligible_impression_fraction"], 0.6)

    def test_uniform_user_weighting(self) -> None:
        report = compute_gauc(
            labels=[0, 1, 0, 0, 1, 1],
            probabilities=[0.1, 0.9, 0.9, 0.8, 0.2, 0.1],
            user_ids=["u1", "u1", "u2", "u2", "u2", "u2"],
            weighting="uniform_users",
        )
        self.assertAlmostEqual(report["value"], 0.5)

    def test_fixed_width_user_codes_match_string_grouping(self) -> None:
        labels = [0, 1, 0, 0, 1, 1]
        probabilities = [0.1, 0.9, 0.9, 0.8, 0.2, 0.1]
        string_report = compute_gauc(
            labels, probabilities, ["u1", "u1", "u2", "u2", "u2", "u2"]
        )
        integer_report = compute_gauc(
            labels, probabilities, [11, 11, 22, 22, 22, 22]
        )

        self.assertEqual(integer_report, string_report)

    def test_returns_none_when_no_user_has_both_classes(self) -> None:
        report = compute_gauc(
            labels=[0, 0, 1, 1],
            probabilities=[0.1, 0.2, 0.8, 0.9],
            user_ids=["negative", "negative", "positive", "positive"],
        )
        self.assertIsNone(report["value"])
        self.assertEqual(report["eligible_users"], 0)
        self.assertEqual(report["skipped_single_class_users"], 2)

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            compute_gauc([0, 1], [0.2], ["u", "u"])
        with self.assertRaises(ValueError):
            compute_gauc([0, 1], [0.2, 0.8], ["u"])
        with self.assertRaises(ValueError):
            compute_gauc([0, 1], [0.2, 1.2], ["u", "u"])
        with self.assertRaises(ValueError):
            compute_gauc(
                [0, 1],
                [0.2, 0.8],
                ["u", "u"],
                weighting="unknown",
            )


if __name__ == "__main__":
    unittest.main()
