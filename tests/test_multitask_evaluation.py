from __future__ import annotations

import importlib.util
import unittest

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    from src.trainer.multitask_evaluation import (
        correct_multitask_sampling_prior,
        evaluate_multitask_probability_bundle,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class MultiTaskEvaluationTest(unittest.TestCase):
    def test_prior_correction_changes_ctr_and_preserves_product(self) -> None:
        raw = {
            "ctr": np.array([0.2, 0.8]),
            "cvr": np.array([0.3, 0.4]),
            "ctcvr": np.array([0.06, 0.32]),
        }
        corrected = correct_multitask_sampling_prior(
            raw,
            negative_keep_probability=0.5,
        )

        self.assertTrue(np.all(corrected["ctr"] < raw["ctr"]))
        np.testing.assert_allclose(corrected["cvr"], raw["cvr"])
        np.testing.assert_allclose(
            corrected["ctcvr"],
            corrected["ctr"] * corrected["cvr"],
        )

    def test_evaluation_uses_clicked_space_for_cvr(self) -> None:
        arrays = {
            "click": np.array([0, 1, 1, 0], dtype=np.uint8),
            "conversion": np.array([0, 0, 1, 0], dtype=np.uint8),
            "ctcvr": np.array([0, 0, 1, 0], dtype=np.uint8),
            "user": np.array([1, 1, 1, 1], dtype=np.int64),
        }
        probabilities = {
            "ctr": np.array([0.1, 0.8, 0.9, 0.2]),
            "cvr": np.array([0.9, 0.2, 0.8, 0.9]),
            "ctcvr": np.array([0.09, 0.16, 0.72, 0.18]),
        }

        metrics = evaluate_multitask_probability_bundle(
            arrays,
            probabilities,
            calibration_num_bins=2,
            calibration_binning_strategy="equal_frequency",
        )

        self.assertEqual(metrics["ctr"]["samples"], 4.0)
        self.assertEqual(metrics["cvr"]["samples"], 2.0)
        self.assertEqual(metrics["ctcvr"]["samples"], 4.0)
        self.assertEqual(metrics["cvr"]["calibration"]["requested_bins"], 2)


if __name__ == "__main__":
    unittest.main()
