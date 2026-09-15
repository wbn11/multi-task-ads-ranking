from __future__ import annotations

import unittest

import numpy as np

from src.metrics.multitask import compute_multitask_metrics


class MultiTaskMetricsTest(unittest.TestCase):
    def test_each_task_uses_its_correct_sample_space(self) -> None:
        click = np.array([0, 1, 1, 0], dtype=np.float64)
        conversion = np.array([0, 0, 1, 0], dtype=np.float64)
        ctcvr = click * conversion
        ctr = np.array([0.1, 0.9, 0.8, 0.2])
        cvr = np.array([0.9, 0.1, 0.9, 0.8])
        predicted_ctcvr = ctr * cvr

        metrics = compute_multitask_metrics(
            click_labels=click,
            conversion_labels=conversion,
            ctcvr_labels=ctcvr,
            ctr_probabilities=ctr,
            cvr_probabilities=cvr,
            ctcvr_probabilities=predicted_ctcvr,
            user_ids=["user"] * 4,
        )

        self.assertEqual(metrics["ctr"]["samples"], 4.0)
        self.assertEqual(metrics["cvr"]["samples"], 2.0)
        self.assertEqual(metrics["ctcvr"]["samples"], 4.0)
        self.assertEqual(metrics["ctr"]["auc"], 1.0)
        self.assertEqual(metrics["cvr"]["auc"], 1.0)
        self.assertEqual(metrics["ctcvr"]["auc"], 1.0)
        self.assertEqual(metrics["ctr"]["gauc"], 1.0)
        self.assertEqual(metrics["cvr"]["gauc"], 1.0)
        self.assertEqual(metrics["ctcvr"]["gauc"], 1.0)

    def test_invalid_funnel_labels_raise(self) -> None:
        with self.assertRaises(ValueError):
            compute_multitask_metrics(
                click_labels=[0, 1],
                conversion_labels=[1, 0],
                ctcvr_labels=[0, 0],
                ctr_probabilities=[0.2, 0.8],
                cvr_probabilities=[0.2, 0.8],
                ctcvr_probabilities=[0.04, 0.64],
                user_ids=["user", "user"],
            )


if __name__ == "__main__":
    unittest.main()
