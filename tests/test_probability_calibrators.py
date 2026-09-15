from __future__ import annotations

import unittest

import numpy as np

from src.calibration.isotonic import IsotonicCalibrator
from src.calibration.platt import PlattCalibrator
from src.metrics.binary import binary_auc, binary_log_loss
from src.metrics.calibration import binary_brier_score


class PlattCalibratorTest(unittest.TestCase):
    def test_reduces_log_loss_for_overconfident_probabilities(self) -> None:
        probabilities = np.asarray([0.01] * 50 + [0.99] * 50)
        labels = np.asarray([0] * 40 + [1] * 10 + [0] * 10 + [1] * 40)

        calibrator = PlattCalibrator().fit(probabilities, labels)
        calibrated = calibrator.transform(probabilities)

        self.assertLess(
            binary_log_loss(labels, calibrated),
            binary_log_loss(labels, probabilities),
        )
        self.assertAlmostEqual(
            binary_auc(labels, calibrated),
            binary_auc(labels, probabilities),
        )
        self.assertGreater(calibrator.slope_, 0.0)
        self.assertTrue(np.all(np.diff(calibrated) >= 0.0))
        self.assertAlmostEqual(float(calibrated[:50].mean()), 0.2, places=3)
        self.assertAlmostEqual(float(calibrated[50:].mean()), 0.8, places=3)

    def test_state_round_trip_preserves_predictions(self) -> None:
        probabilities = np.linspace(0.01, 0.99, 100)
        labels = (probabilities > 0.6).astype(np.float64)
        fitted = PlattCalibrator().fit(probabilities, labels)

        restored = PlattCalibrator.from_state_dict(fitted.state_dict())

        np.testing.assert_allclose(
            fitted.transform(probabilities),
            restored.transform(probabilities),
        )

    def test_rejects_invalid_fit_and_unfitted_transform(self) -> None:
        with self.assertRaises(ValueError):
            PlattCalibrator().fit([0.1, 0.2], [0, 0])
        with self.assertRaises(RuntimeError):
            PlattCalibrator().transform([0.5])


class IsotonicCalibratorTest(unittest.TestCase):
    def test_pava_pools_adjacent_violations(self) -> None:
        probabilities = np.asarray([0.1, 0.2, 0.3, 0.4])
        labels = np.asarray([0, 1, 0, 1])

        calibrator = IsotonicCalibrator().fit(probabilities, labels)
        calibrated = calibrator.transform(probabilities)

        np.testing.assert_allclose(calibrated, [0.0, 0.5, 0.5, 1.0])
        self.assertTrue(np.all(np.diff(calibrated) >= 0.0))
        self.assertLessEqual(
            binary_brier_score(labels, calibrated),
            binary_brier_score(labels, probabilities),
        )

    def test_clips_outside_training_range_and_round_trips(self) -> None:
        fitted = IsotonicCalibrator().fit(
            [0.2, 0.4, 0.6, 0.8],
            [0, 0, 1, 1],
        )
        restored = IsotonicCalibrator.from_state_dict(fitted.state_dict())

        expected = fitted.transform([0.0, 0.3, 0.7, 1.0])
        actual = restored.transform([0.0, 0.3, 0.7, 1.0])

        np.testing.assert_allclose(actual, expected)
        self.assertEqual(actual[0], fitted.y_thresholds_[0])
        self.assertEqual(actual[-1], fitted.y_thresholds_[-1])

    def test_rejects_invalid_fit_and_unfitted_transform(self) -> None:
        with self.assertRaises(ValueError):
            IsotonicCalibrator().fit([0.1, 0.2], [1, 1])
        with self.assertRaises(RuntimeError):
            IsotonicCalibrator().transform([0.5])


if __name__ == "__main__":
    unittest.main()
