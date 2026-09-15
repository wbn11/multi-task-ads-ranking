from __future__ import annotations

import unittest

from src.data.sampler import configure_negative_downsampling


class RecordingDataset:
    def __init__(self, *, samples: int, clicks: int) -> None:
        self.metadata = {"counts": {"samples": samples, "clicks": clicks}}
        self.configured: tuple[float, int] | None = None

    def set_negative_sampling(self, *, keep_probability: float, seed: int) -> None:
        self.configured = (keep_probability, seed)


class NegativeDownsamplingTest(unittest.TestCase):
    def test_computes_keep_probability_from_requested_ratio(self) -> None:
        dataset = RecordingDataset(samples=100, clicks=10)
        summary = configure_negative_downsampling(
            dataset,
            negative_to_positive_ratio=5.0,
            seed=2026,
        )

        self.assertEqual(dataset.configured, (50.0 / 90.0, 2026))
        self.assertEqual(summary["expected_samples_per_epoch"], 60)
        self.assertEqual(
            summary["strategy"],
            "uniform_non_click_bernoulli_resampled_each_epoch",
        )

    def test_large_ratio_keeps_every_negative(self) -> None:
        dataset = RecordingDataset(samples=30, clicks=10)
        summary = configure_negative_downsampling(
            dataset,
            negative_to_positive_ratio=10.0,
            seed=1,
        )

        self.assertEqual(summary["negative_keep_probability"], 1.0)
        self.assertEqual(dataset.configured, (1.0, 1))

    def test_invalid_dataset_or_ratio_raises(self) -> None:
        with self.assertRaises(ValueError):
            configure_negative_downsampling(
                RecordingDataset(samples=10, clicks=0),
                negative_to_positive_ratio=1.0,
                seed=1,
            )
        with self.assertRaises(ValueError):
            configure_negative_downsampling(
                RecordingDataset(samples=10, clicks=1),
                negative_to_positive_ratio=0.0,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
