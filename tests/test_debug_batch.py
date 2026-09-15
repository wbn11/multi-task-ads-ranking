from __future__ import annotations

import unittest

from src.data.debug_batch import select_funnel_overfit_samples


class FunnelOverfitBatchTest(unittest.TestCase):
    def test_selects_requested_valid_funnel_states_deterministically(self) -> None:
        dataset = [
            {"id": 0, "click": 0, "conversion": 0},
            {"id": 1, "click": 1, "conversion": 0},
            {"id": 2, "click": 1, "conversion": 1},
            {"id": 3, "click": 0, "conversion": 0},
            {"id": 4, "click": 1, "conversion": 0},
            {"id": 5, "click": 0, "conversion": 0},
        ]

        selected = select_funnel_overfit_samples(
            dataset,
            batch_size=5,
            conversion_positives=1,
            clicked_non_conversions=2,
            seed=2026,
        )
        repeated = select_funnel_overfit_samples(
            dataset,
            batch_size=5,
            conversion_positives=1,
            clicked_non_conversions=2,
            seed=2026,
        )

        self.assertEqual(selected, repeated)
        self.assertEqual(sum(row["conversion"] for row in selected), 1)
        self.assertEqual(sum(row["click"] for row in selected), 3)
        self.assertEqual(len(selected), 5)

    def test_invalid_funnel_state_raises(self) -> None:
        dataset = [{"click": 0, "conversion": 1}]
        with self.assertRaises(RuntimeError):
            select_funnel_overfit_samples(
                dataset,
                batch_size=3,
                conversion_positives=1,
                clicked_non_conversions=1,
                seed=2026,
            )


if __name__ == "__main__":
    unittest.main()
