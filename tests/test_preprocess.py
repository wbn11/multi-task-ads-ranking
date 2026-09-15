from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from src.data.dataloader import create_dataloader
from src.data.dataset import AdsDataset
from src.data.feature_encoder import UNK_INDEX
from src.data.feature_schema import ALL_FIELD_IDS
from src.data.preprocess import (
    DATASET_FORMAT_VERSION,
    build_processed_datasets,
    stable_evaluation_split,
)
from src.data.smoke_test import run_dataloader_smoke_test
from src.data.validate import validate_processed_dataset


PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


def feature_blob(*triplets: tuple[str, str, str]) -> str:
    return "\x01".join(
        f"{field_id}\x02{feature_id}\x03{value}"
        for field_id, feature_id, value in triplets
    )


def write_fixture(root: Path) -> None:
    train_common = [
        f"c1,1,{feature_blob(('101', 'user_a', '1'))}",
        f"c2,1,{feature_blob(('101', 'user_b', '1'))}",
    ]
    train_samples = [
        f"1,1,1,c1,1,{feature_blob(('205', 'item_a', '1'))}",
        f"2,0,0,c1,1,{feature_blob(('205', 'item_a', '1'))}",
        f"3,0,1,c2,1,{feature_blob(('205', 'item_b', '1'))}",
        f"4,0,0,c2,1,{feature_blob(('205', 'item_b', '1'))}",
    ]
    test_common = [
        f"tc1,1,{feature_blob(('101', 'user_new', '1'))}",
    ]
    test_samples = [
        f"test_{index},0,0,tc1,1,{feature_blob(('205', 'item_new', '1'))}"
        for index in range(30)
    ]
    for name, lines in (
        ("common_features_train.csv", train_common),
        ("sample_skeleton_train.csv", train_samples),
        ("common_features_test.csv", test_common),
        ("sample_skeleton_test.csv", test_samples),
    ):
        (root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


@unittest.skipUnless(PYARROW_AVAILABLE, "PyArrow is not installed")
class PreprocessTest(unittest.TestCase):
    def assert_batches_equal(self, expected, actual) -> None:
        self.assertEqual(expected["sample_id"], actual["sample_id"])
        self.assertEqual(
            expected["common_feature_id"], actual["common_feature_id"]
        )
        for name in ("user_group_id", "click", "conversion", "ctcvr"):
            self.assertTrue(
                expected[name].equal(actual[name]),
                f"batch tensor differs: {name}",
            )
        self.assertEqual(set(expected["features"]), set(ALL_FIELD_IDS))
        self.assertEqual(set(actual["features"]), set(ALL_FIELD_IDS))
        for field_id in ALL_FIELD_IDS:
            for name in ("ids", "values", "offsets"):
                self.assertTrue(
                    expected["features"][field_id][name].equal(
                        actual["features"][field_id][name]
                    ),
                    f"field {field_id} differs: {name}",
                )

    def test_streaming_build_split_join_and_train_only_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir = root / "raw"
            output_dir = root / "processed"
            raw_dir.mkdir()
            write_fixture(raw_dir)

            manifest = build_processed_datasets(
                raw_dir=raw_dir,
                output_dir=output_dir,
                min_frequency=2,
                shard_size=2,
                validation_fraction=0.5,
                split_seed=2026,
                evaluation_start_row=0,
                progress_every=0,
            )
            self.assertEqual(
                manifest["format_version"], DATASET_FORMAT_VERSION
            )
            self.assertEqual(manifest["splits"]["train"]["samples"], 3)
            self.assertEqual(
                manifest["splits"]["validation"]["samples"]
                + manifest["splits"]["test"]["samples"],
                30,
            )

            train = AdsDataset(output_dir, split="train")
            validation = AdsDataset(output_dir, split="validation")
            test = AdsDataset(output_dir, split="test")
            try:
                train_rows = list(train)
                validation_rows = list(validation)
                test_rows = list(test)
            finally:
                train.close()
                validation.close()
                test.close()

            self.assertEqual(len(train_rows), 3)
            self.assertEqual(train_rows[0]["user_id"], "user_a")
            self.assertEqual(train_rows[0]["ctcvr"], 1)
            validation_ids = {row["sample_id"] for row in validation_rows}
            test_ids = {row["sample_id"] for row in test_rows}
            self.assertFalse(validation_ids & test_ids)
            self.assertEqual(len(validation_ids | test_ids), 30)
            for row in validation_rows + test_rows:
                self.assertEqual(row["features"]["101"]["ids"], [UNK_INDEX])
                self.assertEqual(row["features"]["205"]["ids"], [UNK_INDEX])

            for split in ("train", "validation", "test"):
                report = validate_processed_dataset(output_dir, split=split)
                self.assertTrue(report["valid"], report)

            train_for_loader = AdsDataset(output_dir, split="train")
            try:
                loader = create_dataloader(
                    train_for_loader,
                    batch_size=2,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                )
                batch = next(iter(loader))
                self.assertEqual(tuple(batch["click"].shape), (2,))
                self.assertEqual(tuple(batch["user_group_id"].shape), (2,))
                self.assertEqual(
                    tuple(batch["features"]["101"]["offsets"].shape), (3,)
                )
            finally:
                train_for_loader.close()

            # The fast Arrow path must preserve the legacy collator exactly,
            # including a batch that crosses the two-row shard boundary.
            legacy_dataset = AdsDataset(
                output_dir, split="train", seed=2026
            )
            columnar_dataset = AdsDataset(
                output_dir, split="train", seed=2026
            )
            try:
                legacy_batches = list(
                    create_dataloader(
                        legacy_dataset,
                        batch_size=3,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False,
                        columnar_batching=False,
                    )
                )
                columnar_loader = create_dataloader(
                    columnar_dataset,
                    batch_size=3,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    columnar_batching=True,
                )
                columnar_batches = list(columnar_loader)
                self.assertEqual(
                    getattr(columnar_loader.dataset, "batching_mode", None),
                    "arrow_columnar",
                )
            finally:
                legacy_dataset.close()
                columnar_dataset.close()
            self.assertEqual(len(legacy_batches), len(columnar_batches))
            for expected, actual in zip(legacy_batches, columnar_batches):
                self.assert_batches_equal(expected, actual)

            # Vectorized Bernoulli draws must select the same negatives as the
            # row-wise implementation for a fixed epoch and seed.
            legacy_sampled = AdsDataset(
                output_dir, split="train", seed=2026
            )
            columnar_sampled = AdsDataset(
                output_dir, split="train", seed=2026
            )
            for dataset in (legacy_sampled, columnar_sampled):
                dataset.set_negative_sampling(
                    keep_probability=0.5, seed=2026
                )
            try:
                legacy_sampled_batches = list(
                    create_dataloader(
                        legacy_sampled,
                        batch_size=2,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False,
                        columnar_batching=False,
                    )
                )
                columnar_sampled_batches = list(
                    create_dataloader(
                        columnar_sampled,
                        batch_size=2,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False,
                        columnar_batching=True,
                    )
                )
            finally:
                legacy_sampled.close()
                columnar_sampled.close()
            self.assertEqual(
                len(legacy_sampled_batches), len(columnar_sampled_batches)
            )
            for expected, actual in zip(
                legacy_sampled_batches, columnar_sampled_batches
            ):
                self.assert_batches_equal(expected, actual)

            smoke_report = run_dataloader_smoke_test(
                processed_dir=output_dir,
                split="train",
                batch_size=1,
                num_workers=0,
                pin_memory=False,
                device="cpu",
                embedding_dim=2,
                warmup_batches=0,
                benchmark_batches=2,
            )
            self.assertTrue(smoke_report["valid"], smoke_report)
            self.assertEqual(
                smoke_report["throughput_benchmark"][
                    "completed_measured_batches"
                ],
                2,
            )
            self.assertEqual(
                smoke_report["throughput_benchmark"]["measured_samples"],
                2,
            )

            sampled_train = AdsDataset(output_dir, split="train", seed=2026)
            sampled_train.set_negative_sampling(
                keep_probability=1e-9, seed=2026
            )
            try:
                sampled_rows = list(sampled_train)
            finally:
                sampled_train.close()
            self.assertEqual(
                [row["sample_id"] for row in sampled_rows], ["1"]
            )

    def test_hash_split_is_stable(self) -> None:
        first = stable_evaluation_split(
            "sample-42", validation_fraction=0.1, seed=2026
        )
        second = stable_evaluation_split(
            "sample-42", validation_fraction=0.1, seed=2026
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
