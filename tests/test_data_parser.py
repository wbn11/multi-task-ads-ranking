from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.inspect import inspect_dataset
from src.data.parser import (
    AliCCPFormatError,
    parse_feature_blob,
    parse_sample_skeleton_line,
)


def feature_blob(*triplets: tuple[str, str, str]) -> str:
    return "\x01".join(
        f"{field_id}\x02{feature_id}\x03{value}"
        for field_id, feature_id, value in triplets
    )


class ParserTest(unittest.TestCase):
    def test_feature_parser_preserves_repeated_anonymous_fields(self) -> None:
        features = parse_feature_blob(
            feature_blob(("field_1", "feature_a", "1"), ("field_1", "feature_b", "2")),
            expected_count=2,
        )
        self.assertEqual([feature.field_id for feature in features], ["field_1", "field_1"])
        self.assertEqual([feature.feature_id for feature in features], ["feature_a", "feature_b"])

    def test_declared_feature_count_is_validated(self) -> None:
        line = f"sample_1,1,0,common_1,2,{feature_blob(('205', 'item_1', '1'))}\n"
        with self.assertRaises(AliCCPFormatError):
            parse_sample_skeleton_line(line)

    def test_inspection_joins_and_drops_invalid_label_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            common_path = root / "common_features_train.csv"
            skeleton_path = root / "sample_skeleton_train.csv"
            output_path = root / "debug.jsonl"
            report_path = root / "stats.json"

            common_path.write_text(
                "\n".join(
                    (
                        f"common_1,2,{feature_blob(('101', 'user_1', '1'), ('121', 'profile_1', '1'))}",
                        f"common_2,1,{feature_blob(('101', 'user_2', '1'))}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            skeleton_path.write_text(
                "\n".join(
                    (
                        f"sample_1,1,1,common_1,1,{feature_blob(('205', 'item_1', '1'))}",
                        f"sample_2,0,0,common_1,1,{feature_blob(('205', 'item_2', '1'))}",
                        f"sample_3,0,1,common_2,1,{feature_blob(('205', 'item_3', '1'))}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            report = inspect_dataset(
                skeleton_path,
                common_path,
                max_samples=10_000,
                output_path=output_path,
                report_path=report_path,
            )

            self.assertEqual(report["labels"]["rows_read"], 3)
            self.assertEqual(report["labels"]["samples_emitted"], 2)
            self.assertEqual(report["labels"]["invalid_click_0_conversion_1"], 1)
            self.assertEqual(report["labels"]["ctr"], 0.5)
            self.assertEqual(report["labels"]["cvr_on_clicked_samples"], 1.0)
            self.assertEqual(report["labels"]["ctcvr"], 0.5)
            self.assertEqual(
                report["labels"]["combination_counts_before_filter"],
                {
                    "click_0_conversion_0": 1,
                    "click_0_conversion_1": 1,
                    "click_1_conversion_0": 0,
                    "click_1_conversion_1": 1,
                },
            )
            self.assertTrue(report["validation"]["passed"])
            self.assertTrue(report["join"]["referenced_common_keys_are_unique"])
            self.assertTrue(report["join"]["row_count_preserved_before_label_filter"])
            self.assertEqual(report["join"]["samples_per_common_id"]["max"], 2)
            self.assertEqual(report["features"]["field_count"], 3)
            self.assertEqual(report["features"]["cardinality_by_field"]["205"], 2)
            self.assertTrue(report["validation"]["field_source_valid"])

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["ctcvr"], 1)
            self.assertEqual(rows[0]["dense_features"], {})
            self.assertTrue(report_path.is_file())

    def test_inspection_reports_duplicate_referenced_common_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            common_path = root / "common_features_train.csv"
            skeleton_path = root / "sample_skeleton_train.csv"

            common_path.write_text(
                "\n".join(
                    (
                        f"common_1,1,{feature_blob(('101', 'user_1', '1'))}",
                        f"common_1,1,{feature_blob(('101', 'user_2', '1'))}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            skeleton_path.write_text(
                f"sample_1,1,0,common_1,1,{feature_blob(('205', 'item_1', '1'))}\n",
                encoding="utf-8",
            )

            report = inspect_dataset(
                skeleton_path,
                common_path,
                max_samples=100_000,
            )

            self.assertFalse(report["validation"]["passed"])
            self.assertFalse(report["validation"]["join_valid"])
            self.assertEqual(report["join"]["duplicate_referenced_common_ids"], 1)
            self.assertEqual(
                report["join"]["duplicate_referenced_common_id_examples"],
                ["common_1"],
            )


if __name__ == "__main__":
    unittest.main()
