"""End-to-end validation for compact, encoded Ali-CCP datasets."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset import AdsDataset
from .feature_encoder import FeatureEncoder, UNK_INDEX
from .feature_schema import ALL_FIELD_IDS
from .preprocess import _load_yaml


def validate_processed_dataset(
    processed_dir: str | Path,
    *,
    split: str,
    max_error_examples: int = 10,
) -> dict[str, Any]:
    """Read every model sample and verify labels, joins, and encoded features."""

    if max_error_examples <= 0:
        raise ValueError("max_error_examples must be positive")

    root = Path(processed_dir)
    encoder = FeatureEncoder.load(root / "vocab.json")
    dataset = AdsDataset(root, split=split)
    allowed_fields = set(ALL_FIELD_IDS)
    fields_seen: Counter[str] = Counter()
    error_count = 0
    error_examples: list[str] = []
    clicks = 0
    conversions = 0
    ctcvr_positives = 0
    samples_with_user_id = 0
    samples_missing_user_id = 0
    samples_seen = 0

    def record_error(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(error_examples) < max_error_examples:
            error_examples.append(message)

    try:
        for index, sample in enumerate(dataset):
            samples_seen += 1
            click = sample["click"]
            conversion = sample["conversion"]
            ctcvr = sample["ctcvr"]
            user_id = sample["user_id"]
            if user_id is None:
                samples_missing_user_id += 1
            elif isinstance(user_id, str) and user_id:
                samples_with_user_id += 1
            else:
                record_error(f"sample {index}: user_id must be a string or null")
            if click not in (0, 1) or conversion not in (0, 1):
                record_error(f"sample {index}: labels must be binary")
            if conversion == 1 and click == 0:
                record_error(f"sample {index}: conversion=1 while click=0")
            if ctcvr != click * conversion:
                record_error(f"sample {index}: ctcvr does not equal click*conversion")
            clicks += click
            conversions += conversion
            ctcvr_positives += ctcvr

            features = sample["features"]
            unexpected_fields = set(features) - allowed_fields
            if unexpected_fields:
                record_error(
                    f"sample {index}: unexpected fields {sorted(unexpected_fields)}"
                )
            for field_id, field in features.items():
                fields_seen[field_id] += 1
                if field_id not in allowed_fields:
                    continue
                ids = field.get("ids", [])
                values = field.get("values", [])
                if len(ids) != len(values) or not ids:
                    record_error(
                        f"sample {index}, field {field_id}: invalid ids/values lengths"
                    )
                    continue
                vocabulary = encoder.field_vocabularies[field_id]
                for encoded_id, value in zip(ids, values):
                    if not UNK_INDEX <= int(encoded_id) < vocabulary.vocab_size:
                        record_error(
                            f"sample {index}, field {field_id}: encoded ID "
                            f"{encoded_id} outside [1, {vocabulary.vocab_size})"
                        )
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        record_error(
                            f"sample {index}, field {field_id}: non-numeric value {value!r}"
                        )
                    else:
                        if not math.isfinite(numeric_value):
                            record_error(
                                f"sample {index}, field {field_id}: non-finite value"
                            )
    finally:
        dataset.close()

    expected = dataset.metadata["counts"]
    actual_counts = {
        "samples": samples_seen,
        "clicks": clicks,
        "conversions": conversions,
        "ctcvr_positives": ctcvr_positives,
        "samples_with_user_id": samples_with_user_id,
        "samples_missing_user_id": samples_missing_user_id,
    }
    metadata_consistent = (
        actual_counts["samples"] == expected["samples"]
        and actual_counts["clicks"] == expected["clicks"]
        and actual_counts["ctcvr_positives"] == expected["ctcvr_positives"]
        and (
            "samples_with_user_id" not in dataset.metadata["user_id"]
            or actual_counts["samples_with_user_id"]
            == dataset.metadata["user_id"]["samples_with_user_id"]
        )
        and (
            "samples_missing_user_id" not in dataset.metadata["user_id"]
            or actual_counts["samples_missing_user_id"]
            == dataset.metadata["user_id"]["samples_missing_user_id"]
        )
    )
    if not metadata_consistent:
        record_error("recomputed counts do not match metadata.json")

    return {
        "split": split,
        "valid": error_count == 0,
        "counts": actual_counts,
        "metadata_consistent": metadata_consistent,
        "schema": {
            "declared_field_count": len(ALL_FIELD_IDS),
            "observed_field_count": len(fields_seen),
            "fields_seen_in_samples": dict(fields_seen),
        },
        "errors": {
            "count": error_count,
            "examples": error_examples,
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a processed Ali-CCP dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    config = _load_yaml(args.config)
    data_config = config.get("data")
    if not isinstance(data_config, dict):
        raise SystemExit("config must contain a data mapping")
    processed_dir = args.processed_dir or Path(data_config["processed_dir"])
    report = validate_processed_dataset(processed_dir, split=args.split)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
