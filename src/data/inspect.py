"""Build and validate a small joined Ali-CCP sample without loading full data."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .feature_schema import (
    EXPECTED_COMMON_FIELDS,
    EXPECTED_SAMPLE_FIELDS,
    FIELD_SPEC_BY_ID,
)
from .parser import (
    CommonFeatureRecord,
    SampleSkeletonRecord,
    SparseFeature,
    iter_sample_skeleton,
    load_selected_common_features,
    resolve_split_files,
)


def _feature_json(feature: SparseFeature, source: str) -> dict[str, str]:
    return feature.as_dict(source)


def _joined_sample(
    sample: SampleSkeletonRecord, common: CommonFeatureRecord
) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "common_feature_id": sample.common_feature_id,
        "sparse_features": [
            *(_feature_json(feature, "sample") for feature in sample.features),
            *(_feature_json(feature, "common") for feature in common.features),
        ],
        "dense_features": {},
        "click": sample.click,
        "conversion": sample.conversion,
        "ctcvr": sample.click * sample.conversion,
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _nearest_rank_percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[index]


def _association_distribution(group_sizes: Iterable[int]) -> dict[str, float | int | None]:
    sizes = list(group_sizes)
    if not sizes:
        return {
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
            "singleton_groups": 0,
        }
    ordered = sorted(sizes)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median: float = float(ordered[middle])
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "min": ordered[0],
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "p95": _nearest_rank_percentile(ordered, 95),
        "max": ordered[-1],
        "singleton_groups": sum(size == 1 for size in ordered),
    }


def _update_cardinality(
    cardinality: dict[str, dict[str, set[str]]],
    features: Iterable[SparseFeature],
    source: str,
) -> int:
    count = 0
    for feature in features:
        cardinality[source][feature.field_id].add(feature.feature_id)
        count += 1
    return count


def inspect_dataset(
    skeleton_path: str | Path,
    common_path: str | Path,
    *,
    max_samples: int = 100_000,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    skeleton_path = Path(skeleton_path)
    common_path = Path(common_path)
    samples = list(iter_sample_skeleton(skeleton_path, max_rows=max_samples))
    common_group_counts = Counter(sample.common_feature_id for sample in samples)
    required_common_ids = set(common_group_counts)
    common_selection = load_selected_common_features(
        common_path,
        required_common_ids,
        audit_all_rows=True,
    )
    common_records = common_selection.records
    missing_common_ids = sorted(required_common_ids - common_records.keys())
    if missing_common_ids:
        preview = ", ".join(missing_common_ids[:5])
        raise ValueError(
            f"{len(missing_common_ids)} referenced common_feature_id values are missing; "
            f"first IDs: {preview}"
        )

    invalid_samples = [sample for sample in samples if sample.has_invalid_label_chain]
    valid_samples = [sample for sample in samples if not sample.has_invalid_label_chain]
    label_combinations = Counter(
        f"click_{sample.click}_conversion_{sample.conversion}" for sample in samples
    )
    valid_common_group_counts = Counter(
        sample.common_feature_id for sample in valid_samples
    )
    cardinality: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    click_count = sum(sample.click for sample in valid_samples)
    conversion_count = sum(sample.conversion for sample in valid_samples)
    ctcvr_count = sum(
        sample.click * sample.conversion for sample in valid_samples
    )
    sample_feature_occurrences = 0
    for sample in valid_samples:
        sample_feature_occurrences += _update_cardinality(
            cardinality, sample.features, "sample"
        )

    common_unique_feature_occurrences = 0
    for common_id in valid_common_group_counts:
        common_unique_feature_occurrences += _update_cardinality(
            cardinality, common_records[common_id].features, "common"
        )
    common_joined_feature_occurrences = sum(
        group_size * len(common_records[common_id].features)
        for common_id, group_size in valid_common_group_counts.items()
    )

    duplicate_common_ids = common_selection.duplicate_ids
    join_keys_are_unique = not duplicate_common_ids
    if output_path is not None and not join_keys_are_unique:
        raise ValueError(
            "Cannot emit joined samples because referenced common_feature_id "
            "values are duplicated"
        )

    destination = Path(output_path) if output_path is not None else None
    temporary_destination = (
        destination.with_suffix(destination.suffix + ".tmp") if destination else None
    )
    if temporary_destination:
        temporary_destination.parent.mkdir(parents=True, exist_ok=True)
        writer = temporary_destination.open("w", encoding="utf-8", newline="\n")
    else:
        writer = None

    try:
        for sample in valid_samples:
            common = common_records[sample.common_feature_id]
            if writer:
                json.dump(
                    _joined_sample(sample, common),
                    writer,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                writer.write("\n")
    finally:
        if writer:
            writer.close()

    if temporary_destination and destination:
        temporary_destination.replace(destination)

    per_source_cardinality = {
        source: {
            field_id: len(feature_ids)
            for field_id, feature_ids in sorted(fields.items())
        }
        for source, fields in sorted(cardinality.items())
    }
    all_fields: dict[str, set[str]] = defaultdict(set)
    for fields in cardinality.values():
        for field_id, feature_ids in fields.items():
            all_fields[field_id].update(feature_ids)

    observed_sample_fields = set(cardinality["sample"])
    observed_common_fields = set(cardinality["common"])
    unexpected_sample_fields = sorted(observed_sample_fields - EXPECTED_SAMPLE_FIELDS)
    unexpected_common_fields = sorted(observed_common_fields - EXPECTED_COMMON_FIELDS)
    misplaced_user_fields = sorted(observed_sample_fields & EXPECTED_COMMON_FIELDS)
    misplaced_non_user_fields = sorted(observed_common_fields & EXPECTED_SAMPLE_FIELDS)
    overlapping_source_fields = sorted(observed_sample_fields & observed_common_fields)
    field_source_valid = not any(
        (
            unexpected_sample_fields,
            unexpected_common_fields,
            misplaced_user_fields,
            misplaced_non_user_fields,
            overlapping_source_fields,
        )
    )
    join_valid = not missing_common_ids and join_keys_are_unique
    emitted_count = len(valid_samples)
    report: dict[str, Any] = {
        "inputs": {
            "sample_skeleton": str(skeleton_path.resolve()),
            "common_features": str(common_path.resolve()),
            "max_samples": max_samples,
        },
        "validation": {
            "passed": join_valid and field_source_valid,
            "join_valid": join_valid,
            "field_source_valid": field_source_valid,
        },
        "labels": {
            "rows_read": len(samples),
            "samples_emitted": emitted_count,
            "combination_counts_before_filter": {
                key: label_combinations.get(key, 0)
                for key in (
                    "click_0_conversion_0",
                    "click_0_conversion_1",
                    "click_1_conversion_0",
                    "click_1_conversion_1",
                )
            },
            "invalid_click_0_conversion_1": len(invalid_samples),
            "invalid_label_policy": "drop",
            "click_count": click_count,
            "conversion_count": conversion_count,
            "ctcvr_count": ctcvr_count,
            "ctr": _safe_rate(click_count, emitted_count),
            "cvr_on_clicked_samples": _safe_rate(ctcvr_count, click_count),
            "ctcvr": _safe_rate(ctcvr_count, emitted_count),
        },
        "join": {
            "join_type": "sample_skeleton LEFT JOIN common_features",
            "join_key": "common_feature_id",
            "skeleton_rows_before_label_filter": len(samples),
            "referenced_common_ids": len(required_common_ids),
            "matched_common_ids": len(common_records),
            "missing_common_ids": 0,
            "common_file_rows_scanned": common_selection.rows_scanned,
            "duplicate_referenced_common_ids": len(duplicate_common_ids),
            "duplicate_referenced_common_id_examples": list(
                duplicate_common_ids[:20]
            ),
            "referenced_common_keys_are_unique": join_keys_are_unique,
            "row_count_preserved_before_label_filter": join_valid,
            "samples_per_common_id": _association_distribution(
                common_group_counts.values()
            ),
        },
        "features": {
            "schema_note": (
                "Field-level semantics follow the official Ali-CCP schema; concrete "
                "feature_id values remain anonymized. No explicit fixed-width dense "
                "field is present in the published raw schema."
            ),
            "field_count": len(all_fields),
            "observed_sample_fields": sorted(observed_sample_fields),
            "observed_common_fields": sorted(observed_common_fields),
            "expected_sample_fields_not_observed": sorted(
                EXPECTED_SAMPLE_FIELDS - observed_sample_fields
            ),
            "expected_common_fields_not_observed": sorted(
                EXPECTED_COMMON_FIELDS - observed_common_fields
            ),
            "unexpected_sample_fields": unexpected_sample_fields,
            "unexpected_common_fields": unexpected_common_fields,
            "misplaced_user_fields_in_sample": misplaced_user_fields,
            "misplaced_non_user_fields_in_common": misplaced_non_user_fields,
            "fields_present_in_both_sources": overlapping_source_fields,
            "official_schema_for_observed_fields": {
                field_id: FIELD_SPEC_BY_ID[field_id].as_dict()
                for field_id in sorted(all_fields)
                if field_id in FIELD_SPEC_BY_ID
            },
            "feature_occurrences": {
                "sample_rows": sample_feature_occurrences,
                "common_unique_records": common_unique_feature_occurrences,
                "common_after_join": common_joined_feature_occurrences,
            },
            "cardinality_by_field": {
                field_id: len(feature_ids)
                for field_id, feature_ids in sorted(all_fields.items())
            },
            "cardinality_by_source_and_field": per_source_cardinality,
        },
    }

    if report_path is not None:
        report_destination = Path(report_path)
        report_destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = report_destination.with_suffix(
            report_destination.suffix + ".tmp"
        )
        temporary_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_report.replace(report_destination)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join and inspect a small Ali-CCP raw-data sample."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--skeleton", type=Path)
    parser.add_argument("--common", type=Path)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional joined JSONL output. Omit for inspection-only mode; "
            "expanded common features can make this file several GB."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON report path; defaults to data/processed/inspect_<split>_<N>.json",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if (args.skeleton is None) != (args.common is None):
        raise SystemExit("--skeleton and --common must be provided together")
    if args.skeleton is None:
        skeleton_path, common_path = resolve_split_files(args.raw_dir, args.split)
    else:
        skeleton_path, common_path = args.skeleton, args.common

    report_path = args.report or Path(
        f"data/processed/inspect_{args.split}_{args.max_samples}.json"
    )

    report = inspect_dataset(
        skeleton_path,
        common_path,
        max_samples=args.max_samples,
        output_path=args.output,
        report_path=report_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["validation"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
