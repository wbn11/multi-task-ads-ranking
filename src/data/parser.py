"""Low-level parsers for the raw Ali-CCP text format.

This module only decodes and validates the physical file format. Field-domain
semantics belong to the inspection/feature-schema layer, not the line parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


FEATURE_SEPARATOR = "\x01"
FIELD_FEATURE_SEPARATOR = "\x02"
FEATURE_VALUE_SEPARATOR = "\x03"


class AliCCPFormatError(ValueError):
    """Raised when a raw Ali-CCP row does not match the expected structure."""


@dataclass(frozen=True, slots=True)
class SparseFeature:
    """One anonymous weighted categorical feature from a raw feature blob."""

    field_id: str
    feature_id: str
    value: str

    def as_dict(self, source: str) -> dict[str, str]:
        return {
            "source": source,
            "field_id": self.field_id,
            "feature_id": self.feature_id,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class SampleSkeletonRecord:
    sample_id: str
    click: int
    conversion: int
    common_feature_id: str
    features: tuple[SparseFeature, ...]

    @property
    def has_invalid_label_chain(self) -> bool:
        return self.click == 0 and self.conversion == 1


@dataclass(frozen=True, slots=True)
class CommonFeatureRecord:
    common_feature_id: str
    features: tuple[SparseFeature, ...]


@dataclass(frozen=True, slots=True)
class CommonFeatureSelection:
    """Selected common rows plus association-audit metadata."""

    records: dict[str, CommonFeatureRecord]
    occurrence_counts: dict[str, int]
    rows_scanned: int

    @property
    def duplicate_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                common_id
                for common_id, count in self.occurrence_counts.items()
                if count > 1
            )
        )


def _parse_binary_label(raw_value: str, *, name: str, context: str) -> int:
    if raw_value not in {"0", "1"}:
        raise AliCCPFormatError(
            f"{context}: {name} must be 0 or 1, got {raw_value!r}"
        )
    return int(raw_value)


def parse_feature_blob(
    blob: str,
    *,
    expected_count: int | None = None,
    context: str = "feature blob",
) -> tuple[SparseFeature, ...]:
    """Parse ``field\x02feature\x03value`` triplets separated by ``\x01``."""

    if not blob:
        features: tuple[SparseFeature, ...] = ()
    else:
        parsed: list[SparseFeature] = []
        for index, token in enumerate(blob.split(FEATURE_SEPARATOR), start=1):
            field_parts = token.split(FIELD_FEATURE_SEPARATOR)
            if len(field_parts) != 2:
                raise AliCCPFormatError(
                    f"{context}: feature {index} must contain exactly one "
                    f"field/feature separator"
                )
            field_id, encoded_feature = field_parts
            value_parts = encoded_feature.split(FEATURE_VALUE_SEPARATOR)
            if len(value_parts) != 2:
                raise AliCCPFormatError(
                    f"{context}: feature {index} must contain exactly one "
                    f"feature/value separator"
                )
            feature_id, value = value_parts
            if not field_id or not feature_id or not value:
                raise AliCCPFormatError(
                    f"{context}: feature {index} contains an empty component"
                )
            parsed.append(SparseFeature(field_id, feature_id, value))
        features = tuple(parsed)

    if expected_count is not None and len(features) != expected_count:
        raise AliCCPFormatError(
            f"{context}: declared {expected_count} features but parsed "
            f"{len(features)}"
        )
    return features


def parse_sample_skeleton_line(
    line: str, *, line_number: int | None = None
) -> SampleSkeletonRecord:
    context = f"sample_skeleton line {line_number}" if line_number else "sample_skeleton"
    columns = line.rstrip("\r\n").split(",", maxsplit=5)
    if len(columns) != 6:
        raise AliCCPFormatError(
            f"{context}: expected 6 comma-separated columns, got {len(columns)}"
        )

    sample_id, click_raw, conversion_raw, common_id, count_raw, feature_blob = columns
    try:
        declared_count = int(count_raw)
    except ValueError as error:
        raise AliCCPFormatError(
            f"{context}: feature count must be an integer, got {count_raw!r}"
        ) from error
    if declared_count < 0:
        raise AliCCPFormatError(f"{context}: feature count cannot be negative")
    if not sample_id or not common_id:
        raise AliCCPFormatError(f"{context}: sample/common feature ID cannot be empty")

    return SampleSkeletonRecord(
        sample_id=sample_id,
        click=_parse_binary_label(click_raw, name="click", context=context),
        conversion=_parse_binary_label(
            conversion_raw, name="conversion", context=context
        ),
        common_feature_id=common_id,
        features=parse_feature_blob(
            feature_blob, expected_count=declared_count, context=context
        ),
    )


def parse_common_feature_line(
    line: str, *, line_number: int | None = None
) -> CommonFeatureRecord:
    context = f"common_features line {line_number}" if line_number else "common_features"
    columns = line.rstrip("\r\n").split(",", maxsplit=2)
    if len(columns) != 3:
        raise AliCCPFormatError(
            f"{context}: expected 3 comma-separated columns, got {len(columns)}"
        )

    common_id, count_raw, feature_blob = columns
    try:
        declared_count = int(count_raw)
    except ValueError as error:
        raise AliCCPFormatError(
            f"{context}: feature count must be an integer, got {count_raw!r}"
        ) from error
    if declared_count < 0:
        raise AliCCPFormatError(f"{context}: feature count cannot be negative")
    if not common_id:
        raise AliCCPFormatError(f"{context}: common feature ID cannot be empty")

    return CommonFeatureRecord(
        common_feature_id=common_id,
        features=parse_feature_blob(
            feature_blob, expected_count=declared_count, context=context
        ),
    )


def iter_sample_skeleton(
    path: str | Path,
    *,
    start_row: int = 0,
    max_rows: int | None = None,
) -> Iterator[SampleSkeletonRecord]:
    """Iterate a contiguous, file-order-preserving range of sample rows."""

    if start_row < 0:
        raise ValueError("start_row cannot be negative")
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows cannot be negative")
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        for line_number, line in enumerate(source, start=1):
            if line_number <= start_row:
                continue
            if max_rows is not None and line_number > start_row + max_rows:
                break
            yield parse_sample_skeleton_line(line, line_number=line_number)


def resolve_split_files(raw_dir: str | Path, split: str) -> tuple[Path, Path]:
    """Resolve flat, ``train/``, or ``sample_train/`` archive layouts."""

    if split not in {"train", "test"}:
        raise ValueError(f"split must be train or test, got {split!r}")
    root = Path(raw_dir)
    candidate_directories = (root, root / split, root / f"sample_{split}")
    expected_names = (
        f"sample_skeleton_{split}.csv",
        f"common_features_{split}.csv",
    )
    for directory in candidate_directories:
        skeleton_path = directory / expected_names[0]
        common_path = directory / expected_names[1]
        if skeleton_path.is_file() and common_path.is_file():
            return skeleton_path, common_path

    searched = "\n".join(
        f"  - {directory / expected_names[0]}\n"
        f"    {directory / expected_names[1]}"
        for directory in candidate_directories
    )
    raise FileNotFoundError(
        f"Could not locate the Ali-CCP {split} pair. Searched:\n{searched}"
    )


def load_selected_common_features(
    path: str | Path,
    required_ids: set[str],
    *,
    audit_all_rows: bool = False,
) -> CommonFeatureSelection:
    """Load referenced common rows and optionally audit duplicate target keys.

    Non-target feature blobs are never decoded. With ``audit_all_rows=True`` the
    complete common file is scanned so duplicate occurrences of referenced keys
    cannot hide after the first match.
    """

    remaining = set(required_ids)
    selected: dict[str, CommonFeatureRecord] = {}
    occurrence_counts = {common_id: 0 for common_id in required_ids}
    rows_scanned = 0
    if not remaining:
        return CommonFeatureSelection(selected, occurrence_counts, rows_scanned)

    with Path(path).open("r", encoding="utf-8", newline="") as source:
        for line_number, line in enumerate(source, start=1):
            rows_scanned = line_number
            # The common table is several GB. Avoid decoding hundreds of
            # feature triplets for rows not referenced by the selected sample.
            first_comma = line.find(",")
            if first_comma < 0:
                raise AliCCPFormatError(
                    f"common_features line {line_number}: missing comma separator"
                )
            common_feature_id = line[:first_comma]
            if common_feature_id not in required_ids:
                continue
            occurrence_counts[common_feature_id] += 1
            if common_feature_id in remaining:
                record = parse_common_feature_line(line, line_number=line_number)
                selected[record.common_feature_id] = record
                remaining.remove(record.common_feature_id)
            if not audit_all_rows and not remaining:
                break
    return CommonFeatureSelection(selected, occurrence_counts, rows_scanned)
