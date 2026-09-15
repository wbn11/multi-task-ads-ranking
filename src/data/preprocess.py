"""Build Ali-CCP model data as independently readable Parquet shards.

Raw samples are never materialized as one Python list, common rows are located
through a disk-backed byte-offset index, and each output shard contains only
the common rows referenced by that shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .feature_encoder import FeatureEncoder, UNK_INDEX
from .feature_schema import ALL_FIELD_IDS, COMMON_FIELD_IDS, FIELD_SPEC_BY_ID, SAMPLE_FIELD_IDS
from .parser import (
    AliCCPFormatError,
    CommonFeatureRecord,
    SampleSkeletonRecord,
    SparseFeature,
    iter_sample_skeleton,
    parse_common_feature_line,
    resolve_split_files,
)
DATASET_FORMAT_VERSION = 1
STORAGE_FORMAT = "parquet_shards"
_HASH_SPACE = 1 << 64
_COMMON_FIELD_ID_SET = frozenset(COMMON_FIELD_IDS)
_SAMPLE_FIELD_ID_SET = frozenset(SAMPLE_FIELD_IDS)


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyArrow is required for preprocessing; install requirements.txt "
            "before running scripts/preprocess.py"
        ) from error
    return pa, pq


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _extract_raw_user_id(record: CommonFeatureRecord) -> str | None:
    user_ids = [
        feature.feature_id
        for feature in record.features
        if feature.field_id == "101"
    ]
    if len(user_ids) > 1:
        raise ValueError(
            f"common feature {record.common_feature_id!r} contains multiple "
            "field-101 user IDs"
        )
    return user_ids[0] if user_ids else None


def _count_features(
    frequencies: dict[str, Counter[str]],
    features: Iterable[SparseFeature],
    *,
    weight: int = 1,
) -> None:
    if weight <= 0:
        raise ValueError("feature occurrence weight must be positive")
    for feature in features:
        if feature.field_id not in frequencies:
            raise ValueError(f"unexpected Ali-CCP field_id: {feature.field_id}")
        frequencies[feature.field_id][feature.feature_id] += weight


def _update_oov(
    encoder: FeatureEncoder,
    stats: dict[str, Counter[str]],
    features: Iterable[SparseFeature],
    *,
    weight: int = 1,
) -> None:
    for feature in features:
        field_stats = stats[feature.field_id]
        field_stats["tokens"] += weight
        if encoder.encode_token(feature.field_id, feature.feature_id) == UNK_INDEX:
            field_stats["mapped_to_unk"] += weight


def _oov_report(stats: Mapping[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for field_id in ALL_FIELD_IDS:
        field_stats = stats.get(field_id, Counter())
        tokens = int(field_stats["tokens"])
        unknown = int(field_stats["mapped_to_unk"])
        report[field_id] = {
            "name": FIELD_SPEC_BY_ID[field_id].name,
            "tokens": tokens,
            "mapped_to_unk": unknown,
            "oov_rate": _safe_rate(unknown, tokens),
        }
    return report


def _training_oov_stats(
    encoder: FeatureEncoder,
    frequencies: Mapping[str, Counter[str]],
) -> dict[str, Counter[str]]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for field_id in ALL_FIELD_IDS:
        for token, count in frequencies[field_id].items():
            stats[field_id]["tokens"] += count
            if encoder.encode_token(field_id, token) == UNK_INDEX:
                stats[field_id]["mapped_to_unk"] += count
    return stats


def stable_evaluation_split(
    sample_id: str,
    *,
    validation_fraction: float,
    seed: int,
) -> str:
    """Assign one official-test impression without using Python's salted hash."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    digest = hashlib.blake2b(
        f"{seed}:{sample_id}".encode("utf-8"), digest_size=8
    ).digest()
    unit_value = int.from_bytes(digest, byteorder="big") / _HASH_SPACE
    return "validation" if unit_value < validation_fraction else "test"


@dataclass(slots=True)
class SplitScan:
    name: str
    rows_selected: int = 0
    invalid_dropped: int = 0
    clicks: int = 0
    conversions: int = 0
    common_reuse: Counter[str] = field(default_factory=Counter)

    def observe(self, sample: SampleSkeletonRecord) -> None:
        self.rows_selected += 1
        if sample.has_invalid_label_chain:
            self.invalid_dropped += 1
            return
        self.clicks += sample.click
        self.conversions += sample.conversion
        self.common_reuse[sample.common_feature_id] += 1

    @property
    def samples(self) -> int:
        return self.rows_selected - self.invalid_dropped


def _scan_samples(
    skeleton_path: Path,
    *,
    start_row: int,
    max_rows: int | None,
    route: Callable[[SampleSkeletonRecord], str],
    split_names: tuple[str, ...],
    frequencies: dict[str, Counter[str]] | None,
    progress_every: int,
) -> dict[str, SplitScan]:
    scans = {name: SplitScan(name) for name in split_names}
    rows_seen = 0
    for sample in iter_sample_skeleton(
        skeleton_path, start_row=start_row, max_rows=max_rows
    ):
        rows_seen += 1
        split = route(sample)
        try:
            scan = scans[split]
        except KeyError as error:
            raise ValueError(f"sample router returned unknown split {split!r}") from error
        scan.observe(sample)
        if not sample.has_invalid_label_chain and frequencies is not None:
            _count_features(frequencies, sample.features)
        if progress_every and rows_seen % progress_every == 0:
            logging.info(
                "sample scan %s: %s rows",
                skeleton_path.name,
                f"{rows_seen:,}",
            )
    return scans


class CommonOffsetIndex:
    """Disk-backed mapping from anonymous common ID to its raw byte offset."""

    def __init__(self, path: Path, *, create: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        if create:
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS offsets ("
                "common_id TEXT PRIMARY KEY, byte_offset INTEGER NOT NULL)"
            )
            self.connection.execute("DELETE FROM offsets")
            self.connection.commit()

    def add_many(self, rows: list[tuple[str, int]]) -> None:
        try:
            self.connection.executemany(
                "INSERT INTO offsets(common_id, byte_offset) VALUES (?, ?)", rows
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("common_features contains a duplicated referenced ID") from error

    def get(self, common_id: str) -> int:
        row = self.connection.execute(
            "SELECT byte_offset FROM offsets WHERE common_id = ?", (common_id,)
        ).fetchone()
        if row is None:
            raise KeyError(common_id)
        return int(row[0])

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM offsets").fetchone()[0])

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def _index_common_rows(
    common_path: Path,
    *,
    index: CommonOffsetIndex,
    usage_by_split: Mapping[str, Counter[str]],
    frequencies: dict[str, Counter[str]] | None,
    encoder: FeatureEncoder | None,
    oov_by_split: Mapping[str, dict[str, Counter[str]]] | None,
    progress_every: int,
) -> int:
    required_ids: set[str] = set()
    for usage in usage_by_split.values():
        required_ids.update(usage)

    pending: list[tuple[str, int]] = []
    found: set[str] = set()
    rows_scanned = 0
    with common_path.open("rb") as source:
        while True:
            offset = source.tell()
            raw_line = source.readline()
            if not raw_line:
                break
            rows_scanned += 1
            first_comma = raw_line.find(b",")
            if first_comma < 0:
                raise AliCCPFormatError(
                    f"common_features line {rows_scanned}: missing comma separator"
                )
            common_id = raw_line[:first_comma].decode("utf-8")
            if common_id in required_ids:
                if common_id in found:
                    raise ValueError(f"duplicated referenced common ID: {common_id}")
                found.add(common_id)
                pending.append((common_id, offset))
                record = parse_common_feature_line(
                    raw_line.decode("utf-8"), line_number=rows_scanned
                )
                if frequencies is not None:
                    total_reuse = sum(
                        usage.get(common_id, 0) for usage in usage_by_split.values()
                    )
                    _count_features(frequencies, record.features, weight=total_reuse)
                if encoder is not None and oov_by_split is not None:
                    for split, usage in usage_by_split.items():
                        reuse = usage.get(common_id, 0)
                        if reuse:
                            _update_oov(
                                encoder,
                                oov_by_split[split],
                                record.features,
                                weight=reuse,
                            )
                if len(pending) >= 10_000:
                    index.add_many(pending)
                    index.commit()
                    pending.clear()
            if progress_every and rows_scanned % progress_every == 0:
                logging.info(
                    "common scan %s: %s rows, %s referenced",
                    common_path.name,
                    f"{rows_scanned:,}",
                    f"{len(found):,}",
                )
    if pending:
        index.add_many(pending)
        index.commit()

    if len(found) != len(required_ids):
        missing = [common_id for common_id in required_ids if common_id not in found]
        raise ValueError(
            f"{len(missing)} referenced common IDs are missing; examples={missing[:5]}"
        )
    return rows_scanned


class CommonRecordLoader:
    def __init__(
        self,
        common_path: Path,
        index: CommonOffsetIndex,
        encoder: FeatureEncoder,
    ) -> None:
        self.reader = common_path.open("rb")
        self.index = index
        self.encoder = encoder

    def load(self, common_id: str) -> dict[str, Any]:
        self.reader.seek(self.index.get(common_id))
        raw_line = self.reader.readline()
        record = parse_common_feature_line(raw_line.decode("utf-8"))
        if record.common_feature_id != common_id:
            raise RuntimeError(
                f"common offset index mismatch: expected={common_id}, "
                f"actual={record.common_feature_id}"
            )
        unexpected_fields = {
            feature.field_id for feature in record.features
        } - _COMMON_FIELD_ID_SET
        if unexpected_fields:
            raise ValueError(
                f"common record {common_id} contains sample-side fields: "
                f"{sorted(unexpected_fields)}"
            )
        return {
            "common_feature_id": common_id,
            "user_id": _extract_raw_user_id(record),
            "features": self.encoder.encode_features(record.features),
        }

    def close(self) -> None:
        self.reader.close()


def _feature_columns(
    records: list[dict[str, Any]], field_ids: tuple[str, ...]
) -> dict[str, list[list[int] | list[float]]]:
    columns: dict[str, list[list[int] | list[float]]] = {}
    for field_id in field_ids:
        id_rows: list[list[int]] = []
        value_rows: list[list[float]] = []
        for record in records:
            field_value = record["features"].get(field_id)
            if field_value is None:
                id_rows.append([])
                value_rows.append([])
            else:
                id_rows.append([int(value) for value in field_value["ids"]])
                value_rows.append([float(value) for value in field_value["values"]])
        columns[f"field_{field_id}_ids"] = id_rows
        columns[f"field_{field_id}_values"] = value_rows
    return columns


def _write_parquet_atomic(
    path: Path,
    columns: Mapping[str, Any],
    *,
    compression: str,
) -> None:
    pa, pq = _require_pyarrow()
    arrays: dict[str, Any] = {}
    for name, values in columns.items():
        if name in {"click", "conversion"}:
            target_type = pa.uint8()
        elif name == "common_index":
            target_type = pa.int32()
        elif name.endswith("_ids"):
            target_type = pa.list_(pa.int32())
        elif name.endswith("_values"):
            target_type = pa.list_(pa.float32())
        else:
            target_type = pa.string()
        arrays[name] = pa.array(values, type=target_type)
    table = pa.table(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        table,
        temporary,
        compression=compression,
        use_dictionary=True,
        row_group_size=32_768,
    )
    temporary.replace(path)


class ParquetShardWriter:
    def __init__(
        self,
        *,
        output_root: Path,
        split: str,
        shard_size: int,
        compression: str,
        common_loader: CommonRecordLoader,
        encoder: FeatureEncoder,
        oov_stats: dict[str, Counter[str]] | None,
    ) -> None:
        self.output_root = output_root
        self.split = split
        self.shard_size = shard_size
        self.compression = compression
        self.common_loader = common_loader
        self.encoder = encoder
        self.oov_stats = oov_stats
        self.samples: list[dict[str, Any]] = []
        self.common_records: list[dict[str, Any]] = []
        self.common_indices: dict[str, int] = {}
        self.shards: list[dict[str, Any]] = []

    def add(self, sample: SampleSkeletonRecord) -> None:
        unexpected_fields = {
            feature.field_id for feature in sample.features
        } - _SAMPLE_FIELD_ID_SET
        if unexpected_fields:
            raise ValueError(
                f"sample {sample.sample_id} contains common-side fields: "
                f"{sorted(unexpected_fields)}"
            )
        common_id = sample.common_feature_id
        common_index = self.common_indices.get(common_id)
        if common_index is None:
            common_index = len(self.common_records)
            self.common_indices[common_id] = common_index
            self.common_records.append(self.common_loader.load(common_id))
        if self.oov_stats is not None:
            _update_oov(self.encoder, self.oov_stats, sample.features)
        self.samples.append(
            {
                "sample_id": sample.sample_id,
                "common_index": common_index,
                "features": self.encoder.encode_features(sample.features),
                "click": sample.click,
                "conversion": sample.conversion,
            }
        )
        if len(self.samples) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.samples:
            return
        shard_index = len(self.shards)
        filename = f"part-{shard_index:05d}.parquet"
        sample_relative = Path(self.split) / "samples" / filename
        common_relative = Path(self.split) / "common" / filename
        sample_columns: dict[str, Any] = {
            "sample_id": [record["sample_id"] for record in self.samples],
            "common_index": [record["common_index"] for record in self.samples],
            "click": [record["click"] for record in self.samples],
            "conversion": [record["conversion"] for record in self.samples],
            **_feature_columns(self.samples, SAMPLE_FIELD_IDS),
        }
        common_columns: dict[str, Any] = {
            "common_feature_id": [
                record["common_feature_id"] for record in self.common_records
            ],
            "user_id": [record["user_id"] for record in self.common_records],
            **_feature_columns(self.common_records, COMMON_FIELD_IDS),
        }
        _write_parquet_atomic(
            self.output_root / sample_relative,
            sample_columns,
            compression=self.compression,
        )
        _write_parquet_atomic(
            self.output_root / common_relative,
            common_columns,
            compression=self.compression,
        )
        self.shards.append(
            {
                "index": shard_index,
                "samples": sample_relative.as_posix(),
                "common": common_relative.as_posix(),
                "sample_count": len(self.samples),
                "common_record_count": len(self.common_records),
            }
        )
        logging.info(
            "wrote %s shard %05d: %s samples, %s common rows",
            self.split,
            shard_index,
            f"{len(self.samples):,}",
            f"{len(self.common_records):,}",
        )
        self.samples.clear()
        self.common_records.clear()
        self.common_indices.clear()


def _write_sample_shards(
    skeleton_path: Path,
    *,
    start_row: int,
    max_rows: int | None,
    route: Callable[[SampleSkeletonRecord], str],
    writers: Mapping[str, ParquetShardWriter],
    progress_every: int,
) -> None:
    rows_seen = 0
    for sample in iter_sample_skeleton(
        skeleton_path, start_row=start_row, max_rows=max_rows
    ):
        rows_seen += 1
        if not sample.has_invalid_label_chain:
            writers[route(sample)].add(sample)
        if progress_every and rows_seen % progress_every == 0:
            logging.info(
                "shard write %s: %s rows",
                skeleton_path.name,
                f"{rows_seen:,}",
            )
    for writer in writers.values():
        writer.flush()


def _split_metadata(
    *,
    split: str,
    scan: SplitScan,
    source_skeleton: Path,
    source_common: Path,
    source_start_row: int,
    source_max_rows: int | None,
    shards: list[dict[str, Any]],
    oov_stats: Mapping[str, Counter[str]],
    encoder: FeatureEncoder,
) -> dict[str, Any]:
    sample_count = sum(int(shard["sample_count"]) for shard in shards)
    shard_common_count = sum(int(shard["common_record_count"]) for shard in shards)
    if sample_count != scan.samples:
        raise RuntimeError(
            f"{split} sample count changed between scan and shard write: "
            f"scan={scan.samples}, written={sample_count}"
        )
    return {
        "format_version": DATASET_FORMAT_VERSION,
        "storage_format": STORAGE_FORMAT,
        "split": split,
        "source": {
            "sample_skeleton": str(source_skeleton.resolve()),
            "common_features": str(source_common.resolve()),
            "start_row": source_start_row,
            "max_rows": source_max_rows,
            "rows_selected": scan.rows_selected,
        },
        "counts": {
            "samples": sample_count,
            "common_records": len(scan.common_reuse),
            "shard_local_common_records": shard_common_count,
            "invalid_click_0_conversion_1_dropped": scan.invalid_dropped,
            "clicks": scan.clicks,
            "conversions": scan.conversions,
            "ctcvr_positives": scan.conversions,
        },
        "rates": {
            "ctr": _safe_rate(scan.clicks, sample_count),
            "cvr_on_clicked_samples": _safe_rate(scan.conversions, scan.clicks),
            "ctcvr": _safe_rate(scan.conversions, sample_count),
        },
        "user_id": {
            "source_field_id": "101",
            "representation": "raw_anonymized_feature_id",
        },
        "encoder": {
            "fit_split": "train",
            "min_frequency": encoder.min_frequency,
            "special_indices": {"pad": 0, "unk": 1},
            "fields": encoder.field_summary(),
        },
        "oov_counting_unit": "feature occurrences in model samples",
        "oov": _oov_report(oov_stats),
        "shards": shards,
    }


def build_processed_datasets(
    *,
    raw_dir: str | Path,
    output_dir: str | Path,
    min_frequency: int,
    shard_size: int,
    validation_fraction: float,
    split_seed: int,
    evaluation_start_row: int,
    train_max_samples: int | None = None,
    evaluation_max_samples: int | None = None,
    compression: str = "zstd",
    progress_every: int = 1_000_000,
) -> dict[str, Any]:
    """Build train/validation/test data without materializing raw rows."""

    _require_pyarrow()
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if evaluation_start_row < 0:
        raise ValueError("evaluation_start_row cannot be negative")
    if train_max_samples is not None and train_max_samples <= 0:
        raise ValueError("train_max_samples must be positive or null")
    if evaluation_max_samples is not None and evaluation_max_samples <= 0:
        raise ValueError("evaluation_max_samples must be positive or null")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")

    output_root = Path(output_dir)
    manifest_path = output_root / "manifest.json"
    vocab_path = output_root / "vocab.json"
    if manifest_path.exists() or vocab_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing processed dataset: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    train_skeleton, train_common = resolve_split_files(raw_dir, "train")
    test_skeleton, test_common = resolve_split_files(raw_dir, "test")
    frequencies = {field_id: Counter() for field_id in ALL_FIELD_IDS}

    logging.info("pass 1/5: scan training samples and count sample-side features")
    train_scans = _scan_samples(
        train_skeleton,
        start_row=0,
        max_rows=train_max_samples,
        route=lambda _: "train",
        split_names=("train",),
        frequencies=frequencies,
        progress_every=progress_every,
    )
    train_index = CommonOffsetIndex(
        output_root / "cache" / "train_common_offsets.sqlite3", create=True
    )
    logging.info("pass 2/5: index referenced training common rows and finish counts")
    train_common_rows = _index_common_rows(
        train_common,
        index=train_index,
        usage_by_split={"train": train_scans["train"].common_reuse},
        frequencies=frequencies,
        encoder=None,
        oov_by_split=None,
        progress_every=progress_every,
    )
    encoder = FeatureEncoder.fit(frequencies, min_frequency=min_frequency)
    encoder.save(vocab_path)
    train_oov = _training_oov_stats(encoder, frequencies)

    logging.info("pass 3/5: encode training Parquet shards")
    train_loader = CommonRecordLoader(train_common, train_index, encoder)
    train_writer = ParquetShardWriter(
        output_root=output_root,
        split="train",
        shard_size=shard_size,
        compression=compression,
        common_loader=train_loader,
        encoder=encoder,
        oov_stats=None,
    )
    try:
        _write_sample_shards(
            train_skeleton,
            start_row=0,
            max_rows=train_max_samples,
            route=lambda _: "train",
            writers={"train": train_writer},
            progress_every=progress_every,
        )
    finally:
        train_loader.close()
        train_index.close()

    evaluation_route = lambda sample: stable_evaluation_split(
        sample.sample_id,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )
    logging.info("pass 4/5: scan official-test tail and index its common rows")
    evaluation_scans = _scan_samples(
        test_skeleton,
        start_row=evaluation_start_row,
        max_rows=evaluation_max_samples,
        route=evaluation_route,
        split_names=("validation", "test"),
        frequencies=None,
        progress_every=progress_every,
    )
    evaluation_index = CommonOffsetIndex(
        output_root / "cache" / "test_common_offsets.sqlite3", create=True
    )
    evaluation_oov = {
        "validation": defaultdict(Counter),
        "test": defaultdict(Counter),
    }
    test_common_rows = _index_common_rows(
        test_common,
        index=evaluation_index,
        usage_by_split={
            split: evaluation_scans[split].common_reuse
            for split in ("validation", "test")
        },
        frequencies=None,
        encoder=encoder,
        oov_by_split=evaluation_oov,
        progress_every=progress_every,
    )

    logging.info("pass 5/5: encode validation/test Parquet shards")
    evaluation_loader = CommonRecordLoader(test_common, evaluation_index, encoder)
    evaluation_writers = {
        split: ParquetShardWriter(
            output_root=output_root,
            split=split,
            shard_size=shard_size,
            compression=compression,
            common_loader=evaluation_loader,
            encoder=encoder,
            oov_stats=evaluation_oov[split],
        )
        for split in ("validation", "test")
    }
    try:
        _write_sample_shards(
            test_skeleton,
            start_row=evaluation_start_row,
            max_rows=evaluation_max_samples,
            route=evaluation_route,
            writers=evaluation_writers,
            progress_every=progress_every,
        )
    finally:
        evaluation_loader.close()
        evaluation_index.close()

    metadata_by_split: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        is_train = split == "train"
        scan = train_scans["train"] if is_train else evaluation_scans[split]
        writer = train_writer if is_train else evaluation_writers[split]
        metadata = _split_metadata(
            split=split,
            scan=scan,
            source_skeleton=train_skeleton if is_train else test_skeleton,
            source_common=train_common if is_train else test_common,
            source_start_row=0 if is_train else evaluation_start_row,
            source_max_rows=train_max_samples if is_train else evaluation_max_samples,
            shards=writer.shards,
            oov_stats=train_oov if is_train else evaluation_oov[split],
            encoder=encoder,
        )
        _write_json_atomic(output_root / split / "metadata.json", metadata)
        metadata_by_split[split] = metadata

    manifest: dict[str, Any] = {
        "format_version": DATASET_FORMAT_VERSION,
        "storage_format": STORAGE_FORMAT,
        "vocabulary": "vocab.json",
        "field_ids": list(ALL_FIELD_IDS),
        "common_field_ids": list(COMMON_FIELD_IDS),
        "sample_field_ids": list(SAMPLE_FIELD_IDS),
        "shard_size": shard_size,
        "compression": compression,
        "split_policy": {
            "train": "all valid rows from official train",
            "development_rows_excluded_from_official_test": evaluation_start_row,
            "evaluation_assignment": "blake2b(sample_id, seed)",
            "validation_fraction": validation_fraction,
            "seed": split_seed,
        },
        "source_common_rows_scanned": {
            "train": train_common_rows,
            "test": test_common_rows,
        },
        "splits": {
            split: {
                "metadata": f"{split}/metadata.json",
                "samples": metadata_by_split[split]["counts"]["samples"],
                "shards": metadata_by_split[split]["shards"],
            }
            for split in ("train", "validation", "test")
        },
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive or null")
    return parsed


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyYAML is required for config files; install requirements.txt"
        ) from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("data config root must be a mapping")
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Parquet Ali-CCP data for training."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-max-samples", type=int)
    parser.add_argument("--evaluation-max-samples", type=int)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    config = _load_yaml(args.config)
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise SystemExit("config must contain a data mapping")
    full = data.get("full_preprocessing")
    if not isinstance(full, Mapping):
        raise SystemExit("config must contain data.full_preprocessing")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    manifest = build_processed_datasets(
        raw_dir=args.raw_dir or Path(str(data["raw_dir"])),
        output_dir=args.output_dir or Path(str(data["processed_dir"])),
        min_frequency=int(data["min_frequency"]),
        shard_size=int(full["shard_size"]),
        validation_fraction=float(full["validation_fraction"]),
        split_seed=int(full["split_seed"]),
        evaluation_start_row=int(full["evaluation_start_row"]),
        train_max_samples=(
            args.train_max_samples
            if args.train_max_samples is not None
            else _optional_positive_int(full.get("train_max_samples"), name="train_max_samples")
        ),
        evaluation_max_samples=(
            args.evaluation_max_samples
            if args.evaluation_max_samples is not None
            else _optional_positive_int(
                full.get("evaluation_max_samples"), name="evaluation_max_samples"
            )
        ),
        compression=str(full.get("compression", "zstd")),
        progress_every=int(full.get("progress_every", 1_000_000)),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
