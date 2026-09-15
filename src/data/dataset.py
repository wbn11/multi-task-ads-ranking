"""Iterable reader and batch collation for processed Ali-CCP Parquet data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .feature_schema import COMMON_FIELD_IDS, SAMPLE_FIELD_IDS
from .feature_encoder import UNK_INDEX
from .feature_schema import ALL_FIELD_IDS
from .preprocess import DATASET_FORMAT_VERSION, STORAGE_FORMAT


MISSING_USER_GROUP_ID = -1


def stable_user_group_id(user_id: object) -> int:
    """Return a compact, deterministic grouping key for full-scale GAUC.

    Raw anonymized user IDs remain available on individual dataset samples for
    inspection and validation.  Batches use this fixed-width key so evaluation
    does not retain tens of millions of Python string objects.
    """

    if user_id is None:
        return MISSING_USER_GROUP_ID
    normalized = str(user_id)
    if not normalized:
        return MISSING_USER_GROUP_ID
    digest = hashlib.blake2b(
        normalized.encode("utf-8"),
        digest_size=8,
        person=b"aliccp-gauc",
    ).digest()
    return int.from_bytes(digest, byteorder="little") & ((1 << 63) - 1)

try:
    import torch
    from torch.utils.data import IterableDataset, get_worker_info
except ModuleNotFoundError:  # Keep preprocessing importable without PyTorch.
    torch = None

    class IterableDataset:  # type: ignore[no-redef]
        pass

    def get_worker_info() -> None:  # type: ignore[misc]
        return None


def _require_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyArrow is required to read processed data; install requirements.txt"
        ) from error
    return pq


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyArrow is required to read processed data; install requirements.txt"
        ) from error
    return pa, pc


def _materialize_feature_columns(
    table: Any,
    field_ids: tuple[str, ...],
) -> dict[str, tuple[list[list[int]], list[list[float]]]]:
    """Cross the Arrow/Python boundary once per column instead of per cell."""

    columns: dict[str, tuple[list[list[int]], list[list[float]]]] = {}
    for field_id in field_ids:
        id_rows = table.column(f"field_{field_id}_ids").to_pylist()
        value_rows = table.column(f"field_{field_id}_values").to_pylist()
        columns[field_id] = (id_rows, value_rows)
    return columns


def _row_features(
    columns: dict[str, tuple[list[list[int]], list[list[float]]]],
    row_index: int,
) -> dict[str, dict[str, list[int] | list[float]]]:
    features: dict[str, dict[str, list[int] | list[float]]] = {}
    for field_id, (id_rows, value_rows) in columns.items():
        ids = id_rows[row_index]
        if not ids:
            continue
        values = value_rows[row_index]
        if len(ids) != len(values):
            raise ValueError(f"ids/values length mismatch for field {field_id}")
        features[field_id] = {
            "ids": [int(value) for value in ids],
            "values": [float(value) for value in values],
        }
    return features


class AdsDataset(IterableDataset):
    """Stream shard-local joins while assigning disjoint shards to workers."""

    def __init__(
        self,
        processed_dir: str | Path,
        *,
        split: str,
        shuffle_shards: bool = False,
        shuffle_within_shard: bool = False,
        seed: int = 2026,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.manifest_path = self.processed_dir / "manifest.json"
        self.metadata_path = self.processed_dir / split / "metadata.json"
        for path in (self.manifest_path, self.metadata_path):
            if not path.is_file():
                raise FileNotFoundError(f"processed dataset file is missing: {path}")

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("format_version") != DATASET_FORMAT_VERSION
            or manifest.get("storage_format") != STORAGE_FORMAT
        ):
            raise ValueError("unsupported sharded dataset format")
        split_payload = manifest.get("splits", {}).get(split)
        if not isinstance(split_payload, dict):
            raise ValueError(f"manifest has no split {split!r}")
        self.shards = list(split_payload.get("shards", []))
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.sample_count = int(self.metadata["counts"]["samples"])
        if sum(int(shard["sample_count"]) for shard in self.shards) != self.sample_count:
            raise ValueError("manifest shard counts do not match split metadata")
        for shard in self.shards:
            for key in ("samples", "common"):
                path = self.processed_dir / str(shard[key])
                if not path.is_file():
                    raise FileNotFoundError(f"dataset shard is missing: {path}")

        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within_shard = bool(shuffle_within_shard)
        self.seed = int(seed)
        self._iteration = 0
        self.negative_keep_probability = 1.0
        self.negative_sampling_seed = int(seed)

    def __len__(self) -> int:
        return self.sample_count

    def set_shuffle(self, enabled: bool) -> None:
        self.shuffle_shards = bool(enabled)
        self.shuffle_within_shard = bool(enabled)

    def set_iteration(self, completed_epochs: int) -> None:
        if completed_epochs < 0:
            raise ValueError("completed_epochs cannot be negative")
        self._iteration = int(completed_epochs)

    def set_negative_sampling(self, *, keep_probability: float, seed: int) -> None:
        if not 0.0 < keep_probability <= 1.0:
            raise ValueError("negative keep probability must be in (0, 1]")
        self.negative_keep_probability = float(keep_probability)
        self.negative_sampling_seed = int(seed)

    def _worker_shards(self) -> tuple[list[dict[str, Any]], int]:
        iteration = self._iteration
        self._iteration += 1
        order = np.arange(len(self.shards), dtype=np.int64)
        if self.shuffle_shards and len(order) > 1:
            np.random.default_rng(
                np.random.SeedSequence([self.seed, iteration])
            ).shuffle(order)
        worker = get_worker_info()
        if worker is not None:
            order = order[worker.id :: worker.num_workers]
        return [self.shards[int(index)] for index in order], iteration

    def _iter_shard(
        self,
        shard: dict[str, Any],
        *,
        iteration: int,
    ) -> Iterator[dict[str, Any]]:
        pq = _require_parquet()
        sample_table = pq.read_table(
            self.processed_dir / str(shard["samples"]), memory_map=True
        )
        common_table = pq.read_table(
            self.processed_dir / str(shard["common"]), memory_map=True
        )
        sample_count = int(shard["sample_count"])
        common_count = int(shard["common_record_count"])
        if sample_table.num_rows != sample_count:
            raise ValueError("sample shard row count does not match manifest")
        if common_table.num_rows != common_count:
            raise ValueError("common shard row count does not match manifest")

        common_feature_ids = common_table.column("common_feature_id").to_pylist()
        user_ids = common_table.column("user_id").to_pylist()
        common_feature_columns = _materialize_feature_columns(
            common_table, COMMON_FIELD_IDS
        )
        common_records: list[dict[str, Any]] = []
        for common_index in range(common_count):
            user_id = user_ids[common_index]
            common_records.append(
                {
                    "common_feature_id": common_feature_ids[common_index],
                    "user_id": user_id,
                    "user_group_id": stable_user_group_id(user_id),
                    "features": _row_features(
                        common_feature_columns, common_index
                    ),
                }
            )

        sample_ids = sample_table.column("sample_id").to_pylist()
        common_indices = sample_table.column("common_index").to_numpy()
        clicks = sample_table.column("click").to_numpy()
        conversions = sample_table.column("conversion").to_numpy()
        sample_feature_columns = _materialize_feature_columns(
            sample_table, SAMPLE_FIELD_IDS
        )

        row_order = np.arange(sample_count, dtype=np.int64)
        if self.shuffle_within_shard and sample_count > 1:
            np.random.default_rng(
                np.random.SeedSequence(
                    [self.seed, iteration, int(shard.get("index", 0))]
                )
            ).shuffle(row_order)

        sampling_generator = np.random.default_rng(
            np.random.SeedSequence(
                [
                    self.negative_sampling_seed,
                    iteration,
                    int(shard.get("index", 0)),
                ]
            )
        )

        for raw_index in row_order:
            row_index = int(raw_index)
            click = int(clicks[row_index])
            if (
                click == 0
                and self.negative_keep_probability < 1.0
                and sampling_generator.random() >= self.negative_keep_probability
            ):
                continue
            common_index = int(common_indices[row_index])
            if not 0 <= common_index < common_count:
                raise ValueError(f"common_index outside shard: {common_index}")
            common = common_records[common_index]
            sample_features = _row_features(sample_feature_columns, row_index)
            overlap = set(sample_features) & set(common["features"])
            if overlap:
                raise ValueError(f"sample/common fields overlap: {sorted(overlap)}")
            conversion = int(conversions[row_index])
            yield {
                "sample_id": sample_ids[row_index],
                "common_feature_id": common["common_feature_id"],
                "user_id": common["user_id"],
                "user_group_id": common["user_group_id"],
                "features": {**common["features"], **sample_features},
                "click": click,
                "conversion": conversion,
                "ctcvr": click * conversion,
            }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        shards, iteration = self._worker_shards()
        for shard in shards:
            yield from self._iter_shard(shard, iteration=iteration)

    def close(self) -> None:
        """Keep a uniform lifecycle API; the iterable reader owns no open handle."""


def _combine_column(table: Any, name: str) -> Any:
    """Return one contiguous Arrow array for a table column."""

    return table.column(name).combine_chunks()


def _numpy_copy(array: Any, dtype: Any) -> np.ndarray:
    """Copy an Arrow primitive array into writable, contiguous NumPy storage."""

    return np.asarray(
        array.to_numpy(zero_copy_only=False), dtype=dtype
    ).copy()


def _list_field_tensors(
    table: Any,
    field_id: str,
    *,
    pa: Any,
    pc: Any,
) -> dict[str, Any]:
    """Convert one Arrow ListArray pair directly to EmbeddingBag tensors.

    Missing fields are represented by empty lists in Parquet.  The legacy
    row-wise collator replaces each empty bag with the field-local UNK token;
    this columnar path performs the same operation in Arrow before flattening.
    """

    id_array = _combine_column(table, f"field_{field_id}_ids")
    value_array = _combine_column(table, f"field_{field_id}_values")
    id_lengths = pc.list_value_length(id_array)
    value_lengths = pc.list_value_length(value_array)
    id_length_values = _numpy_copy(id_lengths, np.int64)
    value_length_values = _numpy_copy(value_lengths, np.int64)
    if not np.array_equal(id_length_values, value_length_values):
        raise ValueError(f"ids/values length mismatch for field {field_id}")

    empty = pc.equal(id_lengths, 0)
    id_array = pc.if_else(
        empty,
        pa.scalar([UNK_INDEX], type=pa.list_(pa.int32())),
        id_array,
    )
    value_array = pc.if_else(
        empty,
        pa.scalar([1.0], type=pa.list_(pa.float32())),
        value_array,
    )
    id_offsets = _numpy_copy(id_array.offsets, np.int64)
    value_offsets = _numpy_copy(value_array.offsets, np.int64)
    if not np.array_equal(id_offsets, value_offsets):
        raise ValueError(f"ids/values offsets mismatch for field {field_id}")

    value_start = int(id_offsets[0])
    value_end = int(id_offsets[-1])
    ids = id_array.values.slice(value_start, value_end - value_start)
    values = value_array.values.slice(value_start, value_end - value_start)
    return {
        "ids": torch.from_numpy(_numpy_copy(ids, np.int64)),
        "values": torch.from_numpy(_numpy_copy(values, np.float32)),
        "offsets": torch.from_numpy(id_offsets - value_start),
    }


def _columnar_batch_from_tables(
    *,
    sample_table: Any,
    common_table: Any,
    common_group_ids: np.ndarray,
    row_indices: np.ndarray,
) -> dict[str, Any]:
    """Join and tensorize one batch without creating per-sample Python dicts."""

    if torch is None:
        raise RuntimeError("PyTorch is required for columnar batching")
    pa, pc = _require_pyarrow()
    arrow_indices = pa.array(row_indices, type=pa.int64())
    sample_batch = sample_table.take(arrow_indices)
    common_indices = _numpy_copy(
        _combine_column(sample_batch, "common_index"), np.int64
    )
    if common_indices.size and (
        int(common_indices.min()) < 0
        or int(common_indices.max()) >= common_table.num_rows
    ):
        raise ValueError("common_index outside shard")
    common_batch = common_table.take(pa.array(common_indices, type=pa.int64()))

    features: dict[str, dict[str, Any]] = {}
    for field_id in COMMON_FIELD_IDS:
        features[field_id] = _list_field_tensors(
            common_batch, field_id, pa=pa, pc=pc
        )
    for field_id in SAMPLE_FIELD_IDS:
        features[field_id] = _list_field_tensors(
            sample_batch, field_id, pa=pa, pc=pc
        )

    clicks = _numpy_copy(_combine_column(sample_batch, "click"), np.float32)
    conversions = _numpy_copy(
        _combine_column(sample_batch, "conversion"), np.float32
    )
    return {
        "sample_id": _combine_column(sample_batch, "sample_id").to_pylist(),
        "common_feature_id": _combine_column(
            common_batch, "common_feature_id"
        ).to_pylist(),
        "user_group_id": torch.from_numpy(
            np.asarray(common_group_ids[common_indices], dtype=np.int64).copy()
        ),
        "features": features,
        "click": torch.from_numpy(clicks),
        "conversion": torch.from_numpy(conversions),
        "ctcvr": torch.from_numpy(clicks * conversions),
    }


def _batch_size(batch: dict[str, Any]) -> int:
    return int(batch["click"].numel())


def _concatenate_columnar_batches(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    """Concatenate two already-collated batches, including ragged offsets."""

    features: dict[str, dict[str, Any]] = {}
    for field_id in ALL_FIELD_IDS:
        first_field = first["features"][field_id]
        second_field = second["features"][field_id]
        first_token_count = int(first_field["ids"].numel())
        features[field_id] = {
            "ids": torch.cat((first_field["ids"], second_field["ids"])),
            "values": torch.cat(
                (first_field["values"], second_field["values"])
            ),
            "offsets": torch.cat(
                (
                    first_field["offsets"],
                    second_field["offsets"][1:] + first_token_count,
                )
            ),
        }
    return {
        "sample_id": first["sample_id"] + second["sample_id"],
        "common_feature_id": (
            first["common_feature_id"] + second["common_feature_id"]
        ),
        "user_group_id": torch.cat(
            (first["user_group_id"], second["user_group_id"])
        ),
        "features": features,
        "click": torch.cat((first["click"], second["click"])),
        "conversion": torch.cat(
            (first["conversion"], second["conversion"])
        ),
        "ctcvr": torch.cat((first["ctcvr"], second["ctcvr"])),
    }


class ArrowBatchAdsDataset(IterableDataset):
    """Yield model-ready Arrow-columnar batches from an :class:`AdsDataset`.

    The wrapper deliberately keeps the source dataset's sharding, shuffling,
    negative-sampling and epoch-resume state.  It changes only tensorization:
    Arrow ListArray buffers become flattened ids/values/offsets directly,
    avoiding millions of temporary per-sample dictionaries and Python lists.
    """

    batching_mode = "arrow_columnar"

    def __init__(
        self,
        source: AdsDataset,
        *,
        batch_size: int,
        drop_last: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.source = source
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.metadata = source.metadata

    def __len__(self) -> int:
        # Trainers use len(loader.dataset) to preallocate evaluation arrays.
        return len(self.source)

    def set_iteration(self, completed_epochs: int) -> None:
        self.source.set_iteration(completed_epochs)

    def _iter_shard_batches(
        self,
        shard: dict[str, Any],
        *,
        iteration: int,
        first_batch_size: int,
    ) -> Iterator[dict[str, Any]]:
        pq = _require_parquet()
        sample_table = pq.read_table(
            self.source.processed_dir / str(shard["samples"]), memory_map=True
        )
        common_table = pq.read_table(
            self.source.processed_dir / str(shard["common"]), memory_map=True
        )
        sample_count = int(shard["sample_count"])
        common_count = int(shard["common_record_count"])
        if sample_table.num_rows != sample_count:
            raise ValueError("sample shard row count does not match manifest")
        if common_table.num_rows != common_count:
            raise ValueError("common shard row count does not match manifest")

        user_ids = _combine_column(common_table, "user_id").to_pylist()
        common_group_ids = np.fromiter(
            (stable_user_group_id(user_id) for user_id in user_ids),
            dtype=np.int64,
            count=common_count,
        )
        row_order = np.arange(sample_count, dtype=np.int64)
        if self.source.shuffle_within_shard and sample_count > 1:
            np.random.default_rng(
                np.random.SeedSequence(
                    [
                        self.source.seed,
                        iteration,
                        int(shard.get("index", 0)),
                    ]
                )
            ).shuffle(row_order)

        if self.source.negative_keep_probability < 1.0:
            clicks = _numpy_copy(
                _combine_column(sample_table, "click"), np.uint8
            )
            ordered_clicks = clicks[row_order]
            negative_positions = ordered_clicks == 0
            sampling_generator = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        self.source.negative_sampling_seed,
                        iteration,
                        int(shard.get("index", 0)),
                    ]
                )
            )
            keep = np.ones(row_order.size, dtype=np.bool_)
            keep[negative_positions] = sampling_generator.random(
                int(negative_positions.sum())
            ) < self.source.negative_keep_probability
            row_order = row_order[keep]

        if not 0 < first_batch_size <= self.batch_size:
            raise ValueError("first_batch_size must be in [1, batch_size]")
        start = 0
        next_size = first_batch_size
        while start < row_order.size:
            end = min(start + next_size, row_order.size)
            yield _columnar_batch_from_tables(
                sample_table=sample_table,
                common_table=common_table,
                common_group_ids=common_group_ids,
                row_indices=row_order[start:end],
            )
            start = end
            next_size = self.batch_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        shards, iteration = self.source._worker_shards()
        pending: dict[str, Any] | None = None
        for shard in shards:
            needed = (
                self.batch_size
                if pending is None
                else self.batch_size - _batch_size(pending)
            )
            for batch in self._iter_shard_batches(
                shard,
                iteration=iteration,
                first_batch_size=needed,
            ):
                if pending is not None:
                    batch = _concatenate_columnar_batches(pending, batch)
                    pending = None
                size = _batch_size(batch)
                if size < self.batch_size:
                    pending = batch
                    continue
                if size != self.batch_size:
                    raise RuntimeError("columnar batch exceeded configured size")
                yield batch
        if pending is not None and not self.drop_last:
            yield pending


def collate_ads_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Create per-field ragged tensors compatible with ``EmbeddingBag``."""

    if torch is None:
        raise RuntimeError("PyTorch is required for collate_ads_batch")
    if not batch:
        raise ValueError("cannot collate an empty batch")

    batched_features: dict[str, dict[str, Any]] = {}
    for field_id in ALL_FIELD_IDS:
        flattened_ids: list[int] = []
        flattened_values: list[float] = []
        offsets = [0]
        for sample in batch:
            field_value = sample["features"].get(field_id)
            if field_value and field_value.get("ids"):
                ids = [int(value) for value in field_value["ids"]]
                values = [float(value) for value in field_value["values"]]
                if len(ids) != len(values):
                    raise ValueError(
                        f"ids/values length mismatch for field {field_id}"
                    )
            else:
                ids = [UNK_INDEX]
                values = [1.0]
            flattened_ids.extend(ids)
            flattened_values.extend(values)
            offsets.append(len(flattened_ids))

        batched_features[field_id] = {
            "ids": torch.tensor(flattened_ids, dtype=torch.long),
            "values": torch.tensor(flattened_values, dtype=torch.float32),
            "offsets": torch.tensor(offsets, dtype=torch.long),
        }

    return {
        "sample_id": [sample["sample_id"] for sample in batch],
        "common_feature_id": [sample["common_feature_id"] for sample in batch],
        # Raw user strings are intentionally not sent through DataLoader IPC.
        # The fixed-width grouping key is sufficient for exact GAUC grouping.
        "user_group_id": torch.tensor(
            [
                sample.get(
                    "user_group_id",
                    stable_user_group_id(sample.get("user_id")),
                )
                for sample in batch
            ],
            dtype=torch.int64,
        ),
        "features": batched_features,
        "click": torch.tensor(
            [sample["click"] for sample in batch], dtype=torch.float32
        ),
        "conversion": torch.tensor(
            [sample["conversion"] for sample in batch], dtype=torch.float32
        ),
        "ctcvr": torch.tensor(
            [sample["ctcvr"] for sample in batch], dtype=torch.float32
        ),
    }
