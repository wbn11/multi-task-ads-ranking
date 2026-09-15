"""DataLoader factory shared by future training programs."""

from __future__ import annotations

from typing import Any

from .dataset import AdsDataset, ArrowBatchAdsDataset, collate_ads_batch


def _identity(value: Any) -> Any:
    """Keep an already-collated columnar batch unchanged."""

    return value


def create_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool = False,
    sampler: Any | None = None,
    columnar_batching: bool = True,
) -> Any:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if shuffle and sampler is not None:
        raise ValueError("shuffle and sampler cannot be enabled together")
    try:
        from torch.utils.data import DataLoader, IterableDataset
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required to create a DataLoader") from error

    if not isinstance(columnar_batching, bool):
        raise TypeError("columnar_batching must be a boolean")
    iterable = isinstance(dataset, IterableDataset)
    if iterable and sampler is not None:
        raise ValueError("sampler is not supported by an IterableDataset")
    if iterable and hasattr(dataset, "set_shuffle"):
        dataset.set_shuffle(shuffle)

    loader_options: dict[str, Any] = {}
    if num_workers > 0:
        loader_options["prefetch_factor"] = 2
    loader_dataset = dataset
    loader_batch_size: int | None = batch_size
    loader_drop_last = drop_last
    loader_collate = collate_ads_batch
    if columnar_batching and isinstance(dataset, AdsDataset):
        loader_dataset = ArrowBatchAdsDataset(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
        )
        loader_batch_size = None
        loader_drop_last = False
        loader_collate = _identity

    return DataLoader(
        loader_dataset,
        batch_size=loader_batch_size,
        shuffle=False if iterable else shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=loader_drop_last,
        persistent_workers=num_workers > 0,
        collate_fn=loader_collate,
        sampler=sampler,
        **loader_options,
    )
