"""Real-batch DataLoader and EmbeddingBag smoke validation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .dataloader import create_dataloader
from .dataset import AdsDataset
from .feature_encoder import FeatureEncoder, UNK_INDEX
from .feature_schema import ALL_FIELD_IDS, FIELD_SPEC_BY_ID
from .preprocess import _load_yaml


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Check the PyTorch wheel and NVIDIA driver compatibility."
        )
    return torch.device(requested)


def run_dataloader_smoke_test(
    *,
    processed_dir: str | Path,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: str = "auto",
    embedding_dim: int = 8,
    benchmark_batches: int = 0,
    warmup_batches: int = 0,
    columnar_batching: bool = True,
) -> dict[str, Any]:
    """Validate one real batch through DataLoader, device transfer, and pooling."""

    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if benchmark_batches < 0 or warmup_batches < 0:
        raise ValueError("benchmark batch counts cannot be negative")
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for the DataLoader smoke test") from error

    target_device = _resolve_device(torch, device)
    encoder = FeatureEncoder.load(Path(processed_dir) / "vocab.json")
    dataset = AdsDataset(processed_dir, split=split)
    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        columnar_batching=columnar_batching,
    )

    started = time.perf_counter()
    iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration as error:
        dataset.close()
        raise RuntimeError(f"processed {split} dataset is empty") from error
    fetch_seconds = time.perf_counter() - started
    actual_batch_size = len(batch["sample_id"])
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    label_report: dict[str, dict[str, Any]] = {}
    for label_name in ("click", "conversion", "ctcvr"):
        tensor = batch[label_name]
        check(tensor.dtype == torch.float32, f"{label_name} must be float32")
        check(
            tuple(tensor.shape) == (actual_batch_size,),
            f"{label_name} shape must be [{actual_batch_size}]",
        )
        check(bool(torch.isfinite(tensor).all()), f"{label_name} contains NaN/Inf")
        check(
            bool(((tensor == 0) | (tensor == 1)).all()),
            f"{label_name} must be binary",
        )
        label_report[label_name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "positive_count": int(tensor.sum().item()),
        }
    check(
        bool((batch["ctcvr"] == batch["click"] * batch["conversion"]).all()),
        "ctcvr must equal click * conversion",
    )
    user_group_ids = batch["user_group_id"]
    check(user_group_ids.dtype == torch.int64, "user_group_id must be int64")
    check(
        tuple(user_group_ids.shape) == (actual_batch_size,),
        f"user_group_id shape must be [{actual_batch_size}]",
    )

    check(
        set(batch["features"]) == set(ALL_FIELD_IDS),
        "batch feature fields do not match the declared schema",
    )
    field_report: dict[str, dict[str, Any]] = {}
    transfer_started = time.perf_counter()
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)

    with torch.no_grad():
        for field_id in ALL_FIELD_IDS:
            field = batch["features"][field_id]
            ids = field["ids"]
            values = field["values"]
            offsets = field["offsets"]
            vocabulary = encoder.field_vocabularies[field_id]
            field_errors_before = len(errors)

            check(ids.dtype == torch.int64, f"field {field_id}: ids must be int64")
            check(
                values.dtype == torch.float32,
                f"field {field_id}: values must be float32",
            )
            check(
                offsets.dtype == torch.int64,
                f"field {field_id}: offsets must be int64",
            )
            check(ids.ndim == values.ndim == offsets.ndim == 1, f"field {field_id}: tensors must be 1-D")
            check(len(ids) == len(values), f"field {field_id}: ids/values length mismatch")
            check(
                len(offsets) == actual_batch_size + 1,
                f"field {field_id}: offsets length must be batch_size + 1",
            )
            check(len(ids) > 0, f"field {field_id}: no IDs in batch")
            if len(offsets) > 0:
                check(int(offsets[0]) == 0, f"field {field_id}: offsets must start at zero")
                check(
                    int(offsets[-1]) == len(ids),
                    f"field {field_id}: last offset must equal ID count",
                )
                check(
                    bool((offsets[1:] >= offsets[:-1]).all()),
                    f"field {field_id}: offsets must be monotonic",
                )
            if len(ids) > 0:
                check(
                    int(ids.min()) >= UNK_INDEX,
                    f"field {field_id}: encoded ID is below UNK index",
                )
                check(
                    int(ids.max()) < vocabulary.vocab_size,
                    f"field {field_id}: encoded ID exceeds vocabulary",
                )
            check(
                bool(torch.isfinite(values).all()),
                f"field {field_id}: values contain NaN/Inf",
            )

            pooled_shape: list[int] | None = None
            if len(errors) == field_errors_before:
                non_blocking = pin_memory and target_device.type == "cuda"
                ids_device = ids.to(target_device, non_blocking=non_blocking)
                values_device = values.to(target_device, non_blocking=non_blocking)
                offsets_device = offsets.to(target_device, non_blocking=non_blocking)
                embedding = torch.nn.EmbeddingBag(
                    vocabulary.vocab_size,
                    embedding_dim,
                    mode="sum",
                    padding_idx=0,
                    include_last_offset=True,
                ).to(target_device)
                pooled = embedding(
                    ids_device,
                    offsets_device,
                    per_sample_weights=values_device,
                )
                pooled_shape = list(pooled.shape)
                check(
                    tuple(pooled.shape) == (actual_batch_size, embedding_dim),
                    f"field {field_id}: unexpected pooled shape {tuple(pooled.shape)}",
                )
                check(
                    bool(torch.isfinite(pooled).all()),
                    f"field {field_id}: pooled embedding contains NaN/Inf",
                )

            bag_lengths = offsets[1:] - offsets[:-1]
            field_report[field_id] = {
                "name": FIELD_SPEC_BY_ID[field_id].name,
                "vocab_size": vocabulary.vocab_size,
                "token_count": len(ids),
                "min_bag_length": int(bag_lengths.min().item()),
                "max_bag_length": int(bag_lengths.max().item()),
                "ids_pinned": bool(ids.is_pinned()),
                "pooled_shape": pooled_shape,
            }

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    transfer_and_pool_seconds = time.perf_counter() - transfer_started

    def transfer_batch(cpu_batch: dict[str, Any]) -> None:
        non_blocking = pin_memory and target_device.type == "cuda"
        for field in cpu_batch["features"].values():
            for tensor in field.values():
                tensor.to(target_device, non_blocking=non_blocking)
        for label_name in ("click", "conversion", "ctcvr"):
            cpu_batch[label_name].to(target_device, non_blocking=non_blocking)

    exhausted = False
    warmed_batches = 0
    for _ in range(warmup_batches):
        try:
            warmup_batch = next(iterator)
        except StopIteration:
            exhausted = True
            break
        transfer_batch(warmup_batch)
        warmed_batches += 1
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)

    measured_batches = 0
    measured_samples = 0
    benchmark_started = time.perf_counter()
    if not exhausted:
        for _ in range(benchmark_batches):
            try:
                benchmark_batch = next(iterator)
            except StopIteration:
                exhausted = True
                break
            transfer_batch(benchmark_batch)
            measured_batches += 1
            measured_samples += int(benchmark_batch["click"].numel())
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    benchmark_seconds = time.perf_counter() - benchmark_started
    dataset.close()

    device_report: dict[str, Any] = {
        "requested": device,
        "resolved": str(target_device),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if target_device.type == "cuda":
        device_report.update(
            {
                "gpu_name": torch.cuda.get_device_name(target_device),
                "peak_memory_mib": torch.cuda.max_memory_allocated(target_device)
                / 1024**2,
            }
        )

    return {
        "valid": not errors,
        "split": split,
        "dataset_size": len(dataset),
        "batch_size": actual_batch_size,
        "batching_mode": getattr(
            loader.dataset, "batching_mode", "legacy_row_collation"
        ),
        "field_count": len(batch["features"]),
        "embedding_dim": embedding_dim,
        "device": device_report,
        "timing_seconds": {
            "first_batch_fetch": fetch_seconds,
            "device_transfer_and_embedding_pool": transfer_and_pool_seconds,
        },
        "throughput_benchmark": {
            "requested_warmup_batches": warmup_batches,
            "completed_warmup_batches": warmed_batches,
            "requested_measured_batches": benchmark_batches,
            "completed_measured_batches": measured_batches,
            "measured_samples": measured_samples,
            "seconds": benchmark_seconds,
            "samples_per_second": (
                measured_samples / benchmark_seconds
                if measured_batches and benchmark_seconds > 0.0
                else None
            ),
            "batches_per_second": (
                measured_batches / benchmark_seconds
                if measured_batches and benchmark_seconds > 0.0
                else None
            ),
            "dataset_exhausted": exhausted,
        },
        "labels": label_report,
        "fields": field_report,
        "errors": errors,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test one real Ali-CCP batch.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--benchmark-batches", type=int)
    parser.add_argument("--warmup-batches", type=int)
    parser.add_argument(
        "--columnar-batching",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Build whole tensor batches from Arrow columns instead of Python rows.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    config = _load_yaml(args.config)
    data_config = config.get("data")
    loader_config = config.get("dataloader")
    if not isinstance(data_config, dict) or not isinstance(loader_config, dict):
        raise SystemExit("config must contain data and dataloader mappings")

    report = run_dataloader_smoke_test(
        processed_dir=args.processed_dir or Path(data_config["processed_dir"]),
        split=args.split,
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else int(loader_config["batch_size"])
        ),
        num_workers=(
            args.num_workers
            if args.num_workers is not None
            else int(loader_config["num_workers"])
        ),
        pin_memory=(
            args.pin_memory
            if args.pin_memory is not None
            else bool(loader_config["pin_memory"])
        ),
        device=args.device,
        embedding_dim=args.embedding_dim,
        benchmark_batches=(
            args.benchmark_batches
            if args.benchmark_batches is not None
            else int(loader_config.get("benchmark_batches", 0))
        ),
        warmup_batches=(
            args.warmup_batches
            if args.warmup_batches is not None
            else int(loader_config.get("benchmark_warmup_batches", 0))
        ),
        columnar_batching=(
            args.columnar_batching
            if args.columnar_batching is not None
            else bool(loader_config.get("columnar_batching", True))
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
