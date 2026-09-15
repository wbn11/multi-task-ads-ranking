"""Overfit one fixed Ali-CCP batch to validate the DCNv2 training path."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torch import Tensor, nn

from src.data.dataloader import create_dataloader
from src.data.dataset import AdsDataset
from src.data.feature_encoder import FeatureEncoder
from src.models.dcnv2 import DCNv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dcnv2.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return payload


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    non_blocking = device.type == "cuda"
    features = {
        field_id: {
            name: tensor.to(device, non_blocking=non_blocking)
            for name, tensor in field.items()
        }
        for field_id, field in batch["features"].items()
    }
    return {
        "features": features,
        "click": batch["click"].to(device, non_blocking=non_blocking),
        "conversion": batch["conversion"].to(
            device, non_blocking=non_blocking
        ),
        "ctcvr": batch["ctcvr"].to(device, non_blocking=non_blocking),
    }


def finite_gradients(model: nn.Module) -> bool:
    checks = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return bool(checks and torch.stack(checks).all())


def mean_for_label(probabilities: Tensor, labels: Tensor, label: int) -> float:
    selected = probabilities[labels == float(label)]
    return float(selected.mean().item()) if selected.numel() else math.nan


def main() -> None:
    args = parse_args()
    config = load_yaml(project_path(args.config))
    model_config = config["model"]
    if str(model_config["name"]).lower() != "dcnv2":
        raise ValueError("overfit_dcnv2.py requires model.name=dcnv2")
    overfit_config = config["overfit"]

    data_config = load_yaml(project_path(config["data"]["config"]))["data"]
    processed_dir = project_path(data_config["processed_dir"])
    split = str(config["data"].get("split", "train"))
    batch_size = int(
        args.batch_size if args.batch_size is not None else overfit_config["batch_size"]
    )
    steps = int(args.steps if args.steps is not None else overfit_config["steps"])
    requested_device = str(args.device or overfit_config["device"])
    seed = int(overfit_config["seed"])
    minimum_loss_reduction = float(overfit_config["minimum_loss_reduction"])

    if batch_size <= 1:
        raise ValueError("batch_size must be greater than one")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 < minimum_loss_reduction < 1.0:
        raise ValueError("minimum_loss_reduction must be between zero and one")

    set_seed(seed)
    device = resolve_device(requested_device)
    dataset = AdsDataset(processed_dir, split=split)
    try:
        loader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(overfit_config["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        batch = move_batch(next(iter(loader)), device)
        labels = batch["click"]
        positives = int(labels.sum().item())
        negatives = int(labels.numel() - positives)
        if positives == 0 or negatives == 0:
            raise RuntimeError(
                "the fixed batch must contain both positive and negative click labels"
            )

        encoder = FeatureEncoder.load(processed_dir / "vocab.json")
        model = DCNv2.from_feature_encoder(
            encoder,
            embedding_dim=int(model_config["embedding_dim"]),
            num_cross_layers=int(model_config["num_cross_layers"]),
            hidden_dims=tuple(int(value) for value in model_config["hidden_dims"]),
            dropout=float(model_config["dropout"]),
            embedding_pooling=str(model_config["embedding_pooling"]),
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        learning_rate = float(overfit_config["learning_rate"])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=float(overfit_config["weight_decay"]),
        )

        model.eval()
        with torch.no_grad():
            initial_output = model(batch)
            initial_loss = float(
                criterion(initial_output["ctr_logit"], labels).item()
            )

        trace_steps = {1, steps}
        trace_steps.update(
            range(max(1, steps // 4), steps + 1, max(1, steps // 4))
        )
        loss_trace: list[dict[str, float | int]] = []
        gradients_are_finite = True
        model.train()
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = criterion(output["ctr_logit"], labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            gradients_are_finite = gradients_are_finite and finite_gradients(model)
            if not gradients_are_finite:
                raise RuntimeError(f"non-finite gradient at step {step}")
            optimizer.step()
            if step in trace_steps:
                loss_trace.append({"step": step, "loss": float(loss.item())})

        model.eval()
        with torch.no_grad():
            final_output = model(batch)
            final_loss = float(criterion(final_output["ctr_logit"], labels).item())
            probabilities = final_output["ctr"]

        loss_reduction = (initial_loss - final_loss) / initial_loss
        positive_mean = mean_for_label(probabilities, labels, 1)
        negative_mean = mean_for_label(probabilities, labels, 0)
        valid = bool(
            math.isfinite(final_loss)
            and gradients_are_finite
            and loss_reduction >= minimum_loss_reduction
            and positive_mean > negative_mean
        )
        report = {
            "valid": valid,
            "model": "dcnv2",
            "target": "click",
            "device": str(device),
            "split": split,
            "dataset_size": len(dataset),
            "batch_size": int(labels.numel()),
            "positive_labels": positives,
            "negative_labels": negatives,
            "field_count": len(model.field_ids),
            "embedding_dim": model.embedding_dim,
            "embedding_pooling": model.embedding_pooling,
            "num_cross_layers": model.num_cross_layers,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "steps": steps,
            "learning_rate": learning_rate,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction_fraction": loss_reduction,
            "minimum_loss_reduction": minimum_loss_reduction,
            "positive_prediction_mean": positive_mean,
            "negative_prediction_mean": negative_mean,
            "gradients_finite": gradients_are_finite,
            "loss_trace": loss_trace,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not valid:
            raise SystemExit(1)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
