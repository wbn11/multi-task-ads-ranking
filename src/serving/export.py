"""Export a trusted training checkpoint as a self-contained TorchScript ranker."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.data.dataset import collate_ads_batch
from src.data.feature_encoder import FeatureEncoder
from src.data.feature_schema import ALL_FIELD_IDS
from src.models.factory import build_multitask_model


EXPORT_FORMAT_VERSION = 1


class _InferenceHeads(nn.Module):
    """Expose only the three prediction tensors to the traced graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, features: dict[str, dict[str, Tensor]]
    ) -> tuple[Tensor, Tensor, Tensor]:
        output = self.model({"features": features})
        return output["ctr"], output["cvr"], output["ctcvr"]


def _probe_features(
    batch_size: int,
    *,
    varied: bool,
    known_ids: Mapping[str, int] | None = None,
) -> dict[str, dict[str, Tensor]]:
    samples: list[dict[str, Any]] = []
    for index in range(batch_size):
        fields: dict[str, dict[str, list[int] | list[float]]] = {}
        if varied:
            user_id = 1 if known_ids is None else known_ids.get("101", 1)
            item_id = 1 if known_ids is None else known_ids.get("205", 1)
            fields["101"] = {"ids": [user_id], "values": [1.0]}
            fields["205"] = {
                "ids": [item_id, 1] if index % 2 else [item_id],
                "values": [0.5, 1.5] if index % 2 else [1.0],
            }
        samples.append(
            {
                "sample_id": str(index),
                "common_feature_id": "probe",
                "user_id": "probe",
                "features": fields,
                "click": 0,
                "conversion": 0,
                "ctcvr": 0,
            }
        )
    return collate_ads_batch(samples)["features"]


def _check_parity(
    eager: nn.Module,
    scripted: torch.jit.ScriptModule,
    *,
    device: torch.device,
    known_ids: Mapping[str, int],
) -> float:
    maximum_error = 0.0
    for count in (1, 3):
        features = {
            field: {name: value.to(device) for name, value in tensors.items()}
            for field, tensors in _probe_features(
                count, varied=True, known_ids=known_ids
            ).items()
        }
        with torch.inference_mode():
            expected = eager(features)
            actual = scripted(features)
        for expected_head, actual_head in zip(expected, actual, strict=True):
            error = float((expected_head - actual_head).abs().max().item())
            maximum_error = max(maximum_error, error)
            if not torch.allclose(expected_head, actual_head, rtol=1e-5, atol=1e-6):
                raise RuntimeError(
                    f"TorchScript parity failed for batch_size={count}: "
                    f"maximum absolute error={error}"
                )
    return maximum_error


def export_ranker(
    *,
    run_directory: str | Path,
    processed_directory: str | Path,
    output_directory: str | Path,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Export one full-exposure DCN-PLE-ESMM run; never overwrite artifacts."""

    run_dir = Path(run_directory).resolve()
    processed_dir = Path(processed_directory).resolve()
    output_dir = Path(output_directory).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite export: {output_dir}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device_name must be cpu or cuda")
    device = torch.device(device_name)

    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "best.pt"
    vocab_path = processed_dir / "vocab.json"
    metrics_path = run_dir / "metrics.json"
    for path in (config_path, checkpoint_path, vocab_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"required export input is missing: {path}")

    import yaml

    saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = saved_config["model_config"]["model"]
    if model_config.get("name") != "dcn_ple_esmm":
        raise ValueError("this serving export supports only dcn_ple_esmm")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    keep_probability = float(metrics["negative_sampling"]["negative_keep_probability"])
    if keep_probability != 1.0:
        raise ValueError(
            "this export expects full-exposure training; sampled runs need "
            "explicit prior-correction serving support"
        )
    encoder = FeatureEncoder.load(vocab_path)
    known_ids = {
        field: 2 if encoder.field_vocabularies[field].vocab_size > 2 else 1
        for field in ("101", "205")
    }
    model = build_multitask_model(encoder, model_config).to(device).eval()
    # Training checkpoints include optimizer and RNG state. Load only checkpoints
    # created by this project in a trusted environment.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    eager = _InferenceHeads(model).eval()
    example = {
        field: {name: value.to(device) for name, value in tensors.items()}
        for field, tensors in _probe_features(2, varied=False).items()
    }
    with torch.inference_mode():
        scripted = torch.jit.trace(eager, (example,), strict=False, check_trace=False)
    max_error = _check_parity(eager, scripted, device=device, known_ids=known_ids)

    calibration_path = run_dir / "multitask_calibrators.json"
    calibration: dict[str, Any] | None = None
    if calibration_path.is_file():
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        if (
            payload.get("fit_split") == "validation"
            and payload.get("input_stage") == "raw_heads"
            and payload.get("negative_keep_probability") == 1.0
            and not payload.get("test_used_for_fit")
        ):
            states = payload.get("methods", {}).get("platt")
            if isinstance(states, Mapping) and {"ctr", "cvr"} <= set(states):
                calibration = {"method": "platt", "fit_split": "validation", "heads": states}

    output_dir.mkdir(parents=True)
    torch.jit.save(scripted, str(output_dir / "ranker.ts"))
    # Verify the serialized graph too, not just the in-memory trace.
    reloaded = torch.jit.load(str(output_dir / "ranker.ts"), map_location=device).eval()
    max_error = max(
        max_error,
        _check_parity(eager, reloaded, device=device, known_ids=known_ids),
    )
    shutil.copy2(vocab_path, output_dir / "vocab.json")
    if calibration is not None:
        (output_dir / "platt.json").write_text(
            json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    manifest = {
        "format_version": EXPORT_FORMAT_VERSION,
        "model_name": "dcn_ple_esmm",
        "model_version": run_dir.name,
        "field_ids": list(ALL_FIELD_IDS),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "parameter_count": metrics.get("parameter_count"),
        "probability_identity": "ctcvr=ctr*cvr",
        "raw_score_modes": ["ctr", "ctcvr"],
        "platt_available": calibration is not None,
        "torch_version": torch.__version__,
        "trace_max_absolute_error": max_error,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"valid": True, "output_directory": str(output_dir), **manifest}
