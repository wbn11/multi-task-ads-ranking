"""Shared command-line configuration overrides for training entry points."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import yaml


_TRAINING_ARGUMENTS: dict[str, str] = {
    "seed": "seed",
    "epochs": "epochs",
    "batch_size": "batch_size",
    "num_workers": "num_workers",
    "learning_rate": "learning_rate",
    "weight_decay": "weight_decay",
    "gradient_clip_norm": "gradient_clip_norm",
    "early_stopping_patience": "early_stopping_patience",
    "early_stopping_min_delta": "early_stopping_min_delta",
    "device": "device",
    "amp": "amp",
    "pin_memory": "pin_memory",
    "columnar_batching": "columnar_batching",
}

_MODEL_ARGUMENTS: dict[str, str] = {
    "embedding_dim": "embedding_dim",
    "dropout": "dropout",
    "gate_dropout": "gate_dropout",
    "num_cross_layers": "num_cross_layers",
    "history_attention_hidden_dim": "history_attention_hidden_dim",
    "history_count_prior_strength": "history_count_prior_strength",
}

_LOSS_ARGUMENTS: dict[str, str] = {
    "ctr_weight": "ctr_weight",
    "cvr_weight": "cvr_weight",
    "ctcvr_weight": "ctcvr_weight",
}

_ALLOWED_OVERRIDE_SECTIONS = {"model", "loss", "data", "training", "experiment"}
_PROTECTED_OVERRIDE_PATHS = {"model.name", "experiment.name"}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1)")
    return parsed


def add_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_config: str,
    model_fields: Sequence[str] = (),
    loss_fields: Sequence[str] = (),
    auxiliary_cvr: bool = False,
) -> None:
    """Add consistent file, runtime, model, loss, and sampling overrides.

    Override arguments intentionally default to ``None``. The model YAML is
    the single source of effective defaults; only values explicitly supplied
    on the command line replace it.
    """

    parser.add_argument("--config", default=default_config)
    parser.add_argument("--data-config")
    parser.add_argument("--experiment-name")
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))

    runtime = parser.add_argument_group("training overrides")
    runtime.add_argument("--seed", type=_non_negative_int)
    runtime.add_argument("--epochs", type=_positive_int)
    runtime.add_argument("--batch-size", type=_positive_int)
    runtime.add_argument("--num-workers", type=_non_negative_int)
    runtime.add_argument("--learning-rate", type=_positive_float)
    runtime.add_argument("--weight-decay", type=_non_negative_float)
    runtime.add_argument("--gradient-clip-norm", type=_positive_float)
    runtime.add_argument("--early-stopping-patience", type=_non_negative_int)
    runtime.add_argument("--early-stopping-min-delta", type=_non_negative_float)
    runtime.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=None
    )
    runtime.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=None
    )
    runtime.add_argument(
        "--columnar-batching",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    model = parser.add_argument_group("model overrides")
    unknown_model_fields = set(model_fields) - set(_MODEL_ARGUMENTS)
    if unknown_model_fields:
        raise ValueError(
            f"unsupported model override fields: {unknown_model_fields}"
        )
    if "embedding_dim" in model_fields:
        model.add_argument("--embedding-dim", type=_positive_int)
    if "dropout" in model_fields:
        model.add_argument("--dropout", type=_probability)
    if "gate_dropout" in model_fields:
        model.add_argument("--gate-dropout", type=_probability)
    if "num_cross_layers" in model_fields:
        model.add_argument("--num-cross-layers", type=_positive_int)
    if "history_attention_hidden_dim" in model_fields:
        model.add_argument("--history-attention-hidden-dim", type=_positive_int)
    if "history_count_prior_strength" in model_fields:
        model.add_argument(
            "--history-count-prior-strength", type=_non_negative_float
        )

    loss = parser.add_argument_group("loss overrides")
    unknown_loss_fields = set(loss_fields) - set(_LOSS_ARGUMENTS)
    if unknown_loss_fields:
        raise ValueError(
            f"unsupported loss override fields: {unknown_loss_fields}"
        )
    if "ctr_weight" in loss_fields:
        loss.add_argument(
            "--ctr-loss-weight", dest="ctr_weight", type=_positive_float
        )
    if "cvr_weight" in loss_fields:
        loss.add_argument(
            "--cvr-loss-weight", dest="cvr_weight", type=_positive_float
        )
    if "ctcvr_weight" in loss_fields:
        loss.add_argument(
            "--ctcvr-loss-weight", dest="ctcvr_weight", type=_positive_float
        )
    if auxiliary_cvr:
        loss.add_argument(
            "--auxiliary-cvr-weight",
            type=_non_negative_float,
            help="0 disables the auxiliary clicked-space CVR loss",
        )
        loss.add_argument(
            "--auxiliary-cvr-negative-ratio", type=_positive_float
        )

    sampling = parser.add_argument_group("exposure negative sampling")
    sampling_mode = sampling.add_mutually_exclusive_group()
    sampling_mode.add_argument(
        "--negative-sampling-ratio",
        type=_positive_float,
        help="keep every click and resample this many non-clicks per click",
    )
    sampling_mode.add_argument(
        "--no-negative-sampling",
        action="store_true",
        help="force full-exposure training even if the YAML enables sampling",
    )

    parser.add_argument(
        "--set",
        dest="config_assignments",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help=(
            "advanced YAML override; repeat as needed, for example "
            "--set 'model.hidden_dims=[512,256,128]'"
        ),
    )


def _set_nested(target: dict[str, Any], path: Sequence[str], value: Any) -> None:
    current = target
    for key in path[:-1]:
        child = current.get(key)
        if child is None:
            child = {}
            current[key] = child
        if not isinstance(child, dict):
            raise ValueError(
                f"cannot assign {'.'.join(path)!r}: {key!r} is not a mapping"
            )
        current = child
    current[path[-1]] = value


def parse_config_assignments(assignments: Sequence[str]) -> dict[str, Any]:
    """Parse repeatable ``SECTION.KEY=YAML_VALUE`` assignments."""

    overrides: dict[str, Any] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(
                f"invalid --set value {assignment!r}; expected SECTION.KEY=VALUE"
            )
        raw_path, raw_value = assignment.split("=", 1)
        path = [part.strip() for part in raw_path.split(".")]
        if len(path) < 2 or any(not part for part in path):
            raise ValueError(
                f"invalid --set path {raw_path!r}; expected SECTION.KEY"
            )
        dotted_path = ".".join(path)
        if path[0] not in _ALLOWED_OVERRIDE_SECTIONS:
            raise ValueError(f"unsupported --set section: {path[0]!r}")
        if dotted_path in _PROTECTED_OVERRIDE_PATHS:
            raise ValueError(
                f"{dotted_path} cannot be changed with --set; choose a matching "
                "base config or use --experiment-name"
            )
        value = yaml.safe_load(raw_value)
        _set_nested(overrides, path, value)
    return overrides


def build_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate parsed convenience flags into nested YAML overrides."""

    overrides = parse_config_assignments(args.config_assignments)
    for destination, key in _TRAINING_ARGUMENTS.items():
        value = getattr(args, destination, None)
        if value is not None:
            _set_nested(overrides, ("training", key), value)
    if getattr(args, "seed", None) is not None:
        _set_nested(
            overrides,
            ("training", "negative_sampling", "seed"),
            args.seed,
        )
    for destination, key in _MODEL_ARGUMENTS.items():
        value = getattr(args, destination, None)
        if value is not None:
            _set_nested(overrides, ("model", key), value)
    for destination, key in _LOSS_ARGUMENTS.items():
        value = getattr(args, destination, None)
        if value is not None:
            _set_nested(overrides, ("loss", key), value)

    negative_ratio = getattr(args, "negative_sampling_ratio", None)
    if negative_ratio is not None:
        _set_nested(
            overrides,
            ("training", "negative_sampling", "enabled"),
            True,
        )
        _set_nested(
            overrides,
            ("training", "negative_sampling", "negative_to_positive_ratio"),
            negative_ratio,
        )
    elif getattr(args, "no_negative_sampling", False):
        _set_nested(
            overrides,
            ("training", "negative_sampling", "enabled"),
            False,
        )

    auxiliary_weight = getattr(args, "auxiliary_cvr_weight", None)
    if auxiliary_weight is not None:
        _set_nested(
            overrides,
            ("loss", "auxiliary_cvr_loss", "enabled"),
            auxiliary_weight > 0.0,
        )
        _set_nested(
            overrides,
            ("loss", "auxiliary_cvr_loss", "weight"),
            auxiliary_weight,
        )
    auxiliary_ratio = getattr(args, "auxiliary_cvr_negative_ratio", None)
    if auxiliary_ratio is not None:
        _set_nested(
            overrides,
            ("loss", "auxiliary_cvr_loss", "negative_to_positive_ratio"),
            auxiliary_ratio,
        )
    return overrides


def merge_config_overrides(
    base: Mapping[str, Any], overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Recursively merge overrides without mutating the loaded YAML mapping."""

    merged = deepcopy(dict(base))
    if not overrides:
        return merged
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_config_overrides(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged
