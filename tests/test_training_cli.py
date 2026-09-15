"""Tests for YAML-backed training command-line overrides."""

from __future__ import annotations

import argparse
import unittest

from src.trainer.cli import (
    add_training_arguments,
    build_config_overrides,
    merge_config_overrides,
    parse_config_assignments,
)


class TrainingCLITest(unittest.TestCase):
    def make_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        add_training_arguments(
            parser,
            default_config="configs/dcn_ple_esmm.yaml",
            model_fields=(
                "embedding_dim",
                "dropout",
                "gate_dropout",
                "num_cross_layers",
            ),
            loss_fields=("ctr_weight", "ctcvr_weight"),
            auxiliary_cvr=True,
        )
        return parser

    def test_no_overrides_keeps_yaml_as_the_only_default_source(self) -> None:
        args = self.make_parser().parse_args([])

        self.assertEqual(args.config, "configs/dcn_ple_esmm.yaml")
        self.assertEqual(build_config_overrides(args), {})

    def test_convenience_flags_build_nested_overrides(self) -> None:
        args = self.make_parser().parse_args(
            [
                "--seed",
                "2027",
                "--device",
                "cuda",
                "--learning-rate",
                "0.0005",
                "--dropout",
                "0.2",
                "--gate-dropout",
                "0.1",
                "--negative-sampling-ratio",
                "5",
                "--auxiliary-cvr-weight",
                "0.02",
                "--auxiliary-cvr-negative-ratio",
                "20",
            ]
        )

        self.assertEqual(
            build_config_overrides(args),
            {
                "training": {
                    "seed": 2027,
                    "device": "cuda",
                    "learning_rate": 0.0005,
                    "negative_sampling": {
                        "enabled": True,
                        "negative_to_positive_ratio": 5.0,
                        "seed": 2027,
                    },
                },
                "model": {"dropout": 0.2, "gate_dropout": 0.1},
                "loss": {
                    "auxiliary_cvr_loss": {
                        "enabled": True,
                        "weight": 0.02,
                        "negative_to_positive_ratio": 20.0,
                    }
                },
            },
        )

    def test_advanced_set_parses_yaml_values(self) -> None:
        overrides = parse_config_assignments(
            [
                "model.hidden_dims=[512, 256, 128]",
                "training.amp=false",
            ]
        )

        self.assertEqual(overrides["model"]["hidden_dims"], [512, 256, 128])
        self.assertFalse(overrides["training"]["amp"])

    def test_recursive_merge_does_not_mutate_base(self) -> None:
        base = {
            "model": {"name": "dcn_ple_esmm", "dropout": 0.2},
            "training": {
                "seed": 2026,
                "negative_sampling": {"enabled": False, "seed": 2026},
            },
        }
        merged = merge_config_overrides(
            base,
            {
                "training": {
                    "negative_sampling": {
                        "enabled": True,
                        "negative_to_positive_ratio": 10.0,
                    }
                }
            },
        )

        self.assertFalse(base["training"]["negative_sampling"]["enabled"])
        self.assertEqual(
            merged["training"]["negative_sampling"],
            {
                "enabled": True,
                "seed": 2026,
                "negative_to_positive_ratio": 10.0,
            },
        )

    def test_model_name_cannot_be_replaced_by_advanced_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be changed"):
            parse_config_assignments(["model.name=lr"])


if __name__ == "__main__":
    unittest.main()
