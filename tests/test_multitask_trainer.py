from __future__ import annotations

import importlib.util
import logging
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from torch import nn

    from src.losses.multitask_loss import (
        ESMMLoss,
        ESMMWithAuxiliaryCVRLoss,
    )
    from src.trainer.dcn_ple_esmm_trainer import DCNPLEESMMTrainer
    from src.trainer.esmm_trainer import ESMMTrainer
    from src.trainer.expert_gate_trainer import ExpertGateTrainer
    from src.trainer.mmoe_trainer import MMoETrainer
    from src.trainer.multitask_trainer_base import MultiTaskTrainerBase
    from src.trainer.ple_trainer import PLETrainer
    from src.trainer.ple_esmm_trainer import PLEESMMTrainer


def make_logger() -> logging.Logger:
    logger = logging.getLogger("test_multitask_trainer")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


if TORCH_AVAILABLE:

    class AnchorModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            return {"anchor": self.anchor}


    class PLEAnchorModel(AnchorModel):
        num_shared_experts = 2
        num_task_experts = 1


    class KnownComponentLoss(nn.Module):
        def forward(self, outputs, batch):
            anchor = outputs["anchor"] * 0.0
            ctr_loss = anchor + batch["click"].mean()
            cvr_loss = anchor + batch["conversion"].sum()
            return {
                "loss": ctr_loss + cvr_loss,
                "ctr_loss": ctr_loss,
                "cvr_loss": cvr_loss,
            }


    class KnownComponentCountLoss(KnownComponentLoss):
        def forward(self, outputs, batch):
            losses = super().forward(outputs, batch)
            losses["selected_count"] = batch["conversion"].sum()
            return losses


def make_batch(click: list[float], conversion: list[float]):
    click_tensor = torch.tensor(click)
    conversion_tensor = torch.tensor(conversion)
    return {
        "features": {},
        "click": click_tensor,
        "conversion": conversion_tensor,
        "ctcvr": click_tensor * conversion_tensor,
    }


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class MultiTaskTrainerAggregationTest(unittest.TestCase):
    def test_components_use_their_declared_sample_spaces(self) -> None:
        model = AnchorModel()
        trainer = MultiTaskTrainerBase(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            criterion=KnownComponentLoss(),
            loss_component_scopes={
                "ctr_loss": "all",
                "cvr_loss": "clicked",
            },
            loss_component_weights={"ctr_loss": 1.0, "cvr_loss": 1.0},
            device=torch.device("cpu"),
            amp=False,
            gradient_clip_norm=None,
            logger=make_logger(),
        )
        loader = [
            make_batch([1.0, 0.0], [1.0, 0.0]),
            make_batch([1.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
        ]

        metrics = trainer.train_epoch(loader)

        self.assertAlmostEqual(metrics["ctr_loss"], 3.0 / 5.0, places=6)
        self.assertAlmostEqual(metrics["cvr_loss"], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(metrics["loss"], 14.0 / 15.0, places=6)
        self.assertEqual(metrics["samples"], 5.0)
        self.assertEqual(metrics["clicked_samples"], 3.0)

    def test_component_can_use_a_criterion_reported_normalizer(self) -> None:
        model = AnchorModel()
        trainer = MultiTaskTrainerBase(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            criterion=KnownComponentCountLoss(),
            loss_component_scopes={
                "ctr_loss": "all",
                "cvr_loss": "clicked",
            },
            loss_component_weights={"ctr_loss": 1.0, "cvr_loss": 1.0},
            loss_component_count_metrics={"cvr_loss": "selected_count"},
            loss_count_metrics=("selected_count",),
            device=torch.device("cpu"),
            amp=False,
            gradient_clip_norm=None,
            logger=make_logger(),
        )
        loader = [
            make_batch([1.0, 0.0], [1.0, 0.0]),
            make_batch([1.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
        ]

        metrics = trainer.train_epoch(loader)

        self.assertAlmostEqual(metrics["cvr_loss"], 1.0, places=6)
        self.assertEqual(metrics["selected_count"], 1.0)

    def test_esmm_trainer_uses_entire_space_ctcvr_loss(self) -> None:
        model = AnchorModel()
        trainer = ESMMTrainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            device=torch.device("cpu"),
            amp=False,
            gradient_clip_norm=None,
            logger=make_logger(),
            loss_config={"ctr_weight": 1.0, "ctcvr_weight": 2.0},
        )

        self.assertIsInstance(trainer.criterion, ESMMLoss)
        self.assertEqual(
            trainer.loss_component_scopes,
            {"ctr_loss": "all", "ctcvr_loss": "all"},
        )
        self.assertEqual(trainer.loss_component_weights["ctcvr_loss"], 2.0)

    def test_esmm_trainer_configures_sampled_auxiliary_cvr_loss(self) -> None:
        model = AnchorModel()
        trainer = ESMMTrainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            device=torch.device("cpu"),
            amp=False,
            gradient_clip_norm=None,
            logger=make_logger(),
            loss_config={
                "ctr_weight": 1.0,
                "ctcvr_weight": 1.0,
                "auxiliary_cvr_loss": {
                    "enabled": True,
                    "weight": 0.1,
                    "negative_to_positive_ratio": 20.0,
                },
            },
        )

        self.assertIsInstance(
            trainer.criterion,
            ESMMWithAuxiliaryCVRLoss,
        )
        self.assertEqual(
            trainer.loss_component_scopes["auxiliary_cvr_loss"],
            "clicked",
        )
        self.assertEqual(
            trainer.loss_component_count_metrics,
            {"auxiliary_cvr_loss": "auxiliary_cvr_sampled_count"},
        )
        self.assertEqual(
            trainer.loss_count_metrics,
            (
                "auxiliary_cvr_positive_count",
                "auxiliary_cvr_negative_count",
                "auxiliary_cvr_sampled_count",
            ),
        )

    def test_dcn_ple_esmm_combines_esmm_loss_and_gate_diagnostics(self) -> None:
        model = PLEAnchorModel()
        trainer = DCNPLEESMMTrainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            device=torch.device("cpu"),
            amp=False,
            gradient_clip_norm=None,
            logger=make_logger(),
            loss_config={"ctr_weight": 1.0, "ctcvr_weight": 2.0},
        )

        self.assertIsInstance(trainer.criterion, ESMMLoss)
        state = trainer._create_evaluation_diagnostics()
        trainer._update_evaluation_diagnostics(
            state,
            {
                "ctr_gate_weights": torch.tensor([[0.2, 0.5, 0.3]]),
                "cvr_gate_weights": torch.tensor([[0.4, 0.4, 0.2]]),
            },
        )
        metrics = trainer._finalize_evaluation_diagnostics(state)
        self.assertIn("gates", metrics)
        self.assertEqual(
            metrics["gate_expert_order"]["ctr"],
            ["shared_1", "shared_2", "ctr_specific_1"],
        )

    def test_expert_gate_trainer_aggregates_gate_diagnostics(self) -> None:
        model = AnchorModel()
        trainer = ExpertGateTrainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            device=torch.device("cpu"),
            amp=False,
            gradient_clip_norm=None,
            logger=make_logger(),
            loss_config={"ctr_weight": 1.0, "cvr_weight": 1.0},
        )
        state = trainer._create_evaluation_diagnostics()
        trainer._update_evaluation_diagnostics(
            state,
            {
                "ctr_gate_weights": torch.tensor([[0.8, 0.2]]),
                "cvr_gate_weights": torch.tensor([[0.3, 0.7]]),
            },
        )

        metrics = trainer._finalize_evaluation_diagnostics(state)

        self.assertEqual(metrics["gates"]["ctr"]["samples"], 1.0)
        self.assertEqual(metrics["gates"]["cvr"]["top1_fractions"], [0.0, 1.0])

    def test_model_specific_trainers_share_gate_behavior(self) -> None:
        self.assertTrue(issubclass(MMoETrainer, ExpertGateTrainer))
        self.assertTrue(issubclass(PLETrainer, ExpertGateTrainer))

    def test_ple_esmm_trainer_combines_esmm_loss_and_gate_diagnostics(self) -> None:
        model = PLEAnchorModel()
        trainer = PLEESMMTrainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            device=torch.device("cpu"),
            amp=False,
            gradient_clip_norm=None,
            logger=make_logger(),
            loss_config={"ctr_weight": 1.0, "ctcvr_weight": 1.0},
        )

        self.assertIsInstance(trainer.criterion, ESMMLoss)
        state = trainer._create_evaluation_diagnostics()
        trainer._update_evaluation_diagnostics(
            state,
            {
                "ctr_gate_weights": torch.tensor([[0.2, 0.5, 0.3]]),
                "cvr_gate_weights": torch.tensor([[0.4, 0.4, 0.2]]),
            },
        )
        metrics = trainer._finalize_evaluation_diagnostics(state)
        self.assertEqual(
            metrics["gate_expert_order"]["cvr"],
            ["shared_1", "shared_2", "cvr_specific_1"],
        )

    def test_ple_trainer_reports_gate_expert_order(self) -> None:
        model = PLEAnchorModel()
        trainer = PLETrainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            device=torch.device("cpu"),
            amp=False,
            gradient_clip_norm=None,
            logger=make_logger(),
            loss_config={"ctr_weight": 1.0, "cvr_weight": 1.0},
        )
        state = trainer._create_evaluation_diagnostics()
        trainer._update_evaluation_diagnostics(
            state,
            {
                "ctr_gate_weights": torch.tensor([[0.2, 0.5, 0.3]]),
                "cvr_gate_weights": torch.tensor([[0.4, 0.4, 0.2]]),
            },
        )

        metrics = trainer._finalize_evaluation_diagnostics(state)

        self.assertEqual(
            metrics["gate_expert_order"]["ctr"],
            ["shared_1", "shared_2", "ctr_specific_1"],
        )
        self.assertEqual(
            metrics["gate_expert_order"]["cvr"],
            ["shared_1", "shared_2", "cvr_specific_1"],
        )


if __name__ == "__main__":
    unittest.main()
