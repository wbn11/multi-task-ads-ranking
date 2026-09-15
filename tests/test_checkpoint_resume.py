from __future__ import annotations

import importlib.util
import logging
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from torch import nn

    from src.trainer.ctr_trainer_base import CTRTrainerBase
    from src.trainer.esmm_trainer import ESMMTrainer


def make_logger() -> logging.Logger:
    logger = logging.getLogger("test_checkpoint_resume")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


if TORCH_AVAILABLE:

    class TinyCTR(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logit = nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            logits = self.logit.expand_as(batch["click"])
            return {"ctr_logit": logits, "ctr": torch.sigmoid(logits)}


    class TinyESMM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ctr_logit = nn.Parameter(torch.zeros(()))
            self.cvr_logit = nn.Parameter(torch.zeros(()))

        def forward(self, batch):
            ctr_logit = self.ctr_logit.expand_as(batch["click"])
            cvr_logit = self.cvr_logit.expand_as(batch["click"])
            ctr = torch.sigmoid(ctr_logit)
            cvr = torch.sigmoid(cvr_logit)
            return {
                "ctr_logit": ctr_logit,
                "cvr_logit": cvr_logit,
                "ctr": ctr,
                "cvr": cvr,
                "ctcvr": ctr * cvr,
            }


def make_batch():
    click = torch.tensor([0.0, 1.0, 1.0, 0.0])
    conversion = torch.tensor([0.0, 0.0, 1.0, 0.0])
    return {
        "features": {},
        "click": click,
        "conversion": conversion,
        "ctcvr": click * conversion,
        "user_group_id": torch.tensor([7, 7, 7, 7], dtype=torch.int64),
    }


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class CheckpointResumeTest(unittest.TestCase):
    def test_ctr_resume_continues_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch = make_batch()
            first_model = TinyCTR()
            first = CTRTrainerBase(
                model=first_model,
                optimizer=torch.optim.SGD(first_model.parameters(), lr=0.1),
                device=torch.device("cpu"),
                amp=False,
                gradient_clip_norm=None,
                logger=make_logger(),
            )
            first.fit(
                train_loader=[batch],
                validation_loader=[batch],
                epochs=1,
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
                checkpoint_path=root / "best.pt",
                latest_checkpoint_path=root / "latest.pt",
            )

            resumed_model = TinyCTR()
            resumed = CTRTrainerBase(
                model=resumed_model,
                optimizer=torch.optim.SGD(resumed_model.parameters(), lr=0.1),
                device=torch.device("cpu"),
                amp=False,
                gradient_clip_norm=None,
                logger=make_logger(),
            )
            result = resumed.fit(
                train_loader=[batch],
                validation_loader=[batch],
                epochs=2,
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
                checkpoint_path=root / "best.pt",
                latest_checkpoint_path=root / "latest.pt",
                resume_checkpoint_path=root / "latest.pt",
            )

            self.assertEqual(result["resumed_from_epoch"], 1)
            self.assertEqual(result["epochs_completed"], 2)
            latest = torch.load(root / "latest.pt", weights_only=False)
            self.assertEqual(latest["epoch"], 2)

    def test_multitask_resume_continues_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            batch = make_batch()
            first_model = TinyESMM()
            first = ESMMTrainer(
                model=first_model,
                optimizer=torch.optim.SGD(first_model.parameters(), lr=0.1),
                device=torch.device("cpu"),
                amp=False,
                gradient_clip_norm=None,
                logger=make_logger(),
                loss_config={"ctr_weight": 1.0, "ctcvr_weight": 1.0},
            )
            first.fit(
                train_loader=[batch],
                validation_loader=[batch],
                epochs=1,
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
                selection_task="ctcvr",
                selection_metric="auc",
                checkpoint_path=root / "best.pt",
                latest_checkpoint_path=root / "latest.pt",
            )

            resumed_model = TinyESMM()
            resumed = ESMMTrainer(
                model=resumed_model,
                optimizer=torch.optim.SGD(resumed_model.parameters(), lr=0.1),
                device=torch.device("cpu"),
                amp=False,
                gradient_clip_norm=None,
                logger=make_logger(),
                loss_config={"ctr_weight": 1.0, "ctcvr_weight": 1.0},
            )
            result = resumed.fit(
                train_loader=[batch],
                validation_loader=[batch],
                epochs=2,
                early_stopping_patience=2,
                early_stopping_min_delta=0.0,
                selection_task="ctcvr",
                selection_metric="auc",
                checkpoint_path=root / "best.pt",
                latest_checkpoint_path=root / "latest.pt",
                resume_checkpoint_path=root / "latest.pt",
            )

            self.assertEqual(result["resumed_from_epoch"], 1)
            self.assertEqual(result["epochs_completed"], 2)


if __name__ == "__main__":
    unittest.main()
