from __future__ import annotations

import importlib.util
import math
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from torch.nn import functional as F

    from src.losses.multitask_loss import (
        ESMMLoss,
        ESMMWithAuxiliaryCVRLoss,
        MaskedCVRMultiTaskLoss,
        build_esmm_loss,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class MaskedCVRMultiTaskLossTest(unittest.TestCase):
    def test_cvr_loss_uses_only_clicked_samples(self) -> None:
        ctr_logit = torch.zeros(3, requires_grad=True)
        cvr_logit = torch.tensor([0.0, 100.0, 0.0], requires_grad=True)
        outputs = {"ctr_logit": ctr_logit, "cvr_logit": cvr_logit}
        batch = {
            "click": torch.tensor([1.0, 0.0, 1.0]),
            "conversion": torch.tensor([1.0, 0.0, 0.0]),
        }

        losses = MaskedCVRMultiTaskLoss()(outputs, batch)
        self.assertAlmostEqual(losses["ctr_loss"].item(), math.log(2), places=6)
        self.assertAlmostEqual(losses["cvr_loss"].item(), math.log(2), places=6)
        self.assertAlmostEqual(
            losses["loss"].item(), 2 * math.log(2), places=6
        )
        self.assertEqual(losses["clicked_count"].item(), 2.0)

        losses["loss"].backward()
        self.assertEqual(cvr_logit.grad[1].item(), 0.0)

    def test_batch_without_clicks_has_zero_cvr_loss(self) -> None:
        ctr_logit = torch.zeros(2, requires_grad=True)
        cvr_logit = torch.tensor([10.0, -10.0], requires_grad=True)
        outputs = {"ctr_logit": ctr_logit, "cvr_logit": cvr_logit}
        batch = {
            "click": torch.zeros(2),
            "conversion": torch.zeros(2),
        }

        losses = MaskedCVRMultiTaskLoss()(outputs, batch)
        self.assertEqual(losses["cvr_loss"].item(), 0.0)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertTrue(torch.equal(cvr_logit.grad, torch.zeros(2)))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ESMMLossTest(unittest.TestCase):
    def test_matches_probability_space_bce_at_regular_logits(self) -> None:
        ctr_logit = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
        cvr_logit = torch.tensor([0.5, -0.5, 1.0], requires_grad=True)
        click = torch.tensor([0.0, 1.0, 1.0])
        ctcvr = torch.tensor([0.0, 0.0, 1.0])
        outputs = {"ctr_logit": ctr_logit, "cvr_logit": cvr_logit}
        batch = {"click": click, "ctcvr": ctcvr}

        losses = ESMMLoss()(outputs, batch)
        expected_ctr = F.binary_cross_entropy_with_logits(ctr_logit, click)
        expected_ctcvr = F.binary_cross_entropy(
            torch.sigmoid(ctr_logit) * torch.sigmoid(cvr_logit),
            ctcvr,
        )

        self.assertTrue(torch.allclose(losses["ctr_loss"], expected_ctr))
        self.assertTrue(torch.allclose(losses["ctcvr_loss"], expected_ctcvr))
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(ctr_logit.grad).all())
        self.assertTrue(torch.isfinite(cvr_logit.grad).all())

    def test_non_clicked_impression_updates_cvr_through_ctcvr(self) -> None:
        ctr_logit = torch.zeros(1, requires_grad=True)
        cvr_logit = torch.zeros(1, requires_grad=True)
        losses = ESMMLoss()(
            {"ctr_logit": ctr_logit, "cvr_logit": cvr_logit},
            {"click": torch.zeros(1), "ctcvr": torch.zeros(1)},
        )

        losses["loss"].backward()
        self.assertGreater(abs(cvr_logit.grad.item()), 0.0)

    def test_extreme_logits_remain_finite(self) -> None:
        losses = ESMMLoss()(
            {
                "ctr_logit": torch.tensor([80.0, -80.0], requires_grad=True),
                "cvr_logit": torch.tensor([80.0, 80.0], requires_grad=True),
            },
            {
                "click": torch.tensor([1.0, 0.0]),
                "ctcvr": torch.tensor([1.0, 0.0]),
            },
        )
        self.assertTrue(torch.isfinite(losses["loss"]))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ESMMWithAuxiliaryCVRLossTest(unittest.TestCase):
    def _inputs(self):
        ctr_logit = torch.zeros(12, requires_grad=True)
        cvr_logit = torch.zeros(12, requires_grad=True)
        click = torch.tensor([1.0] * 10 + [0.0, 0.0])
        conversion = torch.tensor([1.0, 1.0] + [0.0] * 10)
        return (
            {"ctr_logit": ctr_logit, "cvr_logit": cvr_logit},
            {
                "click": click,
                "conversion": conversion,
                "ctcvr": click * conversion,
            },
        )

    def test_training_samples_clicked_negatives_and_keeps_positives(self) -> None:
        outputs, batch = self._inputs()
        criterion = ESMMWithAuxiliaryCVRLoss(
            auxiliary_cvr_weight=0.25,
            negative_to_positive_ratio=2.0,
        )
        torch.manual_seed(7)
        losses = criterion(outputs, batch)

        torch.manual_seed(7)
        expected_random = torch.rand_like(outputs["cvr_logit"])
        negative_mask = (batch["click"] == 1.0) & (
            batch["conversion"] == 0.0
        )
        expected_negative_count = int(
            (negative_mask & (expected_random < 0.5)).sum().item()
        )
        self.assertEqual(losses["auxiliary_cvr_positive_count"].item(), 2)
        self.assertEqual(
            losses["auxiliary_cvr_negative_count"].item(),
            expected_negative_count,
        )
        self.assertEqual(
            losses["auxiliary_cvr_sampled_count"].item(),
            2 + expected_negative_count,
        )
        self.assertAlmostEqual(
            losses["auxiliary_cvr_loss"].item(), math.log(2), places=6
        )
        expected_total = (
            losses["ctr_loss"]
            + losses["ctcvr_loss"]
            + 0.25 * losses["auxiliary_cvr_loss"]
        )
        self.assertTrue(torch.allclose(losses["loss"], expected_total))

    def test_eval_uses_all_clicked_examples(self) -> None:
        outputs, batch = self._inputs()
        criterion = ESMMWithAuxiliaryCVRLoss(
            auxiliary_cvr_weight=0.1,
            negative_to_positive_ratio=2.0,
        )
        criterion.eval()
        losses = criterion(outputs, batch)

        self.assertEqual(losses["auxiliary_cvr_positive_count"].item(), 2)
        self.assertEqual(losses["auxiliary_cvr_negative_count"].item(), 8)
        self.assertEqual(losses["auxiliary_cvr_sampled_count"].item(), 10)

    def test_auxiliary_gradient_excludes_non_clicked_examples(self) -> None:
        outputs, batch = self._inputs()
        criterion = ESMMWithAuxiliaryCVRLoss(
            auxiliary_cvr_weight=0.1,
            negative_to_positive_ratio=100.0,
        )
        losses = criterion(outputs, batch)
        losses["auxiliary_cvr_loss"].backward()

        self.assertTrue(
            torch.equal(
                outputs["cvr_logit"].grad[10:],
                torch.zeros(2),
            )
        )
        self.assertTrue(
            torch.count_nonzero(outputs["cvr_logit"].grad[:10]).item()
            == 10
        )

    def test_batch_without_conversions_has_zero_auxiliary_loss(self) -> None:
        outputs = {
            "ctr_logit": torch.zeros(4, requires_grad=True),
            "cvr_logit": torch.zeros(4, requires_grad=True),
        }
        batch = {
            "click": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "conversion": torch.zeros(4),
            "ctcvr": torch.zeros(4),
        }
        losses = ESMMWithAuxiliaryCVRLoss(
            auxiliary_cvr_weight=0.1,
            negative_to_positive_ratio=20.0,
        )(outputs, batch)

        self.assertEqual(losses["auxiliary_cvr_sampled_count"].item(), 0)
        self.assertEqual(losses["auxiliary_cvr_loss"].item(), 0.0)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_builder_preserves_standard_esmm_by_default(self) -> None:
        standard = build_esmm_loss(
            {"ctr_weight": 1.0, "ctcvr_weight": 1.0}
        )
        disabled = build_esmm_loss(
            {
                "ctr_weight": 1.0,
                "ctcvr_weight": 1.0,
                "auxiliary_cvr_loss": {"enabled": False},
            }
        )
        enabled = build_esmm_loss(
            {
                "ctr_weight": 1.0,
                "ctcvr_weight": 1.0,
                "auxiliary_cvr_loss": {
                    "enabled": True,
                    "weight": 0.1,
                    "negative_to_positive_ratio": 20,
                },
            }
        )

        self.assertIs(type(standard), ESMMLoss)
        self.assertIs(type(disabled), ESMMLoss)
        self.assertIsInstance(enabled, ESMMWithAuxiliaryCVRLoss)


if __name__ == "__main__":
    unittest.main()
