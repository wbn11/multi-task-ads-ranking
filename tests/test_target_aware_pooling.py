from __future__ import annotations

import importlib.util
import unittest
from collections import Counter


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.data.dataset import collate_ads_batch
    from src.data.feature_encoder import FeatureEncoder
    from src.data.feature_schema import ALL_FIELD_IDS
    from src.layers.target_aware_pooling import TargetAwareHistoryPooling
    from src.models.target_aware_dcn_ple_esmm import (
        HISTORY_TARGET_FIELD_PAIRS,
        TargetAwareDCNPLEESMM,
    )
    from src.models.factory import build_multitask_model


def make_history_batch():
    samples = [
        {
            "sample_id": "1",
            "common_feature_id": "c1",
            "user_id": "u1",
            "features": {
                "109_14": {"ids": [2, 3], "values": [3.0, 1.0]},
                "110_14": {"ids": [2, 4], "values": [1.0, 2.0]},
                "127_14": {"ids": [3, 4], "values": [2.0, 1.0]},
                "150_14": {"ids": [2, 3], "values": [1.0, 1.0]},
                "206": {"ids": [2], "values": [1.0]},
                "207": {"ids": [4], "values": [1.0]},
                "216": {"ids": [3], "values": [1.0]},
                "210": {"ids": [2], "values": [1.0]},
            },
            "click": 1,
            "conversion": 1,
            "ctcvr": 1,
        },
        {
            "sample_id": "2",
            "common_feature_id": "c2",
            "user_id": "u2",
            "features": {
                "109_14": {"ids": [3, 2], "values": [1.0, 2.0]},
                "110_14": {"ids": [4, 2], "values": [2.0, 1.0]},
                "127_14": {"ids": [4, 3], "values": [1.0, 3.0]},
                "150_14": {"ids": [3, 2], "values": [1.0, 1.0]},
                "206": {"ids": [3], "values": [1.0]},
                "207": {"ids": [2], "values": [1.0]},
                "216": {"ids": [4], "values": [1.0]},
                "210": {"ids": [3], "values": [1.0]},
            },
            "click": 0,
            "conversion": 0,
            "ctcvr": 0,
        },
    ]
    return collate_ads_batch(samples)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TargetAwareHistoryPoolingTest(unittest.TestCase):
    def test_initialization_matches_count_weighted_mean(self) -> None:
        layer = TargetAwareHistoryPooling(
            2,
            attention_hidden_dim=4,
            count_prior_strength=1.0,
        )
        query = torch.randn(2, 2)
        history = torch.tensor(
            [[2.0, 4.0], [6.0, 8.0], [3.0, 5.0]]
        )
        values = torch.tensor([1.0, 3.0, 2.0])
        offsets = torch.tensor([0, 2, 3])

        pooled, weights = layer(
            query,
            history,
            values,
            offsets,
            return_attention_weights=True,
        )

        expected = torch.tensor([[5.0, 7.0], [3.0, 5.0]])
        self.assertTrue(torch.allclose(pooled, expected, atol=1.0e-6))
        self.assertTrue(
            torch.allclose(weights, torch.tensor([0.25, 0.75, 1.0]))
        )

    def test_weights_are_normalized_inside_each_ragged_segment(self) -> None:
        torch.manual_seed(2026)
        layer = TargetAwareHistoryPooling(
            3,
            attention_hidden_dim=5,
            count_prior_strength=0.5,
        )
        _, weights = layer(
            torch.randn(2, 3),
            torch.randn(5, 3),
            torch.tensor([1.0, 2.0, 3.0, 1.0, 4.0]),
            torch.tensor([0, 2, 5]),
            return_attention_weights=True,
        )

        self.assertTrue(torch.allclose(weights[:2].sum(), torch.tensor(1.0)))
        self.assertTrue(torch.allclose(weights[2:].sum(), torch.tensor(1.0)))
        self.assertTrue(bool((weights >= 0.0).all()))

    def test_learned_activation_depends_on_current_target(self) -> None:
        layer = TargetAwareHistoryPooling(
            1,
            attention_hidden_dim=1,
            count_prior_strength=0.0,
        )
        with torch.no_grad():
            layer.query_projection.weight.fill_(1.0)
            layer.query_projection.bias.zero_()
            layer.key_projection.weight.fill_(1.0)
            layer.key_projection.bias.zero_()
            layer.activation[0].weight.zero_()
            layer.activation[0].weight[0, 2] = 1.0
            layer.activation[0].bias.zero_()
            layer.activation[2].weight.fill_(1.0)
            layer.activation[2].bias.zero_()

        pooled = layer(
            torch.tensor([[1.0], [3.0]]),
            torch.tensor([[1.0], [2.0], [1.0], [2.0]]),
            torch.ones(4),
            torch.tensor([0, 2, 4]),
        )

        self.assertGreater(float(pooled[1, 0]), float(pooled[0, 0]))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TargetAwareDCNPLEESMMTest(unittest.TestCase):
    def test_forward_backward_and_unchanged_input_width(self) -> None:
        model = TargetAwareDCNPLEESMM(
            {field_id: 8 for field_id in ALL_FIELD_IDS},
            embedding_dim=4,
            num_cross_layers=2,
            cross_layer_norm=True,
            num_shared_experts=2,
            num_task_experts=1,
            expert_hidden_dims=(16, 8),
            tower_hidden_dims=(4,),
            dropout=0.0,
            gate_dropout=0.0,
            history_attention_hidden_dim=8,
            history_count_prior_strength=1.0,
        )
        output = model(make_history_batch(), return_gate_weights=True)

        self.assertEqual(model.input_dim, len(ALL_FIELD_IDS) * 4)
        self.assertEqual(
            tuple(model.history_target_field_pairs),
            HISTORY_TARGET_FIELD_PAIRS,
        )
        self.assertEqual(tuple(output["ctr"].shape), (2,))
        self.assertEqual(tuple(output["cvr"].shape), (2,))
        self.assertTrue(
            torch.allclose(output["ctcvr"], output["ctr"] * output["cvr"])
        )
        self.assertTrue(
            torch.allclose(
                output["ctr_gate_weights"].sum(dim=1),
                torch.ones(2),
            )
        )

        (output["ctr_logit"].mean() + output["cvr_logit"].mean()).backward()
        attention_gradients = [
            parameter.grad
            for parameter in model.history_attention.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(attention_gradients)
        self.assertTrue(
            all(
                bool(torch.isfinite(gradient).all())
                for gradient in attention_gradients
            )
        )
        self.assertTrue(
            any(bool(torch.count_nonzero(gradient)) for gradient in attention_gradients)
        )

    def test_factory_rebuilds_target_aware_model_from_saved_config(self) -> None:
        encoder = FeatureEncoder.fit(
            {
                field_id: Counter({"known": 2})
                for field_id in ALL_FIELD_IDS
            },
            min_frequency=2,
        )
        model = build_multitask_model(
            encoder,
            {
                "name": "target_aware_dcn_ple_esmm",
                "embedding_dim": 4,
                "embedding_pooling": "weighted_mean",
                "history_attention_hidden_dim": 8,
                "history_count_prior_strength": 1.0,
                "num_cross_layers": 2,
                "cross_layer_norm": True,
                "num_shared_experts": 2,
                "num_task_experts": 1,
                "expert_hidden_dims": [16, 8],
                "tower_hidden_dims": [4],
                "dropout": 0.0,
                "gate_dropout": 0.0,
            },
        )

        self.assertIsInstance(model, TargetAwareDCNPLEESMM)


if __name__ == "__main__":
    unittest.main()
