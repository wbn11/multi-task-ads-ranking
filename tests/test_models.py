from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    from torch import nn

    from src.data.dataset import collate_ads_batch
    from src.data.feature_schema import ALL_FIELD_IDS
    from src.layers.cross_network import CrossNetworkV2
    from src.layers.embedding import SparseFeatureEmbedding, SparseLinearEmbedding
    from src.layers.expert import Expert, Gate
    from src.layers.fm import FactorizationMachine
    from src.models.dcnv2 import DCNv2
    from src.models.dcn_ple_esmm import DCNPLEESMM
    from src.models.deepfm import DeepFM
    from src.models.esmm import ESMM
    from src.models.lr import LogisticRegressionCTR
    from src.models.mmoe import MMoE
    from src.models.ple import PLE
    from src.models.shared_bottom import SharedBottom


def make_model_batch():
    samples = [
        {
            "sample_id": "1",
            "common_feature_id": "c1",
            "user_id": "u1",
            "features": {
                "101": {"ids": [2], "values": [1.0]},
                "205": {"ids": [3], "values": [2.0]},
            },
            "click": 1,
            "conversion": 0,
            "ctcvr": 0,
        },
        {
            "sample_id": "2",
            "common_feature_id": "c2",
            "user_id": "u2",
            "features": {
                "101": {"ids": [3], "values": [1.0]},
                "205": {"ids": [2], "values": [1.0]},
            },
            "click": 0,
            "conversion": 0,
            "ctcvr": 0,
        },
    ]
    return collate_ads_batch(samples)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class SparseLinearEmbeddingTest(unittest.TestCase):
    def test_weighted_ragged_sum(self) -> None:
        layer = SparseLinearEmbedding(vocab_size=6)
        with torch.no_grad():
            layer.embedding.weight[2, 0] = 2.0
            layer.embedding.weight[3, 0] = 3.0

        output = layer(
            ids=torch.tensor([2, 3, 2]),
            values=torch.tensor([0.5, 2.0, 1.5]),
            offsets=torch.tensor([0, 2, 3]),
        )

        self.assertEqual(tuple(output.shape), (2, 1))
        self.assertTrue(torch.allclose(output[:, 0], torch.tensor([7.0, 3.0])))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class SparseFeatureEmbeddingTest(unittest.TestCase):
    def test_weighted_mean_pooling(self) -> None:
        layer = SparseFeatureEmbedding(
            vocab_size=6, embedding_dim=2, pooling="weighted_mean"
        )
        with torch.no_grad():
            layer.embedding.weight.zero_()
            layer.embedding.weight[2] = torch.tensor([2.0, 4.0])
            layer.embedding.weight[3] = torch.tensor([6.0, 8.0])

        output = layer(
            ids=torch.tensor([2, 3, 2]),
            values=torch.tensor([1.0, 3.0, 2.0]),
            offsets=torch.tensor([0, 2, 3]),
        )

        expected = torch.tensor([[5.0, 7.0], [2.0, 4.0]])
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(torch.allclose(output, expected))

    def test_weighted_mean_is_stable_after_a_large_preceding_bag(self) -> None:
        layer = SparseFeatureEmbedding(
            vocab_size=4, embedding_dim=1, pooling="weighted_mean"
        )
        with torch.no_grad():
            layer.embedding.weight.zero_()
            layer.embedding.weight[2, 0] = 2.0

        # In float32, 2**24 + 1 rounds back to 2**24.  A global prefix-sum
        # difference would therefore calculate a zero denominator for bag 2.
        output = layer(
            ids=torch.tensor([2, 2]),
            values=torch.tensor([float(2**24), 1.0]),
            offsets=torch.tensor([0, 1, 2]),
        )

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(output[:, 0], torch.tensor([2.0, 2.0])))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class FactorizationMachineTest(unittest.TestCase):
    def test_matches_explicit_pairwise_dot_product(self) -> None:
        embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        output = FactorizationMachine()(embeddings)
        self.assertTrue(torch.allclose(output, torch.tensor([11.0])))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class CrossNetworkV2Test(unittest.TestCase):
    def test_one_layer_matches_equation(self) -> None:
        layer = CrossNetworkV2(input_dim=2, num_layers=1)
        with torch.no_grad():
            layer.weights[0].copy_(torch.eye(2))
            layer.biases[0].zero_()
        inputs = torch.tensor([[2.0, 3.0]])
        output = layer(inputs)
        expected = inputs * inputs + inputs
        self.assertTrue(torch.allclose(output, expected))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class LogisticRegressionCTRTest(unittest.TestCase):
    def test_forward_shape_range_and_backward(self) -> None:
        model = LogisticRegressionCTR({field_id: 8 for field_id in ALL_FIELD_IDS})
        batch = make_model_batch()

        output = model(batch)

        self.assertEqual(set(output), {"ctr_logit", "ctr"})
        self.assertEqual(tuple(output["ctr_logit"].shape), (2,))
        self.assertEqual(tuple(output["ctr"].shape), (2,))
        self.assertTrue(bool(torch.isfinite(output["ctr_logit"]).all()))
        self.assertTrue(bool(((output["ctr"] >= 0) & (output["ctr"] <= 1)).all()))

        loss = nn.BCEWithLogitsLoss()(output["ctr_logit"], batch["click"])
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))

    def test_requires_every_schema_field(self) -> None:
        vocab_sizes = {field_id: 8 for field_id in ALL_FIELD_IDS}
        vocab_sizes.pop("301")
        with self.assertRaises(ValueError):
            LogisticRegressionCTR(vocab_sizes)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DeepFMTest(unittest.TestCase):
    def test_forward_shape_range_and_backward(self) -> None:
        model = DeepFM(
            {field_id: 8 for field_id in ALL_FIELD_IDS},
            embedding_dim=4,
            hidden_dims=(16, 8),
            dropout=0.0,
        )
        batch = make_model_batch()

        output = model(batch)

        self.assertEqual(set(output), {"ctr_logit", "ctr"})
        self.assertEqual(tuple(output["ctr_logit"].shape), (2,))
        self.assertEqual(tuple(output["ctr"].shape), (2,))
        self.assertTrue(bool(torch.isfinite(output["ctr_logit"]).all()))
        self.assertTrue(bool(((output["ctr"] >= 0) & (output["ctr"] <= 1)).all()))

        loss = nn.BCEWithLogitsLoss()(output["ctr_logit"], batch["click"])
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))

    def test_requires_every_schema_field(self) -> None:
        vocab_sizes = {field_id: 8 for field_id in ALL_FIELD_IDS}
        vocab_sizes.pop("301")
        with self.assertRaises(ValueError):
            DeepFM(
                vocab_sizes,
                embedding_dim=4,
                hidden_dims=(16, 8),
                dropout=0.0,
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DCNv2Test(unittest.TestCase):
    def test_forward_shape_range_and_backward(self) -> None:
        model = DCNv2(
            {field_id: 8 for field_id in ALL_FIELD_IDS},
            embedding_dim=4,
            num_cross_layers=2,
            hidden_dims=(16, 8),
            dropout=0.0,
        )
        batch = make_model_batch()
        output = model(batch)

        self.assertEqual(set(output), {"ctr_logit", "ctr"})
        self.assertEqual(tuple(output["ctr_logit"].shape), (2,))
        self.assertEqual(tuple(output["ctr"].shape), (2,))
        self.assertTrue(bool(torch.isfinite(output["ctr_logit"]).all()))
        self.assertTrue(bool(((output["ctr"] >= 0) & (output["ctr"] <= 1)).all()))

        loss = nn.BCEWithLogitsLoss()(output["ctr_logit"], batch["click"])
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))

    def test_requires_every_schema_field(self) -> None:
        vocab_sizes = {field_id: 8 for field_id in ALL_FIELD_IDS}
        vocab_sizes.pop("301")
        with self.assertRaises(ValueError):
            DCNv2(
                vocab_sizes,
                embedding_dim=4,
                num_cross_layers=2,
                hidden_dims=(16, 8),
                dropout=0.0,
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class SharedBottomTest(unittest.TestCase):
    def test_forward_shape_range_identity_and_backward(self) -> None:
        model = SharedBottom(
            {field_id: 8 for field_id in ALL_FIELD_IDS},
            embedding_dim=4,
            shared_hidden_dims=(16, 8),
            tower_hidden_dims=(4,),
            dropout=0.0,
        )
        batch = make_model_batch()
        output = model(batch)

        self.assertEqual(
            set(output),
            {"ctr_logit", "cvr_logit", "ctr", "cvr", "ctcvr"},
        )
        for name in ("ctr_logit", "cvr_logit", "ctr", "cvr", "ctcvr"):
            self.assertEqual(tuple(output[name].shape), (2,))
            self.assertTrue(bool(torch.isfinite(output[name]).all()))
        for name in ("ctr", "cvr", "ctcvr"):
            self.assertTrue(
                bool(((output[name] >= 0) & (output[name] <= 1)).all())
            )
        self.assertTrue(
            torch.allclose(output["ctcvr"], output["ctr"] * output["cvr"])
        )

        loss = output["ctr_logit"].mean() + output["cvr_logit"].mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))

    def test_requires_every_schema_field(self) -> None:
        vocab_sizes = {field_id: 8 for field_id in ALL_FIELD_IDS}
        vocab_sizes.pop("301")
        with self.assertRaises(ValueError):
            SharedBottom(
                vocab_sizes,
                embedding_dim=4,
                shared_hidden_dims=(16, 8),
                tower_hidden_dims=(4,),
                dropout=0.0,
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ESMMTest(unittest.TestCase):
    def test_forward_product_identity_and_backward(self) -> None:
        model = ESMM(
            {field_id: 8 for field_id in ALL_FIELD_IDS},
            embedding_dim=4,
            shared_hidden_dims=(16, 8),
            tower_hidden_dims=(4,),
            dropout=0.0,
        )
        output = model(make_model_batch())

        self.assertEqual(
            set(output),
            {"ctr_logit", "cvr_logit", "ctr", "cvr", "ctcvr"},
        )
        self.assertTrue(
            torch.allclose(output["ctcvr"], output["ctr"] * output["cvr"])
        )
        loss = output["ctr_logit"].mean() + output["cvr_logit"].mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ExpertAndGateTest(unittest.TestCase):
    def test_expert_shape_and_gate_probability_simplex(self) -> None:
        inputs = torch.randn(3, 6)
        expert = Expert(6, (5, 4), dropout=0.0)
        gate = Gate(6, num_experts=3)

        expert_output = expert(inputs)
        gate_weights = gate(inputs)

        self.assertEqual(tuple(expert_output.shape), (3, 4))
        self.assertEqual(tuple(gate_weights.shape), (3, 3))
        self.assertTrue(bool((gate_weights >= 0.0).all()))
        self.assertTrue(
            torch.allclose(gate_weights.sum(dim=1), torch.ones(3))
        )

    def test_gate_rejects_a_single_expert(self) -> None:
        with self.assertRaises(ValueError):
            Gate(6, num_experts=1)

    def test_gate_dropout_renormalizes_kept_experts(self) -> None:
        torch.manual_seed(2026)
        gate = Gate(4, num_experts=4, dropout=0.5)
        with torch.no_grad():
            gate.projection.weight.zero_()
            gate.projection.bias.zero_()
        inputs = torch.zeros(64, 4)

        gate.train()
        training_weights = gate(inputs)
        gate.eval()
        evaluation_weights = gate(inputs)

        self.assertTrue(bool((training_weights == 0.0).any()))
        self.assertTrue(
            torch.allclose(training_weights.sum(dim=1), torch.ones(64))
        )
        self.assertTrue(
            torch.allclose(evaluation_weights, torch.full((64, 4), 0.25))
        )

    def test_gate_rejects_invalid_dropout(self) -> None:
        with self.assertRaises(ValueError):
            Gate(6, num_experts=3, dropout=1.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class MMoETest(unittest.TestCase):
    def test_forward_gates_product_identity_and_backward(self) -> None:
        model = MMoE(
            {field_id: 8 for field_id in ALL_FIELD_IDS},
            embedding_dim=4,
            num_experts=3,
            expert_hidden_dims=(16, 8),
            tower_hidden_dims=(4,),
            dropout=0.0,
            gate_dropout=0.2,
        )
        batch = make_model_batch()

        standard_output = model(batch)
        output = model(batch, return_gate_weights=True)

        self.assertEqual(
            set(standard_output),
            {"ctr_logit", "cvr_logit", "ctr", "cvr", "ctcvr"},
        )
        self.assertEqual(tuple(output["ctr_gate_weights"].shape), (2, 3))
        self.assertEqual(tuple(output["cvr_gate_weights"].shape), (2, 3))
        self.assertTrue(
            torch.allclose(
                output["ctr_gate_weights"].sum(dim=1),
                torch.ones(2),
            )
        )
        self.assertTrue(
            torch.allclose(
                output["cvr_gate_weights"].sum(dim=1),
                torch.ones(2),
            )
        )
        self.assertTrue(
            torch.allclose(output["ctcvr"], output["ctr"] * output["cvr"])
        )

        loss = output["ctr_logit"].mean() + output["cvr_logit"].mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class PLETest(unittest.TestCase):
    def test_forward_gates_product_identity_and_backward(self) -> None:
        model = PLE(
            {field_id: 8 for field_id in ALL_FIELD_IDS},
            embedding_dim=4,
            num_shared_experts=2,
            num_task_experts=1,
            expert_hidden_dims=(16, 8),
            tower_hidden_dims=(4,),
            dropout=0.0,
        )
        batch = make_model_batch()

        standard_output = model(batch)
        output = model(batch, return_gate_weights=True)

        self.assertEqual(
            set(standard_output),
            {"ctr_logit", "cvr_logit", "ctr", "cvr", "ctcvr"},
        )
        self.assertEqual(tuple(output["ctr_gate_weights"].shape), (2, 3))
        self.assertEqual(tuple(output["cvr_gate_weights"].shape), (2, 3))
        self.assertTrue(
            torch.allclose(
                output["ctr_gate_weights"].sum(dim=1),
                torch.ones(2),
            )
        )
        self.assertTrue(
            torch.allclose(
                output["cvr_gate_weights"].sum(dim=1),
                torch.ones(2),
            )
        )
        self.assertTrue(
            torch.allclose(output["ctcvr"], output["ctr"] * output["cvr"])
        )

        loss = output["ctr_logit"].mean() + output["cvr_logit"].mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in gradients))

    def test_task_specific_experts_receive_only_their_task_gradient(self) -> None:
        model = PLE(
            {field_id: 8 for field_id in ALL_FIELD_IDS},
            embedding_dim=4,
            num_shared_experts=2,
            num_task_experts=1,
            expert_hidden_dims=(16, 8),
            tower_hidden_dims=(4,),
            dropout=0.0,
        )
        batch = make_model_batch()

        model(batch)["ctr_logit"].sum().backward()
        self.assertTrue(
            any(p.grad is not None for p in model.shared_experts.parameters())
        )
        self.assertTrue(
            any(p.grad is not None for p in model.ctr_experts.parameters())
        )
        self.assertTrue(
            all(p.grad is None for p in model.cvr_experts.parameters())
        )

        model.zero_grad(set_to_none=True)
        model(batch)["cvr_logit"].sum().backward()
        self.assertTrue(
            any(p.grad is not None for p in model.shared_experts.parameters())
        )
        self.assertTrue(
            all(p.grad is None for p in model.ctr_experts.parameters())
        )
        self.assertTrue(
            any(p.grad is not None for p in model.cvr_experts.parameters())
        )

    def test_requires_shared_and_task_specific_experts(self) -> None:
        vocabulary = {field_id: 8 for field_id in ALL_FIELD_IDS}
        with self.assertRaises(ValueError):
            PLE(
                vocabulary,
                embedding_dim=4,
                num_shared_experts=0,
                num_task_experts=1,
                expert_hidden_dims=(8,),
                tower_hidden_dims=(4,),
                dropout=0.0,
            )
        with self.assertRaises(ValueError):
            PLE(
                vocabulary,
                embedding_dim=4,
                num_shared_experts=2,
                num_task_experts=0,
                expert_hidden_dims=(8,),
                tower_hidden_dims=(4,),
                dropout=0.0,
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class DCNPLEESMMTest(unittest.TestCase):
    def test_cross_gates_product_identity_and_backward(self) -> None:
        model = DCNPLEESMM(
            {field_id: 8 for field_id in ALL_FIELD_IDS},
            embedding_dim=4,
            num_cross_layers=2,
            cross_layer_norm=True,
            num_shared_experts=2,
            num_task_experts=1,
            expert_hidden_dims=(16, 8),
            tower_hidden_dims=(4,),
            dropout=0.0,
            gate_dropout=0.1,
        )
        batch = make_model_batch()

        model.eval()
        output = model(batch, return_gate_weights=True)
        self.assertEqual(tuple(output["ctr"].shape), (2,))
        self.assertEqual(tuple(output["cvr"].shape), (2,))
        self.assertEqual(tuple(output["ctcvr"].shape), (2,))
        self.assertEqual(tuple(output["ctr_gate_weights"].shape), (2, 3))
        self.assertTrue(
            torch.allclose(
                output["ctcvr"],
                output["ctr"] * output["cvr"],
            )
        )
        self.assertTrue(
            torch.allclose(
                output["ctr_gate_weights"].sum(dim=1),
                torch.ones(2),
            )
        )

        loss = output["ctr_logit"].mean() + output["cvr_logit"].mean()
        loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.cross_network.parameters()
            )
        )
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(
            all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        )

if __name__ == "__main__":
    unittest.main()
