"""Multi-gate Mixture-of-Experts for joint CTR and CVR prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from src.data.feature_encoder import FeatureEncoder
from src.data.feature_schema import ALL_FIELD_IDS
from src.layers.embedding import SparseFeatureEmbedding
from src.layers.expert import Expert, Gate
from src.layers.mlp import MLP


class MMoE(nn.Module):
    """Share multiple experts while learning one gate per funnel task."""

    def __init__(
        self,
        field_vocab_sizes: Mapping[str, int],
        *,
        embedding_dim: int,
        num_experts: int,
        expert_hidden_dims: Sequence[int],
        tower_hidden_dims: Sequence[int],
        dropout: float,
        gate_dropout: float = 0.0,
        embedding_pooling: str = "weighted_mean",
    ) -> None:
        super().__init__()
        missing_fields = set(ALL_FIELD_IDS) - set(field_vocab_sizes)
        extra_fields = set(field_vocab_sizes) - set(ALL_FIELD_IDS)
        if missing_fields or extra_fields:
            raise ValueError(
                "field vocabulary mismatch: "
                f"missing={sorted(missing_fields)}, extra={sorted(extra_fields)}"
            )
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than one")

        self.field_ids = ALL_FIELD_IDS
        self.embedding_dim = embedding_dim
        self.embedding_pooling = embedding_pooling
        self.num_experts = num_experts
        self.gate_dropout = float(gate_dropout)
        self.feature_embeddings = nn.ModuleDict(
            {
                field_id: SparseFeatureEmbedding(
                    int(field_vocab_sizes[field_id]),
                    embedding_dim,
                    pooling=embedding_pooling,
                )
                for field_id in self.field_ids
            }
        )
        self.input_dim = len(self.field_ids) * embedding_dim
        self.experts = nn.ModuleList(
            [
                Expert(
                    self.input_dim,
                    expert_hidden_dims,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )
        expert_output_dim = self.experts[0].output_dim
        if any(
            expert.output_dim != expert_output_dim for expert in self.experts
        ):
            raise RuntimeError("all experts must have identical output dimensions")
        self.expert_output_dim = expert_output_dim
        self.ctr_gate = Gate(
            self.input_dim,
            num_experts,
            dropout=gate_dropout,
        )
        self.cvr_gate = Gate(
            self.input_dim,
            num_experts,
            dropout=gate_dropout,
        )
        self.ctr_tower = MLP(
            expert_output_dim,
            tower_hidden_dims,
            dropout=dropout,
        )
        self.cvr_tower = MLP(
            expert_output_dim,
            tower_hidden_dims,
            dropout=dropout,
        )
        self.ctr_output = nn.Linear(self.ctr_tower.output_dim, 1)
        self.cvr_output = nn.Linear(self.cvr_tower.output_dim, 1)
        nn.init.xavier_uniform_(self.ctr_output.weight)
        nn.init.zeros_(self.ctr_output.bias)
        nn.init.xavier_uniform_(self.cvr_output.weight)
        nn.init.zeros_(self.cvr_output.bias)

    @classmethod
    def from_feature_encoder(
        cls,
        encoder: FeatureEncoder,
        *,
        embedding_dim: int,
        num_experts: int,
        expert_hidden_dims: Sequence[int],
        tower_hidden_dims: Sequence[int],
        dropout: float,
        gate_dropout: float = 0.0,
        embedding_pooling: str = "weighted_mean",
    ) -> "MMoE":
        return cls(
            {
                field_id: encoder.field_vocabularies[field_id].vocab_size
                for field_id in ALL_FIELD_IDS
            },
            embedding_dim=embedding_dim,
            num_experts=num_experts,
            expert_hidden_dims=expert_hidden_dims,
            tower_hidden_dims=tower_hidden_dims,
            dropout=dropout,
            gate_dropout=gate_dropout,
            embedding_pooling=embedding_pooling,
        )

    def _model_input(self, batch: Mapping[str, Any]) -> Tensor:
        features = batch.get("features")
        if not isinstance(features, Mapping):
            raise ValueError("batch must contain a features mapping")

        dense_embeddings: list[Tensor] = []
        for field_id in self.field_ids:
            field = features.get(field_id)
            if not isinstance(field, Mapping):
                raise ValueError(f"batch is missing feature field {field_id}")
            try:
                ids = field["ids"]
                values = field["values"]
                offsets = field["offsets"]
            except KeyError as error:
                raise ValueError(
                    f"field {field_id} must contain ids, values and offsets"
                ) from error
            dense_embeddings.append(
                self.feature_embeddings[field_id](ids, values, offsets)
            )
        field_embeddings = torch.stack(dense_embeddings, dim=1)
        return field_embeddings.flatten(start_dim=1)

    @staticmethod
    def _mix_experts(expert_outputs: Tensor, gate_weights: Tensor) -> Tensor:
        if expert_outputs.ndim != 3 or gate_weights.ndim != 2:
            raise ValueError("invalid expert output or gate weight shape")
        if expert_outputs.shape[:2] != gate_weights.shape:
            raise ValueError("expert count must match gate weight count")
        return torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        return_gate_weights: bool = False,
    ) -> dict[str, Tensor]:
        model_input = self._model_input(batch)
        expert_outputs = torch.stack(
            [expert(model_input) for expert in self.experts],
            dim=1,
        )
        ctr_gate_weights = self.ctr_gate(model_input)
        cvr_gate_weights = self.cvr_gate(model_input)
        ctr_representation = self._mix_experts(
            expert_outputs,
            ctr_gate_weights,
        )
        cvr_representation = self._mix_experts(
            expert_outputs,
            cvr_gate_weights,
        )
        ctr_logit = self.ctr_output(
            self.ctr_tower(ctr_representation)
        ).squeeze(-1)
        cvr_logit = self.cvr_output(
            self.cvr_tower(cvr_representation)
        ).squeeze(-1)
        ctr = torch.sigmoid(ctr_logit)
        cvr = torch.sigmoid(cvr_logit)
        outputs = {
            "ctr_logit": ctr_logit,
            "cvr_logit": cvr_logit,
            "ctr": ctr,
            "cvr": cvr,
            "ctcvr": ctr * cvr,
        }
        if return_gate_weights:
            outputs.update(
                {
                    "ctr_gate_weights": ctr_gate_weights,
                    "cvr_gate_weights": cvr_gate_weights,
                }
            )
        return outputs
