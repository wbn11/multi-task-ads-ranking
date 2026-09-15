"""Expert and task-specific gate layers used by MMoE."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .mlp import MLP


class Expert(nn.Module):
    """Transform one shared input into an expert-specific representation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = MLP(input_dim, hidden_dims, dropout=dropout)
        self.output_dim = self.network.output_dim

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class Gate(nn.Module):
    """Produce normalized expert weights with optional training-time dropout."""

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_experts <= 1:
            raise ValueError("num_experts must be greater than one")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be inside [0, 1)")
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.dropout = float(dropout)
        self.projection = nn.Linear(input_dim, num_experts)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"gate inputs must have shape [batch, {self.input_dim}]"
            )
        weights = torch.softmax(self.projection(inputs), dim=-1)
        if not self.training or self.dropout == 0.0:
            return weights

        keep_mask = torch.rand_like(weights) >= self.dropout
        masked_weights = weights * keep_mask
        row_sums = masked_weights.sum(dim=-1, keepdim=True)
        all_dropped = row_sums == 0.0
        normalized_weights = masked_weights / row_sums.clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        return torch.where(all_dropped, weights, normalized_weights)
