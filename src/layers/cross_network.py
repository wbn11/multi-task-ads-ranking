"""Explicit feature crossing layers used by DCNv2."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CrossNetworkV2(nn.Module):
    """Full-matrix DCNv2 cross network.

    Each layer computes ``x0 * (W_l @ x_l + b_l) + x_l`` with element-wise
    multiplication. The residual path keeps the original representation while
    successive layers increase the maximum explicit polynomial order.
    """

    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim, input_dim)) for _ in range(num_layers)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        for bias in self.biases:
            nn.init.zeros_(bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"inputs must have shape [batch, {self.input_dim}]"
            )
        x0 = inputs
        crossed = inputs
        for weight, bias in zip(self.weights, self.biases):
            transformed = F.linear(crossed, weight, bias)
            crossed = x0 * transformed + crossed
        return crossed
