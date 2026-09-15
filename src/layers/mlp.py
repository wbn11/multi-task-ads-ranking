"""Configurable multilayer perceptron."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class MLP(nn.Module):
    """Apply Linear-ReLU-Dropout blocks and return the last hidden state."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_dims or any(dimension <= 0 for dimension in hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be inside [0, 1)")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                (
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            previous_dim = hidden_dim
        self.output_dim = previous_dim
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2:
            raise ValueError("MLP inputs must have shape [batch, input_dim]")
        return self.network(inputs)
