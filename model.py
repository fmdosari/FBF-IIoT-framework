from __future__ import annotations

import torch
from torch import nn


class IndustrialMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        h1 = 128 if input_dim >= 64 else 96
        h2 = 64
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(h2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
