"""
5-layer ReLU MLP denoiser with sinusoidal time conditioning.
Time input is continuous t in [0, 1].
"""

import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Maps continuous t in [0,1] to sinusoidal embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10_000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class MLPDenoiser(nn.Module):
    """
    5-layer ReLU MLP, 256 hidden, sinusoidal time embedding, concat conditioning.
    """

    def __init__(self, data_dim: int, hidden_dim: int = 256, n_layers: int = 5,
                 time_dim: int = 128):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)

        layers = []
        in_dim = data_dim + time_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=-1))
