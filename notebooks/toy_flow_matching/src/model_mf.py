"""
MLP denoiser for MeanFlow — conditions on both t and horizon h = t - r.
"""

import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10_000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class MLPDenoiserMF(nn.Module):
    """
    5-layer ReLU MLP conditioned on (z, t, h) via sinusoidal embeddings.
    t = current time, h = horizon (t - r).
    """

    def __init__(self, data_dim: int, hidden_dim: int = 256, n_layers: int = 5,
                 time_dim: int = 128):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_dim)
        self.horizon_embed = SinusoidalEmbedding(time_dim)

        layers = []
        in_dim = data_dim + time_dim * 2  # z + t_emb + h_emb
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, h):
        t_emb = self.time_embed(t)
        h_emb = self.horizon_embed(h)
        return self.net(torch.cat([x, t_emb, h_emb], dim=-1))
