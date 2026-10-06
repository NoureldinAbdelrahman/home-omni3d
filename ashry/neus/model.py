"""Compact NeuS-style signed-distance and color networks."""

from __future__ import annotations

import math

import torch
from torch import nn


def encode(values: torch.Tensor, levels: int) -> torch.Tensor:
    features = [values]
    for level in range(levels):
        frequency = 2.0**level * torch.pi
        features.extend((torch.sin(frequency * values), torch.cos(frequency * values)))
    return torch.cat(features, dim=-1)


class SDFNetwork(nn.Module):
    def __init__(self, position_levels: int = 6, hidden: int = 192, sphere_radius: float = 0.65):
        super().__init__()
        self.position_levels = position_levels
        self.sphere_radius = sphere_radius
        size = 3 * (1 + 2 * position_levels)
        self.mlp = nn.Sequential(
            nn.Linear(size, hidden), nn.Softplus(beta=100),
            nn.Linear(hidden, hidden), nn.Softplus(beta=100),
            nn.Linear(hidden, hidden), nn.Softplus(beta=100),
            nn.Linear(hidden, hidden), nn.Softplus(beta=100),
        )
        self.sdf = nn.Linear(hidden, 1)
        self.feature = nn.Linear(hidden, hidden)
        nn.init.normal_(self.sdf.weight, mean=math.sqrt(math.pi) / math.sqrt(hidden), std=1e-4)
        nn.init.constant_(self.sdf.bias, -sphere_radius)

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.mlp(encode(points, self.position_levels))
        return self.sdf(hidden) + (points.norm(dim=-1, keepdim=True) - self.sphere_radius), self.feature(hidden)


class ColorNetwork(nn.Module):
    def __init__(self, feature_size: int = 192, direction_levels: int = 4, hidden: int = 128):
        super().__init__()
        self.direction_levels = direction_levels
        direction_size = 3 * (1 + 2 * direction_levels)
        self.mlp = nn.Sequential(
            nn.Linear(feature_size + 3 + direction_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(
        self, points: torch.Tensor, normals: torch.Tensor, directions: torch.Tensor, feature: torch.Tensor
    ) -> torch.Tensor:
        values = torch.cat((feature, normals, encode(directions, self.direction_levels)), dim=-1)
        return torch.sigmoid(self.mlp(values))


class NeuS(nn.Module):
    def __init__(self):
        super().__init__()
        self.sdf_network = SDFNetwork()
        self.color_network = ColorNetwork()
        self.log_inv_s = nn.Parameter(torch.tensor(3.0))

    def inv_s(self) -> torch.Tensor:
        return self.log_inv_s.exp().clamp(1.0, 10_000.0)
