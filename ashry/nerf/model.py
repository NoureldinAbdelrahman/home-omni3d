"""NeRF MLP and positional encoding."""

from __future__ import annotations

import torch
from torch import nn


def encode(points: torch.Tensor, levels: int) -> torch.Tensor:
    features = [points]
    for level in range(levels):
        frequency = 2.0**level * torch.pi
        features.extend((torch.sin(frequency * points), torch.cos(frequency * points)))
    return torch.cat(features, dim=-1)


class NeRF(nn.Module):
    def __init__(self, position_levels: int = 10, direction_levels: int = 4, hidden: int = 256):
        super().__init__()
        self.position_levels = position_levels
        self.direction_levels = direction_levels
        position_size = 3 * (1 + 2 * position_levels)
        direction_size = 3 * (1 + 2 * direction_levels)
        self.trunk = nn.Sequential(
            nn.Linear(position_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.sigma = nn.Linear(hidden, 1)
        self.feature = nn.Linear(hidden, hidden)
        self.color = nn.Sequential(
            nn.Linear(hidden + direction_size, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, points: torch.Tensor, directions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(encode(points, self.position_levels))
        density = torch.relu(self.sigma(hidden))
        color_input = torch.cat((self.feature(hidden), encode(directions, self.direction_levels)), dim=-1)
        color = torch.sigmoid(self.color(color_input))
        return density, color
