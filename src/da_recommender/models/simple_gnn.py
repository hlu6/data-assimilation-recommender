"""A simple graph neural network used as the current prediction module."""

import torch
import torch.nn as nn


class SimpleGNN(nn.Module):
    """A lightweight graph prior used in the prediction step."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor, edge_weight: dict[tuple[int, int], tuple[float, int]]) -> torch.Tensor:
        x_new = torch.zeros_like(x)
        deg = torch.zeros(x.shape[0], device=x.device)

        for (u, i), (w, _) in edge_weight.items():
            x_new[u] += w * x[i]
            x_new[i] += w * x[u]
            deg[u] += w
            deg[i] += w

        deg = deg.unsqueeze(1) + 1e-6
        deg[deg == 0] = 1.0
        x_new = x_new / deg
        return torch.relu(self.projection(x_new))
