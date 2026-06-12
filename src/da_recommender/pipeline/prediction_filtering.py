"""Helpers for the prediction, event collection, and state-update stages of the pipeline."""

import torch


def collect_bucket_events(src: torch.Tensor, dst: torch.Tensor, bucket: torch.Tensor, bucket_id: int) -> list[tuple[int, int]]:
    idxs = torch.where(bucket == bucket_id)[0]
    return [(int(src[k]), int(dst[k])) for k in idxs]


def predict_state(x_state: torch.Tensor, predictor, edge_weight: dict, gamma: float) -> torch.Tensor:
    with torch.no_grad():
        return (1 - gamma) * x_state + gamma * predictor(x_state, edge_weight)


def update_state(x_state: torch.Tensor, x_filtered: torch.Tensor, eta: float) -> torch.Tensor:
    x_next = (1 - eta) * x_state + eta * x_filtered
    return torch.clamp(x_next, -20, 20).detach()
