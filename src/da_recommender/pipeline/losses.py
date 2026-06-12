"""Ranking loss functions and negative-sampling utilities for online learning."""

import random

import torch
import torch.nn.functional as F


def sample_negative(user_id: int, item_ids: list[int], x_pred: torch.Tensor, sample_size: int) -> int:
    candidates = random.sample(item_ids, min(sample_size, len(item_ids)))
    scores = [torch.dot(x_pred[user_id], x_pred[item_id]).item() for item_id in candidates]
    hardest_idx = int(torch.tensor(scores).argmax())
    return candidates[hardest_idx]


def compute_bpr_loss(
    x_pred: torch.Tensor,
    events: list[tuple[int, int]],
    item_ids: list[int],
    negative_sample_size: int,
) -> torch.Tensor:
    loss = 0.0
    for user_id, pos_item in events:
        neg_item = sample_negative(user_id, item_ids, x_pred, negative_sample_size)
        pos = torch.clamp((x_pred[user_id] * x_pred[pos_item]).sum(), -10, 10)
        neg = torch.clamp((x_pred[user_id] * x_pred[neg_item]).sum(), -10, 10)
        loss += F.softplus(-(pos - neg))
    return loss / max(len(events), 1)
