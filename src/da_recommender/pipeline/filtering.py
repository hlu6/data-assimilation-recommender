"""Filtering and edge-decay updates applied after each batch of new observations."""

import torch


def filtering_update(
    x_pred: torch.Tensor,
    events: list[tuple[int, int]],
    item_ids: list[int],
    beta: float = 0.01,
    num_competitors: int = 3,
) -> torch.Tensor:
    x_new = x_pred.clone()
    item_ids_tensor = torch.tensor(item_ids, device=x_pred.device)

    for user_id, pos_item in events:
        scores = torch.matmul(x_pred[item_ids_tensor], x_pred[user_id])
        mask = item_ids_tensor != pos_item
        neg_items = item_ids_tensor[mask]
        neg_scores = scores[mask]
        if neg_items.numel() == 0:
            continue

        top_k = min(num_competitors, neg_items.shape[0])
        top_idx = torch.topk(neg_scores, top_k).indices
        competitors = neg_items[top_idx]
        competitor_scores = neg_scores[top_idx]

        pos_score = torch.dot(x_pred[user_id], x_pred[pos_item])
        margin = pos_score - competitor_scores.max()
        gain = torch.sigmoid(-margin)
        innovation = 1 - margin
        weights = torch.softmax(competitor_scores, dim=0)

        for weight, neg_item in zip(weights, competitors):
            x_new[user_id] += beta * gain * innovation * weight * (x_pred[pos_item] - x_pred[neg_item])
            x_new[pos_item] += beta * gain * innovation * weight * x_pred[user_id]
            x_new[neg_item] -= beta * gain * innovation * weight * x_pred[user_id]

    return x_new


def decay_edge_weight(value: float, dt: int, gamma: float = 0.05) -> torch.Tensor:
    return value * torch.exp(torch.tensor(-gamma * dt))
