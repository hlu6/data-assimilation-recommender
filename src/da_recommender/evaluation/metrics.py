"""Ranking metrics used to evaluate recommendation quality over time."""

import math

import torch


def evaluate_hitk(x_pred: torch.Tensor, events: list[tuple[int, int]], item_ids: list[int], k: int = 5) -> float:
    hits = 0
    item_emb = x_pred[item_ids]
    for user_id, item_id in events:
        scores = torch.matmul(item_emb, x_pred[user_id])
        topk = torch.topk(scores, k).indices
        predicted_items = [item_ids[idx] for idx in topk.cpu().numpy()]
        hits += int(item_id in predicted_items)
    return hits / max(len(events), 1)


def evaluate_mrr(x_pred: torch.Tensor, events: list[tuple[int, int]], item_ids: list[int]) -> float:
    total_rr = 0.0
    item_emb = x_pred[item_ids]
    for user_id, item_id in events:
        scores = torch.matmul(item_emb, x_pred[user_id])
        ranked_idx = torch.argsort(scores, descending=True)
        ranked_items = [item_ids[idx] for idx in ranked_idx.cpu().numpy()]
        if item_id in ranked_items:
            rank = ranked_items.index(item_id) + 1
            total_rr += 1.0 / rank
    return total_rr / max(len(events), 1)


def evaluate_ndcgk(x_pred: torch.Tensor, events: list[tuple[int, int]], item_ids: list[int], k: int = 10) -> float:
    total_ndcg = 0.0
    item_emb = x_pred[item_ids]
    for user_id, item_id in events:
        scores = torch.matmul(item_emb, x_pred[user_id])
        topk = torch.topk(scores, k).indices
        predicted_items = [item_ids[idx] for idx in topk.cpu().numpy()]
        if item_id in predicted_items:
            rank = predicted_items.index(item_id) + 1
            total_ndcg += 1.0 / math.log2(rank + 1)
    return total_ndcg / max(len(events), 1)
