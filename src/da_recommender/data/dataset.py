"""Dataset loading and basic interaction-level summary helpers."""

from collections import Counter
from dataclasses import dataclass

import torch
from torch_geometric.datasets import JODIEDataset


@dataclass
class TemporalGraphData:
    src: torch.Tensor
    dst: torch.Tensor
    t: torch.Tensor
    bucket: torch.Tensor
    unique_buckets: torch.Tensor
    item_ids: list[int]
    num_nodes: int
    user_activity: Counter


def load_reddit_dataset(root: str, name: str = "reddit"):
    dataset = JODIEDataset(root=root, name=name)
    return dataset[0]


def describe_interactions(data) -> dict:
    t = data.t.float()
    delta = t[1:] - t[:-1]
    user_counts = Counter(data.src.tolist())
    counts = torch.tensor(list(user_counts.values()), dtype=torch.float)
    return {
        "time_range_seconds": (t.max() - t.min()).item(),
        "mean_gap_seconds": delta.mean().item(),
        "median_gap_seconds": delta.median().item(),
        "max_gap_hours": delta.max().item() / 3600,
        "num_users": len(user_counts),
        "mean_interactions": counts.mean().item(),
        "median_interactions": counts.median().item(),
        "max_interactions": counts.max().item(),
    }
