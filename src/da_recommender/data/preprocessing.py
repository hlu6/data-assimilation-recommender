"""Preprocessing logic for graph construction, connectivity filtering, and temporal preparation."""

from collections import Counter

import networkx as nx
import torch

from .bucketing import build_time_buckets
from .dataset import TemporalGraphData, load_reddit_dataset


def keep_largest_connected_component(data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Counter]:
    graph = nx.Graph()
    graph.add_edges_from(zip(data.src.tolist(), data.dst.tolist()))
    component = set(max(nx.connected_components(graph), key=len))

    mask = torch.tensor(
        [(u in component and i in component) for u, i in zip(data.src.tolist(), data.dst.tolist())]
    )
    src = data.src[mask]
    dst = data.dst[mask]
    t = data.t[mask].float()

    perm = torch.argsort(t)
    src = src[perm]
    dst = dst[perm]
    t = t[perm]

    return src, dst, t, Counter(src.tolist())


def prepare_temporal_graph_data(root: str, name: str, bucket_gap_seconds: float) -> TemporalGraphData:
    raw = load_reddit_dataset(root=root, name=name)
    src, dst, t, user_activity = keep_largest_connected_component(raw)
    bucket, unique_buckets = build_time_buckets(t, gap_seconds=bucket_gap_seconds)
    item_ids = list(torch.unique(dst).cpu().numpy())
    num_nodes = int(max(src.max(), dst.max())) + 1

    return TemporalGraphData(
        src=src,
        dst=dst,
        t=t,
        bucket=bucket,
        unique_buckets=unique_buckets,
        item_ids=item_ids,
        num_nodes=num_nodes,
        user_activity=user_activity,
    )
