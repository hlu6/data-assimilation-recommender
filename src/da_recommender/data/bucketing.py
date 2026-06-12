"""Time discretization helpers for converting event streams into sequential update buckets."""

import torch


def build_time_buckets(timestamps: torch.Tensor, gap_seconds: float) -> tuple[torch.Tensor, torch.Tensor]:
    bucket = torch.zeros_like(timestamps, dtype=torch.long)
    current_bucket = 0
    bucket[0] = current_bucket
    last_time = timestamps[0]

    for idx in range(1, len(timestamps)):
        if timestamps[idx] - last_time >= gap_seconds:
            current_bucket += 1
            last_time = timestamps[idx]
        bucket[idx] = current_bucket

    return bucket, torch.unique(bucket)
