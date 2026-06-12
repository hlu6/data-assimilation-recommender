"""Central experiment settings shared across data loading, models, and the pipeline."""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    dataset_name: str = "reddit"
    data_root: str = "data"
    embedding_dim: int = 32
    bucket_gap_seconds: float = 60.0
    max_steps: int = 2000
    beta: float = 0.2
    gamma: float = 0.0
    eta: float = 0.1
    tau: float = 1.0
    num_competitors: int = 3
    lambda_norm: float = 1e-3
    learning_rate: float = 1e-3
    recent_event_window: int = 300
    negative_sample_size: int = 20
    edge_decay_gamma: float = 0.05
    n_trials: int = 3
    seed: int = 42
