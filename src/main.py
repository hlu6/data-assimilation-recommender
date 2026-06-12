"""Run the dynamic graph recommendation experiment using the configured pipeline."""

from da_recommender.config import ExperimentConfig
from da_recommender.data.preprocessing import prepare_temporal_graph_data
from da_recommender.pipeline.runner import run_parallel_live


def main():
    config = ExperimentConfig()
    graph_data = prepare_temporal_graph_data(
        root=config.data_root,
        name=config.dataset_name,
        bucket_gap_seconds=config.bucket_gap_seconds,
    )
    run_parallel_live(graph_data, config)


if __name__ == "__main__":
    main()
