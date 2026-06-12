"""Parallel experiment runner that ties together data, the predictor, and the filtering pipeline."""

import copy
import random
from dataclasses import asdict
from multiprocessing import Process, Queue

import numpy as np
import torch

from ..config import ExperimentConfig
from ..evaluation.metrics import evaluate_hitk, evaluate_mrr, evaluate_ndcgk
from ..models.simple_gnn import SimpleGNN
from .filtering import decay_edge_weight, filtering_update
from .losses import compute_bpr_loss
from .prediction_filtering import collect_bucket_events, predict_state, update_state


def _run_trial_worker(trial_id: int, seed: int, queue: Queue, graph_data, config: ExperimentConfig):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    predictor = SimpleGNN(config.embedding_dim)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=config.learning_rate)

    target_predictor = copy.deepcopy(predictor)
    for param in target_predictor.parameters():
        param.requires_grad = False

    edge_weight_local = {}
    event_buffer_local = []
    x_state = torch.randn(graph_data.num_nodes, config.embedding_dim) * 0.1

    for step, bucket_id in enumerate(graph_data.unique_buckets[: config.max_steps]):
        bucket_id = bucket_id.item()
        events = collect_bucket_events(graph_data.src, graph_data.dst, graph_data.bucket, bucket_id)
        if not events:
            continue

        event_buffer_local.extend(events)
        x_pred = predict_state(x_state, target_predictor, edge_weight_local, gamma=config.gamma)

        hit5 = evaluate_hitk(x_pred, events, graph_data.item_ids, k=5)
        hit10 = evaluate_hitk(x_pred, events, graph_data.item_ids, k=10)
        mrr = evaluate_mrr(x_pred, events, graph_data.item_ids)
        ndcg10 = evaluate_ndcgk(x_pred, events, graph_data.item_ids, k=10)

        x_filtered = filtering_update(
            x_pred,
            events,
            graph_data.item_ids,
            beta=config.beta,
            num_competitors=config.num_competitors,
        )

        recent_events = event_buffer_local[-config.recent_event_window :]
        x_for_loss = predictor(x_state, edge_weight_local)
        x_norm = x_for_loss / (x_for_loss.norm(dim=1, keepdim=True) + 1e-6)

        loss = compute_bpr_loss(
            x_norm,
            recent_events,
            graph_data.item_ids,
            negative_sample_size=config.negative_sample_size,
        )
        loss += config.lambda_norm * (x_filtered.norm(dim=1) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()

        for param, target_param in zip(predictor.parameters(), target_predictor.parameters()):
            target_param.data = config.tau * target_param.data + (1 - config.tau) * param.data

        x_state = update_state(x_state, x_filtered, eta=config.eta)

        for user_id, item_id in events:
            if (user_id, item_id) in edge_weight_local:
                value, last_bucket = edge_weight_local[(user_id, item_id)]
                dt = bucket_id - last_bucket
                value = decay_edge_weight(value, dt, gamma=config.edge_decay_gamma)
            else:
                value = 0.0
            edge_weight_local[(user_id, item_id)] = (value + 1.0, bucket_id)

        if step % 50 == 0 or step == config.max_steps - 1:
            queue.put(
                {
                    "trial": trial_id,
                    "step": step,
                    "hit5": hit5,
                    "hit10": hit10,
                    "mrr": mrr,
                    "ndcg10": ndcg10,
                    "loss": float(loss.item()),
                }
            )

    queue.put({"trial": trial_id, "done": True})


def run_parallel_live(graph_data, config: ExperimentConfig):
    queue = Queue()
    processes = []
    for trial_idx in range(config.n_trials):
        process = Process(
            target=_run_trial_worker,
            args=(trial_idx, config.seed + trial_idx, queue, graph_data, config),
        )
        process.start()
        processes.append(process)

    results = {}
    done_count = 0

    while done_count < config.n_trials:
        msg = queue.get()
        if "done" in msg:
            done_count += 1
            continue

        step = msg["step"]
        if step not in results:
            results[step] = {"hit5": [], "hit10": [], "mrr": [], "ndcg10": [], "loss": []}

        for key in results[step]:
            results[step][key].append(msg[key])

        if len(results[step]["hit5"]) == config.n_trials:
            print("\n" + "=" * 70)
            print(f"Step {step:4d}")
            print("-" * 70)
            print(f"Hit@5   | mean {np.mean(results[step]['hit5']):.4f} | std {np.std(results[step]['hit5']):.4f}")
            print(f"Hit@10  | mean {np.mean(results[step]['hit10']):.4f} | std {np.std(results[step]['hit10']):.4f}")
            print(f"MRR     | mean {np.mean(results[step]['mrr']):.4f} | std {np.std(results[step]['mrr']):.4f}")
            print(
                f"NDCG@10 | mean {np.mean(results[step]['ndcg10']):.4f} | "
                f"std {np.std(results[step]['ndcg10']):.4f}"
            )
            print(f"Loss    | mean {np.mean(results[step]['loss']):.4f}")

    for process in processes:
        process.join()

    return {"config": asdict(config), "results": results}
