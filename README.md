# Data Assimilation Recommender

An online recommender system for dynamic graphs that combines graph-based prediction with a direct filtering update.

## Overview

Most GNN-based recommender systems learn from historical interactions and update representations through repeated training. This project takes a different view: recommendation on a dynamic graph can be treated as a prediction-correction problem.

At each time step, the model:

- predicts the next embedding state from the current graph structure
- corrects that state using newly observed interactions
- updates the system online as new data arrives

This framing brings ideas from data assimilation into dynamic recommendation, with the goal of making representation updates more explicit, adaptive, and efficient.

## Main Contribution

This project focuses on three main innovations.

### 1. Online updates for dynamic graph recommendation

Instead of treating recommendation as a static offline learning problem, this system models node embeddings as a time-evolving state. User-item interactions arrive sequentially, and the recommender updates online as the graph changes over time.

This makes the framework more suitable for real-world recommendation settings where preferences shift continuously and fresh observations should influence the model immediately.

### 2. Direct filtering step for efficient adaptation

The core difference from many GNN-based methods is the explicit filtering step after prediction.

- A GNN produces a learned prior over the embedding state
- A filtering operator then incorporates the newest observations directly
- The update reinforces observed user-item interactions while pushing away strong negative competitors

This prediction-filtering structure separates global model-based prediction from local observation-driven correction. In practice, this gives a more direct and efficient mechanism for adapting to new events than relying only on repeated graph propagation or retraining.

### 3. Experimental code for comparison with GNN-based baselines

The repository includes the implementation needed to evaluate the method on dynamic recommendation tasks using ranking-based metrics such as:

- `Hit@K`
- `MRR`
- `NDCG`

The experimental pipeline is designed for comparison against related GNN-based recommendation methods, with the goal of showing that the prediction-filtering approach can outperform comparable graph-based baselines on time-evolving interaction data. Full benchmark results can be expanded and documented as the experiments are finalized.

## Method Summary

The update rule is formulated as a prediction-correction process over node embeddings.

### Prediction

A graph neural network produces a prior estimate of the next embedding state using the previous state and graph structure.

### Correction

A filtering step incorporates newly observed interactions and directly adjusts embeddings based on the latest user-item events.

### State update

The corrected state is blended into the running embedding state to maintain an online representation of the system.

This design makes the interaction between learned graph structure and incoming observations explicit, rather than leaving all adaptation to a single end-to-end training procedure.

## Current Setting

The current implementation is built around a temporal recommendation setting using the Reddit dataset from the JODIE benchmark.

The pipeline includes:

- preprocessing to extract the largest connected component
- chronological sorting of interactions
- time bucket construction for sequential updates
- online embedding updates
- ranking-based evaluation for recommendation quality

## Repository Contents

- `index.html`: GitHub Pages project site
- `styles.css`: styling for the project site
- `main.py`: local placeholder Python file

## Project Direction

This repository is being developed as both:

- a research-style exploration of data assimilation for dynamic graph recommendation
- a portfolio project demonstrating recommender systems, graph learning, and online updating

Planned additions include cleaner training code, stronger baseline comparisons, and a dedicated results section with visualizations and benchmark tables.
