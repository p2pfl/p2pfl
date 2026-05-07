# Hierarchical Federated Learning (HFL) - Design Document

**Status:** Implemented
**Date:** 2026-02-24
**Updated:** 2026-04-23

---

## 1. Motivation

The base p2pfl system implements a fully decentralized (flat) peer-to-peer federated learning workflow where every node is equal: all nodes train, gossip partial models, and aggregate to reach global consensus. While this works well for small-to-medium networks, it has limitations in larger deployments:

- **Communication overhead:** Every node must exchange models with every other node in the training set via gossip, leading to O(n^2) communication cost.
- **Scalability:** As the number of nodes grows, the gossip convergence time increases significantly.
- **Network topology mismatch:** In real-world deployments (e.g., IoT, mobile edge computing), nodes are naturally organized in clusters with a local gateway or edge server. The flat P2P topology does not reflect this physical structure.

**Hierarchical Federated Learning (HFL)** introduces a three-level hierarchy: **workers** train locally and send models to their **edge** node, edges aggregate worker models and forward the result to a **root** node, and the root aggregates all edge models and distributes the global model back down through the hierarchy. This reduces cross-cluster communication from O(n^2) to O(n) and matches real-world network topologies.

---

## 2. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Role assignment** | `role` field in `HFLContext` | Roles are configured at the workflow level via a typed context, keeping the `Node` class untouched. |
| **Edge training** | Configurable (`edge_trains` flag) | Edges can either train on their own data + aggregate workers, or act as pure aggregators. Configurable per experiment in the YAML. |
| **Inter-edge communication** | Centralized via root node | Edges send their aggregated model to a root node. The root aggregates and distributes back. No gossip between edges. |
| **Hierarchy depth** | 3 levels (workers -> edges -> root) | Covers the primary use case with a clear separation of concerns. |
| **Aggregation algorithm** | Configurable (default: FedAvg) | Same aggregator is used at both levels (edge and root). FedAvg computes a weighted average by number of training samples. |
| **Zero changes to core** | `Node`, `NodeState`, existing workflows untouched | HFL is a pure workflow extension. |

---

## 3. Architecture Overview

### 3.1 Changes from the base p2pfl (flat P2P)

The HFL workflow is implemented as a new workflow alongside the existing `basic` and `async` workflows. The following components were added or modified:

#### New files

| File | Description |
|---|---|
| `p2pfl/workflow/hfl/context.py` | `HFLContext` dataclass extending `WorkflowContext` with role, topology fields, and `edge_trains` flag |
| `p2pfl/workflow/hfl/workflow.py` | `HFL` workflow class with stage registration, context creation, and validation |
| `p2pfl/workflow/hfl/stages/setup.py` | Branches execution by role: worker, edge, or root |
| `p2pfl/workflow/hfl/stages/worker_train.py` | Worker trains locally and sends model to its edge |
| `p2pfl/workflow/hfl/stages/worker_wait_global.py` | Worker waits for the global model from its edge |
| `p2pfl/workflow/hfl/stages/edge_local_train.py` | Edge trains on its own data (skipped when `edge_trains: false`) |
| `p2pfl/workflow/hfl/stages/edge_aggregate_workers.py` | Edge collects and aggregates worker models |
| `p2pfl/workflow/hfl/stages/edge_sync_root.py` | Edge sends aggregated model to root and waits for global model back |
| `p2pfl/workflow/hfl/stages/edge_distribute.py` | Edge distributes the global model to its workers |
| `p2pfl/workflow/hfl/stages/root_aggregate.py` | Root collects and aggregates edge models |
| `p2pfl/workflow/hfl/stages/root_distribute.py` | Root distributes the global model to all edges |
| `p2pfl/workflow/hfl/stages/round_finished.py` | Shared stage: resets state, advances round, loops or finishes |
| `p2pfl/examples/mnist/mnist_hfl.yaml` | Example YAML configuration for running HFL with MNIST |
| `p2pfl/examples/tiny_diffusion/dataset.py` | Synthetic 2D dataset generator (moons, circles, spiral, cross) |
| `p2pfl/examples/tiny_diffusion/model/diffusion_pytorch.py` | Tiny DDPM denoiser model (LightningModule) for 2D point distributions |
| `p2pfl/examples/tiny_diffusion/tiny_diffusion_hfl.yaml` | YAML configuration for running HFL with the tiny diffusion model |
| `p2pfl/management/csv_exporter.py` | CSV exporter for experiment metrics and communication logs |

#### Modified files

| File | Change |
|---|---|
| `p2pfl/workflow/factory.py` | Registered `"hfl"` in the workflow registry |
| `p2pfl/management/launch_from_yaml.py` | Added `_run_hierarchical()` for creating root/edge/worker topology from YAML |
| `p2pfl/management/cli.py` | Support for multiple YAML files per example directory |
| `p2pfl/management/launch_from_yaml.py` | Added automatic CSV export after each experiment |

#### Not modified

- `Node`, `NodeState` — unchanged
- `BasicDFL`, `AsyncDFL` workflows — unchanged
- `Aggregator`, `Learner`, `CommunicationProtocol` — unchanged

---

## 4. How HFL Training Works

### 4.1 Network topology

```
                    ┌──────┐
                    │ ROOT │  (pure aggregator)
                    └──┬───┘
                 ┌─────┴─────┐
              ┌──┴──┐     ┌──┴──┐
              │EDGE0│     │EDGE1│  (aggregate workers, optionally train)
              └──┬──┘     └──┬──┘
             ┌───┴───┐   ┌───┴───┐
            W0      W1  W2      W3  (train locally)
```

- **Workers** connect to their assigned edge (bidirectional).
- **Edges** connect to the root (bidirectional).
- **No connections** between edges, nor between workers of different clusters.

### 4.2 Stage pipelines

Three roles execute different stage sequences within the same workflow:

```
Worker:  setup → worker_train → worker_wait_global → round_finished → loop
Edge:    setup → [edge_local_train →] edge_aggregate_workers → edge_sync_root → edge_distribute → round_finished → loop
Root:    setup → root_aggregate → root_distribute → round_finished → loop
```

The `edge_local_train` stage is only executed when `edge_trains: true` (default).

### 4.3 Training round with `edge_trains: true`

In this mode, edges have their own data partition and participate in training alongside their workers.

```
Step 1: Local training (parallel)
    Workers W0..W3:  evaluate → train → send model to edge ("worker_model")
    Edges E0, E1:    evaluate → train on own data → store own model

Step 2: Edge aggregation
    Each edge:  collect worker models + own model → FedAvg → flatten contributors to [edge_addr]

Step 3: Root aggregation
    Each edge:  send aggregated model to root ("edge_model")
    Root:       collect all edge models → FedAvg → global model

Step 4: Distribution (top-down)
    Root:       send global model to each edge ("root_global_model")
    Each edge:  send global model to each worker ("global_model")

Step 5: Round complete
    All nodes:  reset peer state, advance round counter, loop or finish
```

**Message flow:** `worker_model` (W→E) → `edge_model` (E→R) → `root_global_model` (R→E) → `global_model` (E→W)

### 4.4 Training round with `edge_trains: false`

In this mode, edges act as pure aggregators/relays. They don't train and don't contribute their own model to the aggregation.

```
Step 1: Local training (parallel, workers only)
    Workers W0..W3:  evaluate → train → send model to edge ("worker_model")
    Edges E0, E1:    go directly to edge_aggregate_workers (no training)

Step 2: Edge aggregation
    Each edge:  collect worker models only (own model excluded) → FedAvg → flatten contributors

Step 3-5: Same as edge_trains: true
```

**Differences:**
- Edges skip `edge_local_train`, going from `setup` directly to `edge_aggregate_workers`.
- Edge models are not included in the aggregation (`_all_worker_models_received` does not expect `ctx.address`).
- Edges only evaluate at the end of the experiment (final round), not every round.

### 4.5 Aggregation details

Both levels use the same aggregator (configured in YAML, default FedAvg):

```
                     Level 1 (edge)                          Level 2 (root)

Input:    [model_w0, model_w1, (model_edge)]    [model_edge0, model_edge1]

FedAvg:   global = Σ(model_i × samples_i)       global = Σ(model_i × samples_i)
                   ─────────────────────                  ─────────────────────
                      Σ(samples_i)                           Σ(samples_i)

Output:   aggregated edge model                  global model
```

After edge aggregation, contributors are **flattened** to `[edge_addr]` so the root tracks by edge address, not individual workers.

---

## 5. Context and Configuration

### 5.1 HFLContext fields

```python
@dataclass
class HFLContext(WorkflowContext):
    role: str = "worker"                           # "worker", "edge", or "root"
    edge_addr: str | None = None                   # For workers: assigned edge
    worker_addrs: list[str] = field(...)           # For edges: managed workers
    root_addr: str | None = None                   # For edges: root node address
    child_edge_addrs: list[str] = field(...)       # For root: child edge addresses
    edge_trains: bool = True                       # Whether edges train on their own data
    peers: dict[str, HFLPeerState] = field(...)    # Per-peer mutable state
```

### 5.2 YAML configuration

```yaml
network:
  protocol: "MemoryCommunicationProtocol"
  topology: "hierarchical"
  hierarchy:
    edge_trains: true          # false = edges are pure aggregators
    clusters:
      - workers: 2             # Cluster 0: 1 edge + 2 workers
      - workers: 2             # Cluster 1: 1 edge + 2 workers
      # Total nodes: 1 (root) + 2×(1 edge + 2 workers) = 7

experiment:
  workflow: hfl
  rounds: 10
  epochs: 5
```

### 5.3 Node assignment

Nodes are created sequentially and assigned roles by position:

```
nodes[0]           → root
nodes[1]           → edge 0
nodes[2], nodes[3] → workers of edge 0
nodes[4]           → edge 1
nodes[5], nodes[6] → workers of edge 1
```

---

## 6. Experimental Results (MNIST, MLP, 10 rounds, 5 epochs)

### With `edge_trains: true` (6 training nodes)

| Round | Accuracy range (all nodes) |
|---|---|
| 0 | 7-13% (random) |
| 1 | ~13% (first aggregation, destructive mixing) |
| 2 | 92-94% |
| 5 | 97-98% |
| 10 | 97.4-98.4% |

### With `edge_trains: false` (4 training nodes)

| Round | Accuracy range (workers only) |
|---|---|
| 0 | 10-14% (random) |
| 1 | ~29% (cleaner first aggregation) |
| 2 | 93-95% |
| 5 | 97-98% |
| 10 | 97.5-98.5% |

Both modes converge to the same final accuracy (~98%). The `edge_trains: false` mode shows slightly better early convergence (round 1: 29% vs 13%) because the first aggregation mixes fewer heterogeneous models.

---

## 7. CSV Export

After each experiment, metrics and communication logs are automatically exported to CSV files in a timestamped subdirectory.

### 7.1 Configuration

```yaml
results:
  output_dir: "results/hfl_mnist"    # Base directory; each run creates a timestamped subdirectory
```

### 7.2 Output files

Each run creates a directory like `results/hfl_mnist/20260423_153012/` with:

- **`metrics.csv`** — Global metrics per node and round.

  | Column | Description |
  |---|---|
  | `experiment` | Experiment name |
  | `node` | Node address |
  | `metric` | Metric name (e.g., `test_loss`, `accuracy`) |
  | `round` | Round number |
  | `value` | Metric value |

- **`communication.csv`** — All messages sent/received with byte sizes.

  | Column | Description |
  |---|---|
  | `timestamp` | When the message was sent/received |
  | `source` | Source node |
  | `destination` | Destination node |
  | `direction` | `sent` or `received` |
  | `cmd` | Command type (e.g., `worker_model`, `beat`) |
  | `package_type` | `message` or `weights` |
  | `package_size` | Size in bytes |
  | `round` | Round number |

---

## 8. Tiny Diffusion Experiment (2D point distributions)

### 8.1 Motivation

Beyond classification (MNIST), we validate HFL with a **generative model**: a tiny DDPM (Denoising Diffusion Probabilistic Model) that learns 2D point distributions. This experiment demonstrates that HFL can aggregate knowledge from non-IID generative data, where each cluster only observes a single distribution shape but the global model learns to generate all shapes.

### 8.2 Dataset

Four synthetic 2D distributions, each normalized to [-1, 1]:

| Label | Distribution | Description |
|---|---|---|
| 0 | Moons | Two interleaving half-circles |
| 1 | Circles | Two concentric circles |
| 2 | Spiral | Single Archimedean spiral |
| 3 | Cross | Four Gaussian blobs at cardinal positions |

- 2000 points per distribution (8000 total), Gaussian noise σ=0.05.
- 80/20 train/test split.
- Partitioned with `DirichletPartitionStrategy(alpha=0.1)` for strong non-IID: each worker's partition is dominated by one distribution type.

Dataset is generated on-the-fly via `source: "custom"` in the YAML (no external download).

### 8.3 Model architecture

Tiny DDPM with an MLP denoiser:

| Component | Details |
|---|---|
| Input | 2D noisy point `x_t` + timestep `t` |
| Time embedding | Sinusoidal (dim=64) → Linear(64, 128) → SiLU |
| Denoiser | MLP: Linear(2+128, 128) → [SiLU → Linear(128, 128)] × 3 → Linear(128, 2) |
| Diffusion steps | T=100, linear beta schedule (1e-4 to 0.02) |
| Optimizer | Adam, lr=1e-3 |
| Loss | MSE between predicted and true noise |

Total parameters: ~100K (lightweight enough for fast federated rounds).

### 8.4 HFL topology

```
                    ┌──────┐
                    │ ROOT │
                    └──┬───┘
           ┌────┬────┼────┬────┐
        ┌──┴──┐ ... (4 edges)
        │EDGE0│
        └──┬──┘
           │
          W0
```

- 9 nodes: 1 root + 4 edges + 4 workers (1 worker per cluster).
- `edge_trains: false` — edges are pure aggregators.
- 20 rounds, 10 epochs per round.

### 8.5 Experimental results (20 rounds, 10 epochs)

| Node | Role | Round 0 loss | Round 10 loss | Round 20 loss |
|---|---|---|---|---|
| node_2 | worker (cluster 0) | 1.005 | 0.387 | 0.447 |
| node_4 | worker (cluster 1) | 0.920 | 0.508 | 0.707 |
| node_6 | worker (cluster 2) | 0.803 | 0.344 | 0.227 |
| node_8 | worker (cluster 3) | 1.037 | 0.590 | 0.510 |

Communication: 11,761 messages, ~193 MB total (99.7% model weights).

The denoising loss decreases across rounds for all workers, showing that FedAvg aggregation at both edge and root levels successfully transfers knowledge across clusters. Each worker's model improves its ability to denoise points from distributions it has never directly observed.
