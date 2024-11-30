# 🏛️ Node

## Overview
The `Node` class represents a federated learning node within a Peer-to-Peer Federated Learning (P2PFL) network. Each node is capable of training machine learning models collaboratively with other nodes in a distributed environment.

## Features
- **Model Initialization**: Set up machine learning models for federated learning.
- **Dataset Handling**: Assign datasets for training and testing.
- **Communication**: Connect and interact with neighboring nodes.
- **Learning Management**: Start and stop distributed learning processes.
- **State Management**: Maintain and manage the state of learning and network activities.

---

## Initialization

To create a `Node`:

```python
node = Node(
    model=MLP(),
    data=MnistFederatedDM(),
    address="127.0.0.1"
)
```

Parameters:
- `model`: A machine learning model implementing the `P2PFLModel` interface.
- `data`: A dataset implementing the `P2PFLDataset` interface.
- `address (optional)`: Node's network address. Defaults to `127.0.0.1`.
- `learner (optional)`: Custom `Learner` class. Defaults to a learner from `LearnerFactory`.
- `aggregator (optional)`: Aggregator for model updates. Defaults to `FedAvg`.
- `protocol (optional)`: Communication protocol class, defaulting to `GrpcCommunicationProtocol`.
- `simulation (optional)`: Whether the node operates in simulation mode. Default is `False`.

## Node Lifecycle

### Starting the Node
To start the node's services:

```python
node.start(wait=False)
```

- `wait`: If `True`, blocks the process until the node stops.

### Stopping the Node
To stop the node and its communication services:

```python
node.stop()
```

## Neighbor Management

### Connect to Another Node
Link the Node to a specific peer:

```python
node.connect("192.168.1.10:666")
```

- `addr`: Address of the target node.

### Get Neighbors
Retrieve a list of connected peers:
```python
neighbors = node.get_neighbors(only_direct=True)
```

- `only_direct`: If `True`, returns only direct neighbors.

### Disconnect from a Node
Remove a specific peer connection:

```python
node.disconnect("192.168.1.10:666")
```

- `addr`: Address of the target node.

## Learning Workflow
### Starting Learning
Start distributed learning across the network:

```python
node.set_start_learning(rounds=3, epochs=2)
```

- `rounds`: Number of training rounds.
- `epochs`: Number of epochs per round.

### Stopping Learning
Stop the learning process in the network:

```python
node.set_stop_learning()
```

## Node State Overview
The `NodeState` keeps track of important information, such as:

- `Status`: Current activity (e.g., Idle, Learning).
- `Round Information`: Current and total rounds of the experiment.
- `Neighbors`: Tracks peer connections and their statuses.
- `Locks and Events`: Manage synchronization during training.

🌟 Ready? **You can view next**: > [Topologies Guide](docs-topologies.md)

<div style="position: fixed; bottom: 10px; right: 10px; font-size: 0.9em; padding: 10px; border: 1px solid #ccc; border-radius: 5px;"> 🌟 You Can View Next: <a href="docs-topologies.md">Topologies Guide</a> </div>