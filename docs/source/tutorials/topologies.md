# 🔀 Network Topologies

In P2PFL, the network topology defines how nodes are interconnected and communicate during federated learning. P2PFL provides the `TopologyFactory` to easily create and manage various network structures, allowing you to simulate different decentralized learning scenarios. **Crucially, these topologies are represented using adjacency matrices**, which provide a clear and efficient way to define connections between nodes.

## Adjacency Matrices: The Foundation of Network Topologies

An adjacency matrix is a square matrix used to represent a finite graph. The elements of the matrix indicate whether pairs of vertices (in our case, nodes) are adjacent or not in the graph. In P2PFL:

*   A value of `1` at position `(i, j)` indicates that node `i` is connected to node `j`.
*   A value of `0` indicates no direct connection.
*   The connections are bidirectional, so the matrix must be symmetric.

Understanding adjacency matrices is key to working with network topologies in P2PFL, as they are the underlying representation used by the `TopologyFactory` to define and manage connections.

## Supported Topologies

P2PFL currently supports the following topologies:

| Topology      | Description                                                                                                                                                              | Use Case                                                                                                    |
| :------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| **Star**      | A central node is directly connected to all other nodes. Other nodes are not directly connected to each other.                                                           | Simulates a scenario with a central coordinator, useful for comparison with traditional federated learning.                         |
| **Full**      | Every node is directly connected to every other node.                                                                                                                   | Provides maximum redundancy and direct peer-to-peer communication, but can be less efficient for large networks.                     |
| **Line**      | Nodes are connected sequentially in a single line. Each node (except the first and last) has exactly two neighbors.                                                       | Useful for simulating scenarios where communication is constrained or ordered, like a chain of devices.                               |
| **Ring**      | Similar to a line, but the first and last nodes are connected, forming a closed loop. Each node has exactly two neighbors.                                                | Suitable for scenarios where data needs to be passed around in a circular fashion, ensuring all nodes receive updates eventually. |

> **Note:** Other topologies like **Grid**, **Random**, and **Erdos-Renyi** are planned for future implementation.

## Creating Topologies with `TopologyFactory`

The `TopologyFactory` class simplifies the creation of network topologies by automatically generating the corresponding adjacency matrices.

### Adjacency Matrix

Let's say you want to simulate a federated learning experiment with five nodes arranged in a **star** topology. Here's how you can generate the adjacency matrix:

```python
from p2pfl.utils.topologies import TopologyFactory, TopologyType

# Define the topology type and number of nodes
topology = TopologyType.STAR
num_nodes = 5

# Generate the adjacency matrix
adjacency_matrix = TopologyFactory.generate_matrix(topology, num_nodes)
print(adjacency_matrix)
```

This will output a 5x5 adjacency matrix representing a star topology, where the first node (index 0) is the central node connected to all others:

```
[[0 1 1 1 1]
 [1 0 0 0 0]
 [1 0 0 0 0]
 [1 0 0 0 0]
 [1 0 0 0 0]]
```

### Connecting Nodes

Once you have an adjacency matrix, you can use the `TopologyFactory.connect_nodes` method to establish connections between your `Node` instances:

```python
from p2pfl.node import Node
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.learning.frameworks.pytorch.lightning_model import MLP, LightningModel
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

# Create some nodes
nodes = [
    Node(LightningModel(MLP()), P2PFLDataset.from_huggingface("p2pfl/MNIST"), addr="127.0.0.1:5001"),
    Node(LightningModel(MLP()), P2PFLDataset.from_huggingface("p2pfl/MNIST"), addr="127.0.0.1:5002"),
    Node(LightningModel(MLP()), P2PFLDataset.from_huggingface("p2pfl/MNIST"), addr="127.0.0.1:5003"),
]

# Generate a full topology
adjacency_matrix = TopologyFactory.generate_matrix(TopologyType.FULL, len(nodes))

# Connect the nodes
TopologyFactory.connect_nodes(adjacency_matrix, nodes)
```

This code will create the necessary connections between your nodes, effectively implementing the full topology you defined by the adjacency matrix.
