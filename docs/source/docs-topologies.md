## 🔀 Network Topologies

In a Peer-to-Peer Federated Learning network, the topology defines how Nodes connect and interact. The `TopologyFactory` provides an efficient way to create and manage various network structures, ensuring proper connectivity among Nodes.

### Supported Topologies

#### 1. Star Topology
- One central Node is directly connected to all other Nodes.
- **Use Case**: Centralized communication and control.
- **Example Adjacency Matrix**:
```
0 1 1 1
1 0 0 0
1 0 0 0
1 0 0 0
```


### 2. Full Topology
- Every Node is connected to every other Node.
- **Use Case**: Maximum redundancy and peer-to-peer interaction.
- **Example Adjacency Matrix**:
```
0 1 1 1
1 0 1 1
1 1 0 1
1 1 1 0
```

### 3. Line Topology
- Nodes are connected in a straight line.
- **Use Case**: Simulates sequential communication.
- **Example Adjacency Matrix**:
```
0 1 0 0
1 0 1 0
0 1 0 1
0 0 1 0
```


### 4. Ring Topology
- Nodes form a closed loop, where each Node is connected to two neighbors.
- **Use Case**: Efficient for circular data passing.
- **Example Adjacency Matrix**:
```
0 1 0 1
1 0 1 0
0 1 0 1
1 0 1 0
```

---

## Generating a Topology
To generate an adjacency matrix for a specific topology:

```python
from p2pfl.utils.topologies import TopologyType, TopologyFactory

# Define the topology type and number of Nodes
topology_type = TopologyType.STAR
num_nodes = 4

# Generate the adjacency matrix
adjacency_matrix = TopologyFactory.generate_matrix(topology_type, num_nodes)
print(adjacency_matrix)
```

---

## Example: Full Topology with 3 Nodes
1. Define the Nodes:
```python
nodes = [
    Node(address="127.0.0.1:5001"),
    Node(address="127.0.0.1:5002"),
    Node(address="127.0.0.1:5003"),
]
```

2. Generate Full Topology:
```python
adjacency_matrix = TopologyFactory.generate_matrix(TopologyType.FULL, len(nodes))
```

3. Connect the Nodes:
```python
TopologyFactory.connect_nodes(adjacency_matrix, nodes)
```

🌟 Ready? **You can view next**: > [Learners](docs-learners.md)

<div style="position: fixed; bottom: 10px; right: 10px; font-size: 0.9em; padding: 10px; border: 1px solid #ccc; border-radius: 5px;"> 🌟 You Can View Next: <a href="docs-learners.md">Learners</a> </div>