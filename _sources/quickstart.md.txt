# 🎬 Quickstart

**P2P Federated Learning (p2pfl)** is a library focused on the easy creation of federated learning systems. It offers flexibility in terms of **communication protocols**, **aggregation algorithms**, and **machine learning frameworks**, allowing you to tailor your federated learning setup to your specific needs.

We strongly recommend using [**p2pfl web services**](https://p2pfl.com) for simplified orchestration and management of your federated learning experiments. This web-based dashboard provides an intuitive interface for controlling nodes, monitoring progress, and visualizing results, making everything more visual and user-friendly.

## 💻 Using the CLI

P2PFL provides a command-line interface (**CLI**) to simplify running experiments. You can launch the CLI using `python -m p2pfl`. This interface provides a convenient way to explore and run different federated learning experiments without manually writing code for each node. You can easily switch to different communication protocols, aggregators, and ML frameworks.

### ⌨️ Main Commands

| Command        | Description                                                                    | Availability |
|----------------|--------------------------------------------------------------------------------|--------------|
| `experiment`   | Run experiments on the p2pfl platform.                                         | ✅            |
| `launch`       | Launch a new node in the p2pfl network.                                        | 🔜 Coming Soon |
| `login`        | Authenticate with the p2pfl platform using your API token.                     | 🔜 Coming Soon |
| `remote`       | Interact with a remote node in the p2pfl network.                              | 🔜 Coming Soon |


### 🧪 Example: `experiment` Command

This command allows you to interact with pre-built examples. Here's how you can use it:

* **List available examples:** `python -m p2pfl experiment list`. This will display a table of available examples with their descriptions.
* **Run an example:** `python -m p2pfl experiment run <example_name> [options]`.This will run the specified example.

For instance, to run the `mnist` example (which currently uses **FedAvg**, **PyTorch** and  **gRPC**) with 2 rounds and 1 epoch:

> Instead of **gRPC**, a local **memory based** protocol can be used with the additional `--use_local_protocol` flag.

```bash
python -m p2pfl experiment run mnist --rounds 2 --epochs 1
```

When the the mnist experiment finishes, the training results will be plotted on the screen.

## 💡 Manual Usage

For a more in-depth understanding of how p2pfl works, we recommend checking the source code of the examples (can be found [here](https://github.com/pguijas/p2pfl/tree/main/p2pfl/examples)). This hands-on approach allows you to explore the flexibility of p2pfl by experimenting with different communication protocols, aggregators, and ML frameworks.

### 🔥 PyTorch Quickstart (MNIST)

This tutorial demonstrates how to train a **Multilayer Perceptron (MLP)** on the **MNIST** dataset using **p2pfl** and **PyTorch**. We'll use a federated version of MNIST, distributed across two nodes.

> 🚧 **Note**: The code of the following nodes must be executed in order. First, run **Node 1**, then **Node 2**.

- **Node 1 (node1.py):** This node will simply start and wait for connections.

```python
from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP

# Start the node
node = Node(
    MLP(),
    MnistFederatedDM(sub_id=0, number_sub=2),
    address=f"127.0.0.1:{6666}" # Introduce a port or remove to use a random one
)
node.start()

input("Press any key to stop\n")

# Stop the node
node.stop()
```

- **Node 2 (node2.py):** This node will start, connect to Node 1, and initiate the federated learning process.

```python
from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
import time

# Start the node
node = Node(
    MLP(),
    MnistFederatedDM(sub_id=1, number_sub=2),
    address="127.0.0.1" # Random port
)
node.start()

# Connect to the first node
node.connect(f"127.0.0.1:{6666}") 

time.sleep(4)

# Start learning process
node.set_start_learning(rounds=2, epochs=1)

# Wait 4 results
while True:
    time.sleep(1)

    if node.round is None:
        break

# Stop the node
node.stop()
```
### 🧠 Brief Explanation

The example above showcases the fundamental components of p2pfl for creating a simple federated learning scenario. Let's break down the key elements:

* **`Node` Class:** This is the cornerstone of p2pfl, representing a single node in the federated learning network. It handles communication, model training, and aggregation.
* **`MLP` Class:** This defines the structure of your machine learning model, in this case, a Multilayer Perceptron.
* **`MnistFederatedDM` Class:** This provides a federated version of the MNIST dataset, distributing data across different nodes.
* **`node.connect()`:** This function establishes a connection between nodes, enabling them to communicate and share model updates.
* **`node.set_start_learning()`:** This initiates the federated learning process, defining the number of rounds and epochs for training.
* **`node.state.round is None`:** This condition is used to check if the learning process has finished.

### ❄️ TensorFlow Quickstart

> 🚧 Not implemented yet. 


