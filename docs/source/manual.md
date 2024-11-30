# 📖 Manual

**P2P Federated Learning (p2pfl)** is a library focused on the easy creation of federated learning systems. It offers flexibility in terms of **communication protocols**, **aggregation algorithms**, and **machine learning frameworks**, allowing you to tailor your federated learning setup to your specific needs.

We strongly recommend using [**p2pfl web services**](https://p2pfl.com) for simplified orchestration and management of your federated learning experiments. This web-based dashboard provides an intuitive interface for controlling nodes, monitoring progress, and visualizing results, making everything more visual and user-friendly.

## ⚠️ Before Starting

> **Note**: For detailed installation instructions, please refer to the [**installation guide**](installation.md). It covers everything you need to install **p2pfl**, including options for users, developers, and advanced manual installations.

## 📋 Index

```{eval-rst}
.. toctree::
   :maxdepth: 4

```

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

For instance, to run the `mnist` example (which currently uses **FedAvg** as aggregator, **PyTorch** as framework and  **gRPC** as communication protocol) with 2 rounds of training with 1 epoch each:

```bash
python -m p2pfl experiment run mnist --rounds 2 --epochs 1
```

When the the mnist experiment finishes, the training results will be plotted on the screen.

> You can see experiment options by running the help command For example : `python -m p2pfl experiment help mnist`


## 💡 Manual Usage

For a more in-depth understanding of how p2pfl works, we recommend checking the source code of the examples (can be found [here](https://github.com/pguijas/p2pfl/tree/main/p2pfl/examples)). This hands-on approach allows you to explore the flexibility of p2pfl by experimenting with different communication protocols, aggregators, and ML frameworks.

### 🔥 PyTorch Quickstart (MNIST)

This tutorial demonstrates how to train a **Multilayer Perceptron (MLP)** on the **MNIST** dataset using **p2pfl** and **PyTorch**. We'll use a federated version of MNIST, distributed across two nodes.

> 🚧 **Note**: The code of the following nodes must be executed in order. First, run **Node 1**, then **Node 2**.

- **Node 1 (node1.py):** This node will simply start and wait for connections.

```python
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.frameworks.pytorch.lightning_model import MLP, LightningModel
from p2pfl.node import Node

# Start the node
node = Node(
    model = LightningModel(MLP()), # Wrap the MLP module into a LightningModel
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST"), # Get dataset from Hugging Face
    address= f"127.0.0.1:{6666}", # Introduce a port or remove to use a random one
)
node.start()

input("Press any key to stop\n")

# Stop the node
node.stop()
```

- **Node 2 (node2.py):** This node will start, connect to Node 1, and initiate the federated learning process.

```python
import time

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.frameworks.pytorch.lightning_model import MLP, LightningModel
from p2pfl.node import Node

# Start the node
node = Node(
    model = MLP(), # Multi Layer Perceptron model
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST"),
    address = "127.0.0.1", # Random port
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

    if node.state.round is None:
        break

# Stop the node
node.stop()
```


### 🔥 TensorFlow Quickstart

This tutorial demonstrates how to train a **Multilayer Perceptron (MLP)** on the **MNIST** dataset using **p2pfl** and **PyTorch**. We'll use a federated version of MNIST, distributed across two nodes.

> 🚧 **Note**: The code of the following nodes must be executed in order. First, run **Node 1**, then **Node 2**.

- **Node 1 (node1.py):** This node will simply start and wait for connections.
```python
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.tensorflow.keras_learner import KerasLearner
from p2pfl.learning.frameworks.tensorflow.keras_model import MLP, KerasModel
from p2pfl.node import Node

# Start the node
node = Node(
    model = KerasModel(MLP()), # Wrap the MLP module into a LightningModel
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST"), # Get dataset from Hugging Face
    address= f"127.0.0.1:{6666}", # Introduce a port or remove to use a random one
)
node.start()

input("Press any key to stop\n")

# Stop the node
node.stop()
```

- **Node 2 (node2.py):** This node will start, connect to Node 1, and initiate the federated learning process.

```python
import time

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.tensorflow.keras_learner import KerasLearner
from p2pfl.learning.frameworks.tensorflow.keras_model import MLP, KerasModel
from p2pfl.node import Node

# Start the node
node = Node(
    model = MLP(), # Multi Layer Perceptron model
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST"),
    address = "127.0.0.1", # Random port
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

    if node.state.round is None:
        break

# Stop the node
node.stop()
```




### 🔥 Flax Quickstart (MNIST)

This tutorial demonstrates how to train a **Multilayer Perceptron (MLP)** on the **MNIST** dataset using **p2pfl** and **Flax**. We'll use a federated version of MNIST, distributed across two nodes.

> 🚧 **Note**: The code of the following nodes must be executed in order. First, run **Node 1**, then **Node 2**.

- **Node 1 (node1.py):** This node will simply start and wait for connections.

```python
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.flax.flax_learner import FlaxLearner
from p2pfl.learning.frameworks.flax.flax_model import MLP, FlaxModel
from p2pfl.node import Node

# Start the node
node = Node(
    model = FlaxModel(MLP()), # Wrap the Multi Layer Perceptron into a FlaxModel
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST"), # Get dataset from Hugging Face
    address= f"127.0.0.1:{6666}", # Introduce a port or remove to use a random one
)
node.start()

input("Press any key to stop\n")

# Stop the node
node.stop()
```

- **Node 2 (node2.py):** This node will start, connect to Node 1, and initiate the federated learning process.

```python
import time

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.flax.flax_learner import FlaxLearner
from p2pfl.learning.frameworks.flax.flax_model import MLP, FlaxModel
from p2pfl.node import Node

# Start the node
node = Node(
    model = FlaxModel(MLP()), # Wrap the Multi Layer Perceptron into a FlaxModel
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST"),
    address = "127.0.0.1", # Random port
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

    if node.state.round is None:
        break

# Stop the node
node.stop()
```
### 🧠 Brief Explanation

The example above showcases the fundamental components of p2pfl for creating a simple federated learning scenario. Let's break down the key elements:

* **`Node` Class:** This is the cornerstone of p2pfl, representing a single node in the federated learning network. It handles communication, model training, and aggregation.
* **`MLP` Class:** This defines the structure of your machine learning model, in this case, a Multilayer Perceptron.
* **`LightingModel`/ `KerasModel`/`FlaxModel` classes** : A P2PFLModel wrapper to use your custom model in the corresponding frameworks.
* **`MnistFederatedDM` Class:** This provides a federated version of the MNIST dataset, distributing data across different nodes.
* **`node.connect()`:** This function establishes a connection between nodes, enabling them to communicate and share model updates.
* **`node.set_start_learning()`:** This initiates the federated learning process, defining the number of rounds and epochs for training.
* **`node.state.round is None`:** This condition is used to check if the learning process has finished.

### 🌐 Ray Integration for Distributed Learning

**Ray** is integrated seamlessly into **p2pfl** to enhance the scalability and efficiency of federated learning by automating the distribution of tasks across available computing resources. Once Ray is installed, it is used automatically without requiring any additional configuration from the user, allowing for efficient parallelism and task management across the network of nodes.

#### 🧩 Cluster Setup

To get started with Ray, the user can set up a cluster of nodes. This setup typically involves configuring the available machines or instances to communicate with each other. After this, Ray automatically takes care of distributing tasks among the nodes, freeing the user from the complexity of manual task management.

Here are the basic steps to set up the cluster:

1. **Start the Ray cluster**: On the head node (the central coordinator):
   ```bash
   ray start --head
   ```
   On worker nodes (additional machines):
   ```bash
   ray start --address='head_node_ip:6379'
   ```

2. **Verify the cluster**: Check the status of the cluster using the following command:
   ```bash
   ray status
   ```

3. **Run p2pfl with Ray**: Once the cluster is set up, Ray will automatically manage the distribution of tasks, and the federated learning process can begin by running the appropriate commands or starting the Node processes as described in the previous examples.