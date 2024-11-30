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
