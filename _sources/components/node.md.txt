# 🌐 Node

The `Node` class is the core component of the P2PFL library, representing a participant in a decentralized federated learning network. It manages a node's dataset, model, local training process, and communication with peers, enabling collaborative training of machine learning models. Acting as a central hub, the `Node` class orchestrates the different components of the federated learning process, providing a simplified user interface and delegating the underlying logic.

## Main Functionality

### Initialization

The `Node` class instantiates all the necessary modules: [`State`](state.md), [`CommunicationProtocol`](communication.md), [`Commands`](commands.md), [`Learning`](learner/learner-index.md) and [`Workflow`](workflows.md). The user only needs to focus on selecting the specific modules they require, as shown below:

> By default, `GrpcCommunicationProtocol` and `FedAvg` are used. The `Learner` will be inferred from the `P2PFLModel` if not specified.

```python
from p2pfl.node import Node
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.learning.frameworks.pytorch.lightning_model import MLP 
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.communication.protocols.grpc.grpc_communication_protocol import GrpcCommunicationProtocol
from p2pfl.learning.aggregators.fedavg import FedAvg


node = Node(
    model=LightningModel(MLP()),  # Wrap your model (MLP in this example) in a LightningModel
    data=P2PFLDataset.from_huggingface("p2pfl/MNIST"),  # Load the MNIST dataset
    addr="127.0.0.1:5000",  # Specify the node's address (optional - a random port will be assigned if not provided)
    learner=LightningLearner, # Optional: Specify the learner class if needed
    aggregator=FedAvg(), # Optional: Specify the aggregator if needed
    protocol=GrpcCommunicationProtocol(), # Optional: Specify the communication protocol if needed
    simulation=True # Optional: Set to True if running a simulation
)

```

It's important to note that the node follows the workflow defined in [`LearningWorkflow`](#LearningWorkflow).

Once the node is instantiated, it simply needs to be started:

> Setting `wait` to `True` blocks the process until the node stops.

```python
node.start(wait=False)
```

To stop the node and its communication services:

```python
node.stop()
```

### Neighbor Management

The node allows connections to other nodes, delegating this logic to the `CommunicationProtocol`. To connect to or disconnect from another node, use `.connect` or `.disconnect` followed by the address of the target node:

```python
# Connect to a neighbor
node.connect("192.168.1.10:6666")

# Disconnect from a neighbor
node.disconnect("192.168.1.10:6666")
```

To obtain a list of connected nodes:

> Setting `only_direct` to `True` returns only directly connected neighbors.

```python
neighbors = node.get_neighbors(only_direct=True)
print(neighbors)
```

### Training Management

To start distributed learning across the network:

```python
node.set_start_learning(rounds=3, epochs=2)
```

To stop the learning process in the network:

```python
node.set_stop_learning()
```

### Model and Data Management

You can get and set the model and data for the learner using the following methods:

```python
# Set the model
node.set_model(new_model)

# Set the data
node.set_data(new_data)

# Get the current model
model = node.get_model()

# Get the current data
data = node.get_data()

# Set the number of epochs for local training
node.set_epochs(5)
```

These methods are useful for dynamically changing the model or data during the experiment, or for initializing the node with specific configurations.  Note that you cannot change the model or data while a training round is in progress.  Attempting to do so will raise a `LearnerRunningException`.

### Logging

The `Node` class interacts with the [`P2PFLogger`](logger.md) to record events and metrics during the federated learning process.  This logger is essential for monitoring training progress, debugging issues, and analyzing results.  It handles concurrent logging from multiple nodes and components, ensuring thread safety and providing contextual information for each log message.  It can also be configured to send logs to a remote server for centralized monitoring.

To access the logs generated during an experiment, you can use the following methods provided by the logger:

```python
from p2pfl.management.logger import logger

# Get local logs (metrics logged during local training epochs)
local_logs = logger.get_local_logs()

# Get global logs (metrics logged after each round, like test accuracy)
global_logs = logger.get_global_logs()

print(f"Local Logs: {local_logs}")
print(f"Global Logs: {global_logs}")
```

For more details about the logger and its functionalities, please refer to the [logger component](logger.md) section.
