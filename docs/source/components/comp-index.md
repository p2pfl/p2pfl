# 🏛️ Components

```{eval-rst}
.. mermaid::
    :align: center

    graph TD
        P2PFL_Node["Node"] --- Logger["Logger"]
        P2PFL_Node["Node"] --- State["State"]
        P2PFL_Node --- Communication["Communication Protocol"]
        P2PFL_Node --- Commands["Commands"]
        P2PFL_Node --- Learning["Learning"]
        P2PFL_Node --- Workflow["Workflow"]

        Learning --- Learner["Learner"]
        Learning --- Dataset["Dataset"]
        Learning --- Aggregator["Aggregator"]

        Communication --- Grpc["gRPC"]
        Communication --- InMemory["In-Memory"]

        Learner --- PyTorch["PyTorch"]
        Learner --- TensorFlow["TensorFlow"]
        Learner --- Flax["Flax"]

        Aggregator --- FedAvg["FedAvg"]
        Aggregator --- Scaffold["Scaffold"]
        Aggregator --- FedMedian["FedMedian"]


        Workflow --- BaseNode["BaseNode Workflow"]
        Workflow --- ProxyNode["ProxyNode Workflow"]

        classDef notImplementedNode fill:#ddd, color:#777;
        class ProxyNode notImplementedNode;

```

P2PFL is built on a **modular architecture** centered around the `Node` class. This modularity provides flexibility and extensibility, allowing you to easily customize and integrate different components. The `Node` class interacts with various modules to orchestrate the operations of a federated learning node.

Let's briefly introduce each module:

* [`Node`](node.md): The core component, orchestrating the federated learning process. It acts as the central hub, interacting with all other modules to manage the node's lifecycle and operations.
* [`Logger`](logger.md): Handles logging and tracking of training events and metrics. Provides valuable information about node behavior and assists in debugging and monitoring.
* [`State`](state.md): Manages the node's internal state, storing essential information and tracking its progress throughout the federated learning process.
* [`CommunicationProtocol`](communication.md): Facilitates communication between nodes, enabling them to exchange models, gradients, and other necessary data. It uses the **template pattern** to support various communication protocols while maintaining a consistent API.
* [`Command`](commands.md): Deines the set of commands that can be executed over the `CommunicationProtocol`. It leverages the **command pattern**, allowing the creation of new commands that can be executed independently of the underlying communication protocol.
* [`Learner`](learner/learners.md): Handles the machine learning aspects, encapsulating the training and evaluation of models. It employs the **template pattern** to seamlessly integrate different machine learning frameworks (like PyTorch, TensorFlow, or Flax) under a unified API.
* [`Dataset`](learner/datasets.md): Manages the dataset used for training the models. It provides an interface to load, preprocess, and distribute data across nodes.
* [`Aggregator`](learner/aggregators.md): Responsible for aggregating model updates from different nodes. It uses the **strategy pattern**, making it easy to implement and switch between various aggregation strategies (e.g., FedAvg, Scaffold, FedMedian).
* [`Workflow`](workflows.md): Defines the sequence of steps a node takes during the federated learning process. It also uses the **strategy pattern** to support different workflows, allowing for customization based on specific requirements (e.g., BaseNode workflow, ProxyNode workflow).

## 📋 Modules

```{eval-rst}
.. toctree::
    :maxdepth: 1

    node
    learner/learner-index
    workflows
    state
    logger
    communication
    commands
```
