# 🏛️ Library Design

This document outlines the design of the p2pfl library, explaining its core components, patterns, and best practices.

While understanding this design isn't strictly necessary to use p2pfl, it's beneficial for effective utilization and especially crucial for extending the library's functionality. This document will help you grasp how p2pfl works under the hood and how to customize or expand it for your specific needs.

## Design principles and patterns

Simplicity and extensibility are at the heart of p2pfl's design. We achieve this through a set of guiding principles and the strategic application of design patterns. Our key design principles are:

* **SOLID:**: Ensures modularity, flexibility, and maintainability, making the library adaptable and easy to evolve.
* **KISS (Keep It Simple, Stupid):** Prioritizing straightforward solutions and minimizing complexity.
* **DRY (Don't Repeat Yourself):**  Reducing code duplication and promoting reusability.
* **YAGNI (You Ain't Gonna Need It):** Focusing on current requirements and avoiding unnecessary features.

Design patterns are essential for decoupling components and upholding these principles. We'll explain the patterns used for each major architectural component in the following subsections.

## General Overview

P2PFL is built on a **modular architecture** centered around the `Node` class. This modularity provides flexibility and extensibility, allowing you to easily customize and integrate different components. The `Node` class interacts with various modules to orchestrate the operations of a federated learning node.

```{eval-rst}
.. mermaid::
    :align: center

    graph LR
        P2PFL_Node["Node"] --- Logger["Logger"]
        P2PFL_Node["Node"] --- State["State"]
        P2PFL_Node --- Communication["Communication Protocol"]
        P2PFL_Node --- Commands["Commands"]
        P2PFL_Node --- Learner["Learner"]
        P2PFL_Node --- Aggregator["Aggregator"]
        P2PFL_Node --- Workflow["Workflow"]

        Communication --- Grpc["gRPC"]
        Communication --- InMemory["In-Memory"]
        Communication --- Unix["Unix Sockets"]

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

Let's briefly introduce each module:

* [`Node`](node.md): The core component, orchestrating the federated learning process. It acts as the central hub, interacting with all other modules to manage the node's lifecycle and operations.
* [`Logger`](logger.md): Handles logging and tracking of training events and metrics. Provides valuable information about node behavior and assists in debugging and monitoring.
* [`State`](state.md): Manages the node's internal state, storing essential information and tracking its progress throughout the federated learning process.
* [`CommunicationProtocol`](communication.md): Facilitates communication between nodes, enabling them to exchange models, gradients, and other necessary data. It uses the **template pattern** to support various communication protocols while maintaining a consistent API.
* [`Command`](commands.md): Deines the set of commands that can be executed over the `CommunicationProtocol`. It leverages the **command pattern**, allowing the creation of new commands that can be executed independently of the underlying communication protocol.
* [`Learner`](learner.md): Handles the machine learning aspects, encapsulating the training and evaluation of models. It employs the **template pattern** to seamlessly integrate different machine learning frameworks (like PyTorch, TensorFlow, or Flax) under a unified API.
* [`Aggregator`](aggregator.md): Responsible for aggregating model updates from different nodes. It uses the **strategy pattern**, making it easy to implement and switch between various aggregation strategies (e.g., FedAvg, Scaffold, FedMedian).
* [`Workflow`](workflow.md): Defines the sequence of steps a node takes during the federated learning process. It also uses the **strategy pattern** to support different workflows, allowing for customization based on specific requirements (e.g., BaseNode workflow, ProxyNode workflow).

## 📋 Modules

```{eval-rst}
.. toctree::

    node
    logger
    state
    communication
    commands
    learners
    aggregators
    workflows
```
