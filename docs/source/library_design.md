# 🏛️ Library Design

This document breafly describes the design of the library.

## Design principles and patterns

The use of design principles and patterns is essential for one of the main goals of the library: simplicity.

First of all, its important to keep in mind some design principles that were followed during the development of the library. Some of them are: `SOLID` principles, `KISS`, `DRY`, `YAGNI`, etc.

As for the design patterns, the main ones will be dealt with in the next section. Nevertheles, it is important to note that their use greatly facilitates decoupling and the respect of the main design principles. 

## General Overview

The core of the library is designed around a **modular architecture**, promoting flexibility and extensibility. 

Flexibility and extensibility are achieved through a high degree of decoupling and ease of customisation of different aspects of the library. Thus, the `Node` class acts as the central entity to the library. It makes use of different modules that globally allow the operation of a Federated Learning node. The following diagram details the main modules of the library. 

```{eval-rst}
.. mermaid::
    :align: center

    graph LR
        P2PFL_Node --- State["State"]
        P2PFL_Node["Node"] --- Communication["Communication Protocol"]
        P2PFL_Node --- Commands["Commands"]
        P2PFL_Node --- Learner["Learner"]
        P2PFL_Node --- Aggregator["Aggregator"]
        P2PFL_Node --- Workflow["Workflow"]

        Communication --- Grpc["gRPC"]
        Communication --- InMemory["In-Memory"]

        Learner --- PyTorch["PyTorch"]
        Learner --- TensorFlow["TensorFlow (Coming Soon)"]

        Aggregator --- FedAvg["FedAvg"]
        Aggregator --- Scaffold["Scaffold (Coming Soon)"]

        Workflow --- BaseNode["BaseNode"]
        Workflow --- ProxyNode["ProxyNode (Coming Soon)"]

    classDef notImplementedNode fill:#ddd, color:#777;
    class TensorFlow notImplementedNode;
    class ProxyNode notImplementedNode;
    class Scaffold notImplementedNode;

```

A brief description of each module will be given below:
- `State`: This module is responsible for managing the state of the node. It is used to store the node's information and to keep track of the node's state.
- `CommunicationProtocol`: This module is responsible for managing the communication between nodes. In terms of patterns, highlight the **template pattern**, which will make it possible to implement different protocols that share the same API. Also note the **command pattern**, which allows the creation of new commands that can be executed regardless of the protocol that manages communications.
- `Commands`: This module is responsible of the commands that can be executed over the `CommunicationProtocol`.
- `Learner`: This module is responsible for managing the learning process. Remark again the **template pattern**, which will make it possible to integrate different machine learning frameworks over the same API.
- `Aggregator`: This module is responsible for managing the aggregation process. It is possible to implement different aggregation strategies using the **strategy pattern**.
- `Workflow`: This module is responsible for managing the node's workflow. It is possible to implement different workflows using the **strategy pattern**.

## Main workflow

The main workflow of the library is as follows:

> TENGO QUE ACTUALZARLO, CAMBIADO | AGREGAR DESCRIPCIONES DE CADA ETAPA

```{eval-rst}
.. mermaid::
    :align: center

    graph LR
        A(StartLearningStage) --> B(VoteTrainSetStage)
        B -- Node in trainset? --> C(TrainStage)
        B -- Node not in trainset? --> D(WaitAggregatedModelsStage)
        C --> E(GossipModelStage) 
        D --> E
        E --> F(RoundFinishedStage)
        F -- No more rounds? --> Finished
        F -- More rounds? --> B
```