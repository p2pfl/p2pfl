# 🔄 Workflows

Workflows in P2PFL define the sequence of operations performed by each node during a federated learning experiment. They provide a structured approach to the training process, ensuring that all nodes follow a consistent set of steps.  Currently, P2PFL implements a single core workflow, [`LearningWorkflow`](#LearningWorkflow), which governs the lifecycle of a federated learning experiment.

## Learning Workflow

The [`LearningWorkflow`](#LearningWorkflow) orchestrates the training process, coordinating actions such as model initialization, training set selection, local training, model aggregation, and evaluation.  It uses a series of stages, each responsible for a specific part of the workflow. The workflow's progression is illustrated in the following diagram:

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

### Stages

1. **[`StartLearningStage`](#StartLearningStage):**  Initializes the federated learning process.  This includes setting up the experiment, initializing the model, and gossiping the initial model parameters to all nodes.

2. **[`VoteTrainSetStage`](#VoteTrainSetStage):** Nodes vote to select a subset of nodes (`train_set`) that will participate in the current training round.  This stage ensures that not all nodes need to participate in every round, which can improve efficiency and scalability.

3. **[`TrainStage`](#TrainStage):**  Nodes in the **train set** perform local training on their datasets, evaluate their local models, and contribute their updates to the aggregation process.

4. **[`WaitAggregatedModelsStage`](#WaitAggregatedModelsStage):** Nodes not participating in the current training round wait for the aggregated model from their neighbors.

5. **[`GossipModelStage`](#GossipModelStage):** All nodes gossip their models (either locally trained or aggregated) to their neighbors. This dissemination of model updates ensures eventual convergence across the decentralized network.

6. **[`RoundFinishedStage`](#RoundFinishedStage):**  Marks the end of a training round.  If more rounds are remaining, the workflow loops back to the [`VoteTrainSetStage`](#VoteTrainSetStage).  Otherwise, the experiment concludes, and final evaluation metrics are calculated.

This workflow ensures a structured and coordinated training process across all nodes in the decentralized network.  The use of stages and the voting mechanism for training set selection provide flexibility and scalability.
