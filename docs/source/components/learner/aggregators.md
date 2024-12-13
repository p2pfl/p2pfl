# 🧩 Aggregators

Aggregators are responsible for combining model updates from multiple nodes during the federated learning process. They play a crucial role in ensuring the model converges effectively and efficiently, orchestrating how nodes collaborate in a decentralized manner.


## Key Features

*   **Model Aggregation**: Aggregators combine updates received from multiple nodes to produce a new, improved global model.
*   **Framework Agnostic**: They are designed to work seamlessly with different machine learning frameworks (PyTorch, TensorFlow, Flax) and communication protocols. (EXPLAIN THAT A NUMPY ARRAY REPRESENTATION IS USED)
*   **Gossip Optimized**: Certain aggregators can perform **partial aggregations**, allowing the aggregation proccess to speed up by avoiding sending n*(n-1) models, being n the number of nodes.


---

## Usage

You can instantiate and pass the aggregator for a node by using the `aggregator` parameter, like:

```python
node = Node(
    ...
    aggregator = FedAvg()
)
```

## Available Aggregators

Currently, the library has support for the following aggregators:

Aggregator       | Description                                                                                   | Parameters Required      | Paper Link
-------------------|-----------------------------------------------------------------------------------------------|--------------------------|-------------------------------------------------------------------------------------------------
FedAvg            | Federated Averaging combines updates using a weighted average based on sample size.           | None                     | Communication-Efficient Learning of Deep Networks from Decentralized Data: https://arxiv.org/abs/1602.05629
FedMedian         | Computes the median of updates for robustness against outliers or adversarial contributions.  | None                     | Robust Aggregation for Federated Learning: https://arxiv.org/abs/1705.05491
SCAFFOLD          | Uses control variates to reduce variance and correct client drift in non-IID data scenarios.  |  **global_lr** : The global learning rate                     | SCAFFOLD: Stochastic Controlled Averaging for Federated Learning: https://arxiv.org/abs/1910.06378



## Partial aggregations

Partial aggregations allow for combining model updates even when not all nodes have contribute
with their updates. This is useful in scenarios like when:
- Nodes are offline or unavailable
- Communication delays prevent for receiving updates from all nodes.
- Partial results are needed to avoid delaying the training process.

When using partial aggregation, the aggregator combines only the models received so far, ensuring training continuation. For example, the `FedAvg` aggregator supports partial aggregation by weighting model updates from available nodes, minimizing the impact of missing contributors. Not all aggregators support partial aggregations, which is controlled by a parameter `partial_aggregation` on each aggregator.

## Aggregator specific requirements

Some aggregators require additional actions from the model during training, such as capturing gradients or calculating specific variables. To handle this, aggregators use callbacks that the model must implement.

Aggregators define their required callbacks through the `get_required_callbacks()` method, and the learner uses these callbacks to gather the necessary information during training. This information is stored in a shared dictionary, `additional_information`, which ensures that all nodes have the data needed for proper aggregation.


