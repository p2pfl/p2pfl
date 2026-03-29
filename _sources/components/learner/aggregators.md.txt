# 🧩 Aggregators

Aggregators are responsible for combining model updates from multiple nodes during the federated learning process. They play a crucial role in ensuring the model converges effectively and efficiently, orchestrating how models collaborate in a decentralized manner.

## Key Features

*   **Framework Agnostic**: Aggregators work seamlessly with neural networks (PyTorch, TensorFlow, Flax) and tree ensembles (XGBoost), using a common representation for model updates.
*   **Extensible**: Easy to implement new aggregators by extending the `Aggregator` class and implementing the `aggregate` method.
*   **Gossip Optimized**: Certain aggregators can perform **partial aggregations**, allowing the aggregation proccess to speed up by avoiding sending `n*(n-1)` models, being n the number of nodes.

## Available Aggregators

Currently, the library has support for the following aggregators:

### Weight Aggregators (Neural Networks)

| Aggregator       | Description                                                                                   | Supports Partial Aggregation (Class Default) | Paper Link                                                                                                |
| :---------------- | :-------------------------------------------------------------------------------------------- | :------------------------------------------: | :-------------------------------------------------------------------------------------------------------- |
| [`FedAvg`](#FedAvg)            | Federated Averaging combines updates using a weighted average based on sample size.           |                       ✅                      | [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) |
| [`FedMedian`](#FedMedian)         | Computes the median of updates for robustness against outliers or adversarial contributions. |                       ❌                      | [Robust Aggregation for Federated Learning](https://arxiv.org/abs/1705.05491)                               |
| [`Scaffold`](#Scaffold)          | Uses control variates to reduce variance and correct client drift in non-IID data scenarios. |                       ❌                      | [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/abs/1910.06378)        |
| [`Krum`](#Krum)            | Byzantine-resilient aggregation that selects updates closest to the majority.                 |                       ❌                      | [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://arxiv.org/abs/1703.02757) |
| [`FedProx`](#FedProx)          | Adds proximal term to handle heterogeneous data and partial work.                             |                       ✅                      | [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)                        |
| [`FedAdam`](#FedAdam)          | Applies Adam optimizer on the server for adaptive learning rates.                             |                       ❌                      | [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295)                                         |
| [`FedAdagrad`](#FedAdagrad)      | Applies Adagrad optimizer on the server for adaptive learning rates.                          |                       ❌                      | [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295)                                         |
| [`FedYogi`](#FedYogi)          | Applies Yogi optimizer on the server for adaptive learning rates.                             |                       ❌                      | [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295)                                         |

### Tree Aggregators (XGBoost)

| Aggregator       | Description                                                                                   | Paper Link                                                                                                |
| :---------------- | :-------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| [`FedXgbBagging`](#FedXgbBagging)    | Combines trees from different clients into a single ensemble via bagging.                     | [Federated XGBoost (Flower)](https://flower.ai/blog/2023-11-29-federated-xgboost-with-bagging-aggregation/) |

### Generic Aggregators

| Aggregator       | Description                                                                                   | Paper Link                                                                                                |
| :---------------- | :-------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| [`SequentialLearning`](#SequentialLearning) | Passes a single model through unchanged for sequential/cyclic training topologies.           | -                                                                                                         |

*An instance's partial aggregation can be turned off using the `disable_partial_aggregation=True` parameter in its constructor if the class supports it.*

> **Callback-Coupled Aggregators**: Some aggregators require a corresponding callback to be registered during training:
> - **FedProx**: Requires ``fedprox`` callback (e.g., ``FedProxCallback`` for PyTorch) to apply the proximal term during local training.
> - **Scaffold**: Requires ``scaffold`` callback (e.g., ``ScaffoldCallback`` for PyTorch) to compute control variates.

## How to Use Aggregators

Aggregators combine models in two primary ways:

1. **Direct Aggregation:** Use the `.aggregate(model)` method to immediately combine a list of `P2PFLModel` instances. This is a straightforward approach for one-time aggregation.

2. **Stateful Aggregation:** This approach is useful within a node's [workflow](../workflows.md) and in multithreaded scenarios. The aggregator acts as a stateful object, accumulating models over time:
    *   **Setup:** Begin a round by calling `.set_nodes_to_aggregate(nodes)` to specify the participating nodes.
    *   **Incremental Addition:** Add models (e.g., `P2PFLModel` instances) as they become available using `.add_model(model)`. This is often handled automatically when model update messages are received.
    *   **Aggregation and Retrieval:** Call `.wait_and_get_aggregation()`. This method blocks until all models from the specified nodes are added, then performs the aggregation and returns the combined model.

    ```python
    # Aggregator node sets up the round:
    aggregator.set_nodes_to_aggregate([node1_id, node2_id])

    # ... (A new local model, p2pflmodel1, is received) ...
    aggregator.add_model(p2pflmodel1)

    # Aggregator node waits and gets the aggregated model:
    aggregated_model = aggregator.wait_and_get_aggregation()
    ```

## Partial Aggregations

> **Note**: We will discuss partial aggregation in more detail in the future.

Partial aggregations offer a way to combine model updates even when not all nodes have contributed. This is particularly useful when combined with the **gossip protocol**, as it significantly reduces the number of models that need to be transmitted between nodes, improving overall efficiency.

In a gossip-based system, a node might receive model updates from multiple other nodes. Instead of forwarding each of these updates individually, the aggregator can combine them into a single, **partially aggregated model**. This approach minimizes communication overhead, as fewer models need to be sent. This also makes the aggregation process more robust, allowing it to proceed even if some nodes are slow or unavailable.

While beneficial, not all aggregation algorithms are suitable for partial aggregation. For instance, while it's mathematically equivalent to full aggregation for algorithms like [FedAvg](#FedAvg), it might negatively impact the model's convergence for others.

## Model Compatibility

P2PFL uses a type-safe hierarchy to ensure aggregators are only used with compatible model types. This prevents runtime errors from incompatible combinations (e.g., trying to average XGBoost trees).

### Aggregator Hierarchy

```
Aggregator (ABC)
├── WeightAggregator (for neural networks)
│   ├── FedAvg, FedMedian, FedProx, Krum, Scaffold
│   └── FedOptBase (FedAdam, FedAdagrad, FedYogi)
├── TreeAggregator (for tree ensembles)
│   └── FedXgbBagging
└── SequentialLearning (for any model type)
```

### Compatibility Rules

| Aggregator Type        | Compatible Model Types                          | Incompatible Model Types |
| :--------------------- | :---------------------------------------------- | :----------------------- |
| `WeightAggregator`     | `WeightBasedModel` (`LightningModel`, `KerasModel`, `FlaxModel`)     | `TreeBasedModel`           |
| `TreeAggregator`       | `TreeBasedModel` (`XGBoostModel`)                                  | `WeightBasedModel`    |
| `SequentialLearning`   | All model types                                 | None                     |

When an incompatible model is passed to an aggregator, an `IncompatibleModelError` is raised:

```python
from p2pfl.learning.aggregators import FedAvg, FedXgbBagging

# This will raise IncompatibleModelError
weight_aggregator = FedAvg()
weight_aggregator.aggregate([xgboost_model])  # Error: Expected WeightBasedModel

# Correct usage
tree_aggregator = FedXgbBagging()
tree_aggregator.aggregate([xgboost_model])  # Works correctly
```

### Automatic Validation

Validation is automatic via the template pattern: when you call `aggregate()`, it automatically validates models before calling your `_aggregate()` implementation. If validation fails, an `IncompatibleModelError` is raised:

```python
from p2pfl.learning.aggregators.aggregator import IncompatibleModelError

try:
    aggregator.aggregate(models)  # Validation happens automatically
except IncompatibleModelError as e:
    print(f"Model type mismatch: {e}")
```

## Creating New Aggregators

When creating new aggregators, inherit from the appropriate base class based on the model type you want to support:

### Choose the Right Base Class

*   **`WeightAggregator`**: For aggregators that work with neural networks (PyTorch, TensorFlow, Flax). Use this when your algorithm averages or combines weight tensors.
*   **`TreeAggregator`**: For aggregators that work with tree ensembles (XGBoost). Use this when your algorithm combines or selects trees.

> **Important**: Do not inherit directly from `Aggregator`. Always use `WeightAggregator` or `TreeAggregator` to ensure proper model compatibility checks.

### Implementation Steps

1.  **Choose Base Class and Define Partial Aggregation**:
    *   Define a class attribute `SUPPORTS_PARTIAL_AGGREGATION: bool` in your new aggregator class. Set it to `True` if your algorithm can correctly handle partial aggregations, `False` otherwise.
    *   The base class's `__init__` method will automatically use this class attribute to set the initial state of `self.partial_aggregation`. It also handles the `disable_partial_aggregation` parameter.

    ```python
    from p2pfl.learning.aggregators.aggregator import WeightAggregator

    class MyCustomAggregator(WeightAggregator):
        SUPPORTS_PARTIAL_AGGREGATION: bool = True  # Or False, depending on the algorithm

        def __init__(self, disable_partial_aggregation: bool = False):
            super().__init__(disable_partial_aggregation=disable_partial_aggregation)

        def _aggregate(self, models):
            # Validation is automatic - aggregate() calls validate_models() first
            # Your aggregation logic here
            ...
    ```

    For tree-based aggregators:

    ```python
    from p2pfl.learning.aggregators.aggregator import TreeAggregator

    class MyTreeAggregator(TreeAggregator):
        SUPPORTS_PARTIAL_AGGREGATION: bool = True

        def __init__(self, disable_partial_aggregation: bool = False):
            super().__init__(disable_partial_aggregation=disable_partial_aggregation)

        def _aggregate(self, models):
            # Validation is automatic - aggregate() calls validate_models() first
            # Your tree combination logic here
            ...
    ```

2.  **Interaction with Optimization Process (Callbacks)**:
    *   Some aggregators might need more than just model weights, requiring specific information from the training process (e.g., gradients). P2PFL uses a callback system for this.
    *   Your aggregator can specify required callbacks via `get_required_callbacks()`. The `CallbackFactory` ensures these are available for the chosen ML framework.

The information gathered by these callbacks is stored in a special dictionary called `additional_info` that's associated with each model. This is where P2PFL's role becomes crucial. It manages this `additional_info` to ensure that the data collected by the callbacks is not only stored but also persists even after the models are aggregated. This means that your aggregator can reliably access this extra information using `model.get_info(callback_name)` when it's combining the models. This mechanism allows for a clean separation of concerns: the framework handles the training loop execution, the callbacks gather the necessary data, and P2PFL ensures that this data is available to the aggregator.

For instance, let's say you're building an aggregator that needs to know the L2 norm of the gradients during training. You'd create a callback, like the `GradientNormCallbackPT` we discussed for PyTorch, that calculates this norm and stores it in `additional_info`. Your aggregator would then specify that it requires this callback.

Here's how the code for the aggregator (`MyAggregator`) and the callback (`GradientNormCallbackPT`) would look:

```python
from p2pfl.learning.aggregators.aggregator import WeightAggregator
from p2pfl.learning.frameworks.callback import P2PFLCallback
from p2pfl.learning.frameworks.callback_factory import CallbackFactory
from p2pfl.learning.frameworks import Framework
import torch

# MyAggregator Implementation (inherits from WeightAggregator for neural networks)
class MyAggregator(WeightAggregator):
    SUPPORTS_PARTIAL_AGGREGATION: bool = True  # Example: This custom aggregator supports it

    def __init__(self, disable_partial_aggregation: bool = False):
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

    def _aggregate(self, models):
        # Validation is automatic - aggregate() calls validate_models() first
        # Your aggregation logic here, using model.get_parameters() and
        # ...
        # To access the gradient norm:
        for model in models:
            gradient_norm = model.get_info("GradientNormCallback")
        # ...
        return aggregated_model

    def get_required_callbacks(self):
        return ["GradientNormCallback"]

# GradientNormCallbackPT Implementation (PyTorch)
class GradientNormCallbackPT(P2PFLCallback):
    """Calculates the L2 norm of model gradients (PyTorch).

    Leverages PyTorch's callback system for execution. P2PFL facilitates
    storage and retrieval of the calculated norm in `additional_info`,
    ensuring it persists after aggregation.
    """

    @staticmethod
    def get_name() -> str:
        return "GradientNormCallback"

    def on_train_batch_end(self, model: torch.nn.Module, **kwargs) -> None:
        """Calculates and stores the gradient norm (implementation placeholder).

        Called by PyTorch's training loop. The calculated norm is stored in
        `additional_info` by P2PFL for later retrieval by the aggregator.
        """
        gradient_norm = 0.0  # Replace with actual gradient norm calculation
        self.set_info(gradient_norm)

# Register the callback for PyTorch
CallbackFactory.register_callback(learner=Framework.PYTORCH.value, callback=GradientNormCallbackPT)
```

During training, the callback would be executed, the gradient norm would be calculated and stored, and your `MyAggregator` could then access this value using `model.get_info("GradientNormCallback")` and use it in its aggregation logic.
