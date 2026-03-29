# 🧠 Learners

Learners are core components in P2PFL, responsible for managing the local training and evaluation of machine learning models on each node. They act as an intermediary between the P2PFL framework and the specific machine learning library you're using. This abstraction allows P2PFL to be framework-agnostic, providing a consistent interface for training models regardless of the underlying library.

| Framework                               | Learner Class                           | Model Type        |
|-----------------------------------------|-----------------------------------------|-------------------|
| [PyTorch](https://pytorch.org/)         | [`LightningLearner`](#LightningLearner) | `WeightBasedModel` |
| [Keras](https://keras.io/)              | [`KerasLearner`](#KerasLearner)         | `WeightBasedModel` |
| [Flax](https://flax.readthedocs.io)     | [`FlaxLearner`](#FlaxLearner)           | `WeightBasedModel` |
| [XGBoost](https://xgboost.readthedocs.io) | [`XGBoostLearner`](#XGBoostLearner)   | `TreeBasedModel`   |


## P2PFLModel

Learners operate on [`P2PFLModel`](#P2PFLModel) instances, which offer a unified way to represent models from different frameworks. This allows learners to interact consistently with models regardless of their origin.

### Model Hierarchy

P2PFL uses a type-safe model hierarchy to distinguish between different model types and ensure compatibility with appropriate aggregators:

```
P2PFLModel (ABC)
├── WeightBasedModel (neural networks)
│   ├── LightningModel (PyTorch)
│   ├── KerasModel (TensorFlow/Keras)
│   └── FlaxModel (Flax/JAX)
└── TreeBasedModel (tree ensembles)
    └── XGBoostModel
```

| Model Type         | Raw Parameter Type                       | Description                              | Compatible Aggregators |
| :----------------- | :--------------------------------------- | :--------------------------------------- | :--------------------- |
| `WeightBasedModel` | `list[np.ndarray]` (float32/float64)     | Weight tensors from neural network layers | `WeightAggregator` (FedAvg, FedMedian, etc.), `SequentialLearning` |
| `TreeBasedModel`   | `dict[str, Any]`                         | Parsed tree structure (XGBoost JSON dict) | `TreeAggregator` (FedXgbBagging), `SequentialLearning` |

The raw types returned by `get_parameters()` differ by model type:

```python
# WeightBasedModel (e.g., LightningModel)
params = model.get_parameters()
# Returns: [np.ndarray(shape=(784, 256), dtype=float32), np.ndarray(shape=(256,), dtype=float32), ...]
# One array per layer: weights, biases, etc.

# TreeBasedModel (e.g., XGBoostModel)
params = model.get_parameters()
# Returns: dict with XGBoost JSON structure
# {
#     "learner": {
#         "gradient_booster": {
#             "model": {
#                 "trees": [...],  # List of tree structures
#                 "tree_info": [...],
#                 "gbtree_model_param": {"num_trees": "5", ...}
#             }
#         }
#     },
#     "version": [3, 1, 2]
# }
```

> **Important**: Do not inherit directly from `P2PFLModel`. Instead, use `WeightBasedModel` for neural networks or `TreeBasedModel` for tree ensembles to ensure proper aggregator compatibility.

### Parameter Serialization

A key aspect of this integration is the ability to serialize and deserialize model parameters:

```python
# Serialize model
serialized_model = model.encode_parameters()  # Returns: bytes
# Deserialize the parameters
# For WeightBasedModel: returns tuple[list[np.ndarray], dict]
# For TreeBasedModel: returns tuple[dict, dict]
params, info = received_model.decode_parameters(serialized_model)
# Or directly update the model
received_model.set_parameters(serialized_model)
```

This serialization mechanism is crucial for exchanging model updates during federated learning.

## Standardized Structure

P2PFL employs the **template pattern** to define a common structure for all learners. The [`Learner`](#Learner) abstract base class outlines the essential methods that every Learner must implement. This standardized structure ensures consistency across different framework integrations and simplifies the development of new Learners.  Training and evaluating a model is straightforward:

```python
# Initialize a Learner instance (PyTorch example)
learner = LightningLearner(
    p2pfl_model, p2pfl_dataset
)
# Train
learner.fit()
# Evaluate the trained model
results = learner.evaluate()
# Print the evaluation results (e.g., accuracy, loss)
print(results)
```


## Training Information on Aggregators

Learners manage callbacks, which are essential for aggregators that require additional information during training. Callbacks allow aggregators to interact with the training process on each node, collecting data or influencing training behavior.  This information is then used by the aggregator to combine model updates effectively.

```python
# Initialize a learner that computes the additional information required by the Scaffold aggregator
learner = LightningLearner(
    p2pfl_model, p2pfl_dataset, aggregator=Scaffold
)
# Train
model = learner.fit()
# Get the additional training information
model.get_info("scaffold")
```
## Ray for Distributed Simulation

P2PFL integrates with **Ray** for efficient distributed simulations of federated learning scenarios on a single machine. The [`VirtualNodeLearner`](#VirtualNodeLearner) class wraps Learners, enabling them to execute as Ray actors. This leverages all available resources for faster simulations.  See the [Simulations](../../tutorials/simulation.md) section for more information.
