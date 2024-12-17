# 🧠 Learners

Learners are core components in P2PFL, responsible for managing the local training and evaluation of machine learning models on each node. They act as an intermediary between the P2PFL framework and the specific machine learning library you're using. This abstraction allows P2PFL to be framework-agnostic, providing a consistent interface for training models regardless of the underlying library.

| Framework                               | Learner Class                           |
|-----------------------------------------|-----------------------------------------|
| [PyTorch](https://pytorch.org/)         | [`LightningLearner`](#LightningLearner) |
| [Keras](https://keras.io/)              | [`KerasLearner`](#KerasLearner)         |
| [Flax](https://flax.readthedocs.io)     | [`FlaxLearner`](#FlaxLearner)           |


## P2PFLModel

Learners operate on [`P2PFLModel`](#P2PFLModel) instances, which offer a unified way to represent models from different frameworks. This allows learners to interact consistently with models regardless of their origin. A key aspect of this integration is the ability to serialize and deserialize model parameters:

```python
# Serialize model
serialized_model = model.encode_parameters()
# Deserialize the parameters
params, info = received_model.decode_parameters(serialized_model)
# Or directly update the model
received_model.update_parameters(serialized_model)
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
