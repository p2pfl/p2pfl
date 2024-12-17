# 🚦 Node State

`NodeState` centralizes critical information, enabling different components within a node to coordinate effectively during federated learning.  Thread locks are used to maintain data consistency during concurrent operations like voting and aggregation.

## Key Attributes

* **`addr` (str):** The node's address.
* **`status` (str):**  The node's current status (e.g., "Idle", "Learning").
* **`experiment` ([Experiment](#experiment)):**  The current experiment's configuration.
* **`simulation` (bool):**  Whether the node is running in a simulation.
* **`models_aggregated` (Dict):**  Aggregated models received from neighbors.
* **`nei_status` (Dict):**  Neighbors' current round numbers.
* **`train_set` (List):** Nodes participating in the current training round.
* **`train_set_votes` (Dict):** Votes for train set selection.
* **Thread Locks (various):**  Ensure thread-safe access to shared state.

## Experiment

The `Experiment` class encapsulates the configuration and state of a federated learning experiment.  It tracks the experiment's name, the total number of rounds, and the current round.

```python
from p2pfl.experiment import Experiment

experiment = Experiment(exp_name="my_experiment", total_rounds=10)
print(f"Experiment name: {experiment.exp_name}")
print(f"Total rounds: {experiment.total_rounds}")
print(f"Current round: {experiment.round}")

experiment.increase_round()
print(f"Current round after increase: {experiment.round}")
```