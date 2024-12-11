# ⌨️ Commands

Commands in P2PFL orchestrate actions and data exchange within the decentralized network. Leveraging the **Command Pattern**, P2PFL decouples command senders and receivers, enabling flexible communication and easy extension with new commands.  The `CommunicationProtocol` receives and routes incoming commands to their respective handlers.

## Command Table

| Type         | Command Class      | Description                                                                    |
|--------------|-------------------|--------------------------------------------------------------------------------|
| Message      | [`StartLearningCommand`](#StartLearningCommand) | Initiates federated learning across the network.                               |
|              | [`StopLearningCommand`](#StopLearningCommand)  | Terminates the federated learning process.                                   |
|              | [`ModelInitializedCommand`](#ModelInitializedCommand) | Signals model initialization on a node.                                     |
|              | [`VoteTrainSetCommand`](#VoteTrainSetCommand) | Orchestrates voting for training set selection.                               |
|              | [`ModelsAggregatedCommand`](#ModelsAggregatedCommand) | Informs neighbors of completed model aggregations.                           |
|              | [`ModelsReadyCommand`](#ModelsReadyCommand) | Signals aggregation completion and readiness for the next stage.             |
|              | [`MetricsCommand`](#MetricsCommand) | Shares evaluation metrics.                                                    |
|              | [`HeartbeatCommand`](#HeartbeatCommand) | Confirms node liveness and detects failures.                                  |
| Weights      | [`InitModelCommand`](#InitModelCommand) | Distributes initial model weights.                                            |
|              | [`PartialModelCommand`](#PartialModelCommand) | Sends a partial model update (used during aggregation).                       |
|              | [`FullModelCommand`](#FullModelCommand) | Sends a complete, aggregated model.                                           |

## Sending Commands

Nodes send commands through their `CommunicationProtocol` instance.  The protocol provides methods for constructing and sending messages containing commands.

For **message commands**, use `build_msg()` to create a message, providing the command name and any required arguments as strings:

```python
# Example: Sending the StartLearningCommand
communication_protocol.build_msg(StartLearningCommand.get_name(), [str(rounds), str(epochs)])

# Example: Sending the MetricsCommand
metrics = {"accuracy": 0.95, "loss": 0.05}
flattened_metrics = [str(item) for pair in metrics.items() for item in pair]
communication_protocol.build_msg(MetricsCommand.get_name(), flattened_metrics)
```

For **weights commands**, use `build_weights()` to create a message containing the serialized model weights, contributors, and the number of samples used in training:

```python
# Example: Sending the PartialModelCommand
serialized_model = model.encode_parameters() # Encode the model parameters into bytes
communication_protocol.build_weights(PartialModelCommand.get_name(), round_number, serialized_model, contributors, num_samples)
```

The `send()` and `broadcast()` methods of the `CommunicationProtocol` are then used to transmit the constructed messages to specific neighbors or the entire network, respectively.  For example, to broadcast a message:

```python
message = self._communication_protocol.build_msg(...) # Or build_weights(...)
self._communication_protocol.broadcast(message)
```

## Receiving and Executing Commands

The `CommunicationProtocol` manages incoming commands.  Crucially, you register command handlers with the protocol using `add_command()`:

```python
# Example: Registering commands (typically done during Node initialization)
commands = [
    StartLearningCommand(...),
    MetricsCommand(...),
    # ... other commands
]
communication_protocol.add_command(commands)  # or add_command(single_command)
```

This creates an internal registry within the `CommunicationProtocol`. When a message arrives, the protocol extracts the command name, retrieves the corresponding `Command` instance from the registry, and executes it using:

```python
command_instance.execute(source_node, round_number, **arguments)
```

The `source_node` and `round_number` provide context, while `arguments` contain any command-specific data.
