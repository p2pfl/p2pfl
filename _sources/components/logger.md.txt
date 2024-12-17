# 📝 Logger

The `P2PFLogger` class is an integral part of the P2PFL framework, providing comprehensive logging functionality for monitoring and debugging federated learning experiments across distributed nodes. The logger captures a wide range of events, from system metrics to node-specific actions, and allows easy management of log levels and formats. Below is an explanation of its core functionalities, with details on decorators that enhance logging operations.

## Logger Initialization

The logger is initialized through the `P2PFLogger` class automatically when you import the module. The logger is designed to handle:

- **Node Information**: A dictionary of node details.
- **Experiment Metrics**: Metrics related to both local and global training progress.
- **Python Logging**: Utilizes Python’s built-in logging module to manage log messages efficiently.

Upon initialization, the logger sets up the default log level and output format. A colored formatter is used to display log messages with distinct colors for clarity.

## Log Level Management

You can configure the logging level dynamically using `set_level(level)` to adjust the verbosity of logs. The available levels are the same as standard python logging (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). The current log level can be checked using the `get_level()` method.

Example usage:
```python
# Set log level to INFO
logger.set_level("INFO")
# Get current log level
current_level = logger.get_level()
```

## Log Message Types

The logger supports various log levels (e.g., INFO, DEBUG, WARNING, ERROR, and CRITICAL), each of which can be used to log messages related to specific events:

- **Info Messages**: General information about node operations or experiment states.
- **Debug Messages**: Detailed debugging information, useful during development or troubleshooting.
- **Warning Messages**: Indications of potential issues that do not require immediate attention but should be noted.
- **Error Messages**: Critical issues that might affect the progress of experiments or the system.
- **Critical Messages**: Severe errors that require immediate attention.

For example:
```python
logger.info(node_addr, "Node initialized successfully.")
logger.debug(node_addr, "Debugging node configuration.")
logger.error(node_addr, "Node failed to connect to the server.")
```

## Node Registration and Status

The logger supports registering and unregistering nodes within the framework. Each node can be associated with an experiment or round, and the logger keeps track of the experiment's start and end, as well as round transitions.

- **Registering a Node**: A new node can be added using the `register_node()` method.
- **Unregistering a Node**: A node can be removed using the `unregister_node()` method.
- **Tracking Node Status**: The logger allows tracking the start and end of experiments or rounds using methods like `experiment_started()`, `experiment_finished()`, `round_started()`, and `round_finished()`.

Example:
```python
logger.register_node(node, simulation=True)
logger.experiment_started(node, experiment)
```

## Metrics Logging

> Important! The experiment related metrics need to register the node before logging. A training metric cannot be logged if a experiment is not started.

The logger can also record experiment metrics such as loss, accuracy, or system performance metrics. Metrics are logged either locally for specific nodes or globally across all nodes involved in the federated learning experiment.

- **Local Metrics**: Metrics specific to an individual node.
- **Global Metrics**: Metrics that aggregate information from all nodes.
Metrics are logged using the log_metric() function, which requires the node’s address, metric name, value, and optionally, the round and step of the experiment.

Example:
```python
logger.log_metric(node_addr, "accuracy", 0.92, round=5, step=100)
```

## Logging Decorators

To enhance the functionality of the logger, decorators are used for specific tasks. These decorators wrap another `P2PFLogger` class and add additional functionality:

- **File Logging (FileLogger class)**: A decorator that wraps a logger class, automatically writing log entries to a specified file. This decorator enhances the base logger by directing the log output to a file in addition to the standard output.
- **Web Service Logging (WebP2PFLogger class)**: A decorator that wraps a logger class and sends log entries to P2PFL web services. This is useful for centralized logging when working with distributed systems, allowing logs to be monitored remotely.
- **Ray Cluster Logging (RayP2PFLogger class)**: A decorator that wraps a logger class and sends logs to a Ray cluster. This is particularly useful in federated learning setups using Ray, where logs need to be captured at both the node and system level in a distributed environment.
- **Async Logging (AsyncLogger class)**: A decorator that wraps the logger to support asynchronous logging. This ensures that logging does not block the main processes in high-performance, distributed environments, improving the overall efficiency of the system.

> Do not use `AsyncLogger` combined with `RayP2PFLogger`.

## Cleanup and Finalization
When the logging is no longer needed (e.g., when an experiment ends), the cleanup() method should be called to remove all handlers and unregister nodes. This ensures that all resources are properly released and that no unnecessary handlers remain active.

```python
logger.cleanup()
```
