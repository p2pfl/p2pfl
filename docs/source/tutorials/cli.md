# 💻 Using the CLI

P2PFL provides a command-line interface (**CLI**) to simplify running experiments. You can launch the CLI using `python -m p2pfl`. This interface provides a convenient way to explore and run different federated learning experiments without manually writing code for each node. You can easily switch to different communication protocols, aggregators, and ML frameworks.

## ⌨️ Main Commands

| Command        | Description                                                                    | Availability |
|----------------|--------------------------------------------------------------------------------|--------------|
| `run`          | Run experiments on the p2pfl platform.                                         | ✅            |
| `launch`       | Launch a new node in the p2pfl network.                                        | 🔜 Coming Soon |
| `login`        | Authenticate with the p2pfl platform using your API token.                     | 🔜 Coming Soon |
| `remote`       | Interact with a remote node in the p2pfl network.                              | 🔜 Coming Soon |

## 🧪 Example: `experiment` Command

This command allows you to interact with pre-built examples. Here's how you can use it:

* **List available examples:** `python -m p2pfl list-examples`. This will display a table of available examples with their descriptions.
* **Run an example:** `python -m p2pfl run <example_name|yaml_file>`.This will run the specified example.

For instance, to run the `mnist` example for 2 rounds of training with 1 epoch each:

```bash
python -m p2pfl run mnist
```
