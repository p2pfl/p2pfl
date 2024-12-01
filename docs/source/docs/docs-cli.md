# 💻 Using the CLI

P2PFL provides a command-line interface (**CLI**) to simplify running experiments. You can launch the CLI using `python -m p2pfl`. This interface provides a convenient way to explore and run different federated learning experiments without manually writing code for each node. You can easily switch to different communication protocols, aggregators, and ML frameworks.

### ⌨️ Main Commands

| Command        | Description                                                                    | Availability |
|----------------|--------------------------------------------------------------------------------|--------------|
| `experiment`   | Run experiments on the p2pfl platform.                                         | ✅            |
| `launch`       | Launch a new node in the p2pfl network.                                        | 🔜 Coming Soon |
| `login`        | Authenticate with the p2pfl platform using your API token.                     | 🔜 Coming Soon |
| `remote`       | Interact with a remote node in the p2pfl network.                              | 🔜 Coming Soon |


### 🧪 Example: `experiment` Command

This command allows you to interact with pre-built examples. Here's how you can use it:

* **List available examples:** `python -m p2pfl experiment list`. This will display a table of available examples with their descriptions.
* **Run an example:** `python -m p2pfl experiment run <example_name> [options]`.This will run the specified example.

For instance, to run the `mnist` example (which currently uses **FedAvg** as aggregator, **PyTorch** as framework and  **gRPC** as communication protocol) with 2 rounds of training with 1 epoch each:

```bash
python -m p2pfl experiment run mnist --rounds 2 --epochs 1
```

When the the mnist experiment finishes, the training results will be plotted on the screen.

> You can see experiment options by running the help command For example : `python -m p2pfl experiment help mnist`



🌟 Ready? **You can view next**: > [Nodes](docs-nodes.md)

<div style="position: fixed; bottom: 10px; right: 10px; font-size: 0.9em; padding: 10px; border: 1px solid #ccc; border-radius: 5px;"> 🌟 You Can View Next: <a href="docs-nodes.md">Nodes</a> </div>