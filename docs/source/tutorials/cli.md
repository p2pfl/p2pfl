# 💻 Using the CLI

P2PFL provides a command-line interface (**CLI**) to simplify running experiments. You can launch the CLI using `python -m p2pfl`. This interface provides a convenient way to explore and run different federated learning experiments without manually writing code for each node. You can easily switch to different communication protocols, aggregators, and ML frameworks.

## ⌨️ Main Commands

| Command        | Description                                                                    | Availability |
|----------------|--------------------------------------------------------------------------------|--------------|
| `experiment`   | Run experiments on the p2pfl platform.                                         | ✅            |
| `launch`       | Launch a new node in the p2pfl network.                                        | 🔜 Coming Soon |
| `login`        | Authenticate with the p2pfl platform using your API token.                     | 🔜 Coming Soon |
| `remote`       | Interact with a remote node in the p2pfl network.                              | 🔜 Coming Soon |


## 🧪 Example: `experiment` Command

This command allows you to interact with pre-built examples. Here's how you can use it:

* **List available examples:** `python -m p2pfl experiment list`. This will display a table of available examples with their descriptions.
* **Run an example:** `python -m p2pfl experiment run <example_name> [options]`.This will run the specified example.

For instance, to run the `mnist` example for 2 rounds of training with 1 epoch each:

```bash
python -m p2pfl experiment run mnist --rounds 2 --epochs 1
```

This command will start the `mnist` experiment with the specified options and the default components. To see the available options for the `mnist` experiment, you can run:

```bash
python -m p2pfl experiment help mnist
```