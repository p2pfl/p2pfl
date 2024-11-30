# 🎬 Quickstart

**P2P Federated Learning (p2pfl)** simplifies federated learning experiments. It allows you to easily configure communication protocols, aggregation algorithms, and ML frameworks.

For quick experimentation, we recommend using the **p2pfl CLI** for managing your federated learning setup.

## ⚠️ Before Starting

> **Note**: For detailed installation instructions, please refer to the [**installation guide**](installation.md). It covers everything you need to install **p2pfl**, including options for users, developers, and advanced manual installations.

## 📋 Quick Experiment
### 🧪 Example: Run MNIST with PyTorch
To run a federated learning experiment on the MNIST dataset using PyTorch, follow these steps:

1. List available examples:

```bash
python -m p2pfl experiment list
```

2. Run the MNIST example:

```bash
python -m p2pfl experiment run mnist --rounds 2 --epochs 1
```

This will run the MNIST experiment with 2 rounds and 1 epoch. The results will be plotted once the experiment finishes.

🎉 You are already doing P2P Federated Learning!
By running the experiment, you have successfully participated in a federated learning setup with multiple nodes, performing model training across distributed data. You've leveraged the power of **p2pfl** to make it simple!

💡 More Details
For full setup and customization, refer to the detailed usage in the [**manual**](manual.md). This includes running experiments manually and understanding the code behind the examples.
