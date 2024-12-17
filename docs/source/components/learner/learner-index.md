# 💡 Learning

P2PFL offers a flexible and customizable learning process, allowing you to adapt your federated learning experiments to your specific requirements.  The learning system is built on three core components:

* **[Learners](learners.md):**  Manage local training, evaluation, and model updates on each node.  They are framework-agnostic (supporting PyTorch, TensorFlow, Flax, etc.) and handle aggregator callbacks.

* **[Datasets](datasets.md):** The [`P2PFLDataset`](#P2PFLDataset) class simplifies dataset loading, partitioning (IID, non-IID), and exporting for federated learning. It supports diverse data sources (CSV, JSON, Parquet, Pandas, Hugging Face, SQL).

* **[Aggregators](aggregators.md):** Combine model updates from different nodes using various strategies (FedAvg, FedMedian, SCAFFOLD, etc.) to create a new global model.

## 📋 Modules

```{eval-rst}
.. toctree::
    :maxdepth: 1

    learners
    datasets
    aggregators
```

