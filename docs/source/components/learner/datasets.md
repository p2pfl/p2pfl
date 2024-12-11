# 🗂️ Datasets

The `P2PFLDataset` class is a key component of the P2PFL framework, providing a flexible and efficient way to manage datasets for federated learning experiments. It simplifies dataset loading, partitioning (both IID and non-IID), and exporting to various machine learning frameworks. It supports a wide range of data sources, making it easy to integrate your existing data into P2PFL.

## Key Features

* **Unified Interface:** Provides a consistent API for working with datasets from different sources.
* **Partitioning Strategies:** Supports various strategies for partitioning data across nodes, including IID and non-IID partitioning.
* **Framework-Specific Export:** Easily export data in formats suitable for PyTorch, TensorFlow, Flax, and other frameworks.
* **Transformations:** Apply custom transformations to your data before training.

## Loading Data

The `P2PFLDataset` class simplifies data loading, offering both built-in functionalities and flexibility for custom integrations.  You can load data directly using the provided class methods:

```{eval-rst}
.. tab-set::

    .. tab-item:: CSV

        .. code-block:: python

            from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

            # From CSV
            p2pfl_dataset = P2PFLDataset.from_csv("path/to/your/data.csv")

    .. tab-item:: JSON

        .. code-block:: python

            from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

            # From JSON
            p2pfl_dataset = P2PFLDataset.from_json("path/to/your/data.json")

    .. tab-item:: Parquet

        .. code-block:: python

            from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

            # From Parquet
            p2pfl_dataset = P2PFLDataset.from_parquet("path/to/your/data.parquet")

    .. tab-item:: HuggingFace Hub

        .. code-block:: python

            from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

            # From Hugging Face Hub (MNIST Example)
            p2pfl_dataset = P2PFLDataset.from_huggingface("p2pfl/MNIST", split="train")

    .. tab-item:: Generator

        .. code-block:: python

            from datasets import Dataset
            from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

            # From a generator function
            def my_generator():
                for i in range(10):
                    yield {"id": i, "image": np.random.rand(28, 28), "label": i % 10}  # Example MNIST-like data

            dataset = Dataset.from_generator(my_generator)
            p2pfl_dataset = P2PFLDataset(dataset)

    .. tab-item:: Pandas

        .. code-block:: python
        
            import pandas as pd
            from datasets import Dataset
            from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

            # From a Pandas DataFrame (Example with image and label columns)
            data = {'image': [np.random.rand(28, 28) for _ in range(10)], 'label': [i % 10 for i in range(10)]}
            df = pd.DataFrame(data)
            dataset = Dataset.from_pandas(df)
            p2pfl_dataset = P2PFLDataset(dataset)


    .. tab-item:: Dictionary/List

        .. code-block:: python

            from datasets import Dataset
            from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

            # From a Python dictionary/list (Example with image and label fields)
            data = [{"image": np.random.rand(28, 28), "label": i % 10} for i in range(10)]  # Example MNIST-like data
            dataset = Dataset.from_dict(data)  # Or Dataset.from_list
            p2pfl_dataset = P2PFLDataset(dataset)
```

Or in case...

---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---
---

 or by creating a hugging face `datasets.Dataset` instance.
### Custom Data Loading (using datasets.Dataset)

For data sources not covered by the built-in loaders, create a `datasets.Dataset` or `datasets.DatasetDict` instance directly using the Hugging Face `datasets` library's flexible loading methods, and then pass it to the `P2PFLDataset` constructor.

```python
from datasets import load_dataset, Dataset, DatasetDict
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

# Example: Loading from a custom text file
dataset = load_dataset("text", data_files="path/to/your/data.txt")
p2pfl_dataset = P2PFLDataset(dataset)

# Example: Creating a DatasetDict with custom splits
train_dataset = Dataset.from_dict({"text": ["Train example 1", "Train example 2"]})
test_dataset = Dataset.from_dict({"text": ["Test example 1", "Test example 2"]})

dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})
p2pfl_dataset = P2PFLDataset(dataset_dict)
```

## Data Access and Manipulation

### Accessing Data

Access samples using `get()`, specifying the index and split:

```python
# Access the 5th item from the training split
item = p2pfl_dataset.get(4, train=True)
print(item)
```

### Transformations

Apply transformations using `set_transforms()`.  Here's a simple example with MNIST:

```python
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from torchvision import transforms

# Load MNIST dataset
mnist_dataset = P2PFLDataset.from_huggingface("p2pfl/MNIST")

# Define a transform to convert the image to a tensor and normalize it
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Set the transform
mnist_dataset.set_transforms(transform)

# Access a transformed sample
transformed_sample = mnist_dataset.get(0, train=True)
print(transformed_sample)
```

### Train/Test Split

Create splits using `generate_train_test_split()`:

```python
# Generate a train/test split with an 80/20 ratio
p2pfl_dataset.generate_train_test_split(test_size=0.2, seed=42)

# Access the number of samples in each split
num_train_samples = p2pfl_dataset.get_num_samples(train=True)
num_test_samples = p2pfl_dataset.get_num_samples(train=False)

print(f"Number of training samples: {num_train_samples}")
print(f"Number of test samples: {num_test_samples}")
```

## Partitioning for Federated Learning

Use `generate_partitions()` with a `DataPartitionStrategy`:

```python
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy, DirichletPartitionStrategy

# Generate 3 IID partitions
partitions = p2pfl_dataset.generate_partitions(3, RandomIIDPartitionStrategy)

# Generate 3 non-IID partitions using Dirichlet distribution with alpha=0.1
partitions = p2pfl_dataset.generate_partitions(3, DirichletPartitionStrategy, alpha=0.1)
```

## Exporting Data

Use `export()` with a `DataExportStrategy`:

```{eval-rst}
.. tab-set::

    .. tab-item:: PyTorch

        .. code-block:: python

            from p2pfl.learning.frameworks.pytorch.lightning_dataset import PyTorchExportStrategy

            # Export the training data for PyTorch
            pytorch_data = p2pfl_dataset.export(PyTorchExportStrategy, train=True, batch_size=32, num_workers=4)

    .. tab-item:: TensorFlow/Keras

        .. code-block:: python

            from p2pfl.learning.frameworks.tensorflow.keras_dataset import KerasExportStrategy

            # Export the training data for TensorFlow/Keras
            tensorflow_data = p2pfl_dataset.export(KerasExportStrategy, train=True, batch_size=32)

    .. tab-item:: Flax

        .. code-block:: python

            from p2pfl.learning.frameworks.flax.flax_dataset import FlaxExportStrategy

            # Export the training data for Flax
            flax_data = p2pfl_dataset.export(FlaxExportStrategy, train=True, batch_size=32)
```
