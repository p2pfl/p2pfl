# 🗂️ Datasets

The [`P2PFLDataset`](#P2PFLDataset) class is a key component of the P2PFL framework, providing a flexible and efficient way to manage datasets for federated learning experiments. It simplifies dataset loading, partitioning (both IID and non-IID), and exporting to various machine learning frameworks. It supports a wide range of data sources, making it easy to integrate your existing data into P2PFL.

## Key Features

*   **Unified Data Handling:** `P2PFLDataset` provides a consistent API for working with datasets, regardless of their original format or source.
*   **Flexible Data Loading:** Load data from various sources, including CSV, JSON, and Parquet files, Pandas DataFrames, Python dictionaries and lists, and the Hugging Face Hub.
*   **Automated Partitioning:** Easily partition your data for federated learning using built-in strategies like [`RandomIIDPartitionStrategy`](#RandomIIDPartitionStrategy) or [`DirichletPartitionStrategy`](#DirichletPartitionStrategy).
*   **Framework-Specific Export:** Export your data in formats readily usable by popular machine learning frameworks like PyTorch, TensorFlow, and Flax.
*   **Data Transformations:** Apply custom transformations to your data before training.

## Loading Data

The `P2PFLDataset` simplifies data loading by offering convenient methods for common data sources. You can directly instantiate a `P2PFLDataset` object from various formats, such as CSV, JSON, or Parquet files, as well as from Pandas DataFrames, Python dictionaries, or lists. For instance, to load data from a CSV file, you can use the `from_csv()` method, providing the file path as an argument. Similarly, `from_json()` and `from_parquet()` methods are available for JSON and Parquet files, respectively.

For those working with datasets hosted on the Hugging Face Hub, the `from_huggingface()` method provides a direct way to load datasets by specifying the dataset name. Additionally, if you have data already loaded in a Pandas DataFrame, you can use the `from_pandas()` method.

Here are some examples of how to use these methods:

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

    .. tab-item:: Hugging Face Hub

        .. code-block:: python

            from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

            # From Hugging Face Hub (MNIST Example)
            p2pfl_dataset = P2PFLDataset.from_huggingface("p2pfl/MNIST", split="train")

    .. tab-item:: Generator

        .. code-block:: python

            from datasets import Dataset
            from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
            import numpy as np

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
```

### Advanced Data Loading

For more complex scenarios or when dealing with data sources not directly supported by the built-in methods, you can leverage the flexibility of the Hugging Face `datasets` library. This allows you to create a `datasets.Dataset` or `datasets.DatasetDict` instance using custom loading scripts or by applying intricate data manipulation techniques before integrating it with P2PFL.

#### Example: Creating a Dataset from Python Lists/Dictionaries

You can easily create a `P2PFLDataset` from Python lists or dictionaries using the `datasets` library:

```python
from datasets import Dataset
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
import numpy as np

# Example data (MNIST-like)
data = [{"image": np.random.rand(28, 28), "label": i % 10} for i in range(10)]

# Create a Hugging Face Dataset
dataset = Dataset.from_list(data)

# Create a P2PFLDataset
p2pfl_dataset = P2PFLDataset(dataset)
```

#### Example: Creating a DatasetDict with Custom Splits

You can also create a `datasets.DatasetDict` with custom splits, allowing you to manage different portions of your data separately:

```python
from datasets import Dataset, DatasetDict
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
import numpy as np

# Create dummy data for train and test splits
train_data = {"image": [np.random.rand(28, 28) for _ in range(50)], "label": [i % 10 for i in range(50)]}
test_data = {"image": [np.random.rand(28, 28) for _ in range(20)], "label": [i % 10 for i in range(20)]}

# Create Dataset objects for each split
train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

# Create a DatasetDict
dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

# Create a P2PFLDataset object
p2pfl_dataset = P2PFLDataset(dataset_dict)
```

These examples demonstrate how you can utilize the `datasets` library's capabilities to load and preprocess your data in a customized manner before using it with `P2PFLDataset`. This approach provides maximum flexibility and control over your data handling pipeline, enabling you to tailor it to the specific needs of your federated learning experiments.

### Train/Test Split

If the dataset does not come with predefined splits, you can create them using `generate_train_test_split()`. Here's an example:

```python
# Generate a train/test split with an 80/20 ratio
p2pfl_dataset.generate_train_test_split(test_size=0.2, seed=42)

# Access the number of samples in each split
num_train_samples = p2pfl_dataset.get_num_samples(train=True)
num_test_samples = p2pfl_dataset.get_num_samples(train=False)

print(f"Number of training samples: {num_train_samples}")
print(f"Number of test samples: {num_test_samples}")
```

## Data Access and Manipulation

The `P2PFLDataset` class provides convenient methods for accessing and manipulating data.

**Accessing Data:** You can retrieve individual samples using the `get()` method. Specify the index of the desired sample and the split (train or test) you want to access:

```python
# Access the 5th item from the training split
item = p2pfl_dataset.get(4, train=True)
print(item)
```

### Data Transforms

P2PFL supports data transformations through a flexible system that leverages the Hugging Face datasets library. Transforms are applied during the dataset processing pipeline before the data is exported to specific ML frameworks.

#### Custom Transform Functions

In P2PFL, transforms are defined as functions that operate on batches of examples. These functions should:

1. Accept a dictionary of examples (where each key is a feature name and each value is a list of feature values)
2. Apply the desired transformations
3. Return the transformed dictionary with the same structure

Here's an example of a transform function for CIFAR10 images:

```python
# p2pfl/examples/cifar10/transforms.py
import torch
import torchvision.transforms as transforms

def cifar10_transforms(examples):
    """Apply normalization to a batch of CIFAR10 examples."""
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    # Transform all images using list comprehension
    transformed_images = [normalize(img if isinstance(img, torch.Tensor) else to_tensor(img)) 
                         for img in examples["image"]]

    return {"image": transformed_images, "label": examples["label"]}
```

#### Programmatically in Python

You can apply transforms programmatically using the `set_transforms()` method:

```python
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.examples.cifar10.transforms import cifar10_transforms

# Load CIFAR10 dataset
cifar10_dataset = P2PFLDataset.from_huggingface("p2pfl/CIFAR10")

# Set the transform function
cifar10_dataset.set_transforms(cifar10_transforms)

# The transforms will be applied when accessing data
transformed_sample = cifar10_dataset.get(0, train=True)
# Or when exporting to a framework-specific format
```

#### In YAML Configuration

Transforms can be configured in YAML files for experiment configuration. This is useful for reproducible experiments:

```yaml
# Dataset settings
dataset:
  source: "huggingface" 
  name: "p2pfl/CIFAR10"
  batch_size: 64
  # Transform configuration
  transforms:
    package: "p2pfl.examples.cifar10.transforms"  # Python package containing the transform
    function: "cifar10_transforms"                # Function name to use
    params: {}                                    # Optional parameters to pass to the function
```
This configuration allows transforms to be easily shared, reused, and versioned as part of your experiment setup.

## Partitioning for Federated Learning

Partitioning data is a crucial step in federated learning, replicating the scenario where data is spread across multiple devices or nodes. The method used for partitioning has a significant impact on the performance, convergence rate, and overall effectiveness of federated learning models. This process is not merely about dividing data; it involves strategically constructing a realistic simulation of a decentralized data environment.

P2PFL offers the capability to investigate various partitioning strategies, each tailored to emulate different real-world scenarios. The table below summarizes the available partitioning strategies within the P2PFL framework:

| Strategy                                                                                | Description                                                                                                                                                              | Use Case                                                                                                    |
| :-------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| [`RandomIIDPartitionStrategy`](#RandomIIDPartitionStrategy)                               | Distributes data randomly across clients, creating an **Independent and Identically Distributed (IID) scenario**.                                                              | Simulates homogeneous client data, useful for baseline comparisons.                                         |
| [`DirichletPartitionStrategy`](#DirichletPartitionStrategy) (Non-IID)                     | Distributes data based on a Dirichlet distribution, controlled by the `alpha` parameter, creating a **non-IID scenario**.                                                    | Simulates heterogeneous client data, reflecting real-world scenarios where clients have different distributions. |

Use `generate_partitions()` to generate a list of `P2PFLDataset` with a given [`DataPartitionStrategy`](#DataPartitionStrategy). The following example demonstrates how to a IID and Non-IID partitioning using the `RandomIIDPartitionStrategy` and `DirichletPartitionStrategy`  respectively:

```python
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy, DirichletPartitionStrategy

# Generate 3 IID partitions
partitions = p2pfl_dataset.generate_partitions(3, RandomIIDPartitionStrategy)

# Generate 3 non-IID partitions using Dirichlet distribution with alpha=0.1
partitions = p2pfl_dataset.generate_partitions(3, DirichletPartitionStrategy, alpha=0.1)
```

## Exporting Data

Once you have loaded, preprocessed, and partitioned your data, the next step is to export it into a format that your chosen machine learning framework can understand. The `P2PFLDataset` class simplifies this process by providing an `export()` method that works seamlessly with different [`DataExportStrategy`](#DataExportStrategy)'s. Each strategy is designed to handle the specific requirements of a particular framework, ensuring that your data is prepared correctly for training or evaluation.

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
