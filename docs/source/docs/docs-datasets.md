# 🗂️ Datasets

The **P2PFLDataset** class handles datasets in the P2PFL framework, abstracting dataset loading, manipulation, partitioning, and exporting. It supports various data sources and operations necessary for federated learning.

## Key Components of the `P2PFLDataset` Class

### Purpose
The `P2PFLDataset` class facilitates the management of datasets, enabling users to load, manipulate, and partition datasets. It uses Hugging Face's `datasets.Dataset` as an intermediate representation.

### Supported Data Sources
The `P2PFLDataset` class can load data from several sources:
- CSV files
- JSON files
- Parquet files
- Python dictionaries and lists
- Pandas DataFrames
- Hugging Face datasets
- SQL databases

To load data, it is recommended to use the `datasets.Dataset` object from the Hugging Face `datasets` module and pass it to the `P2PFLDataset` constructor.

### Example of Data Loading
```python
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

# Create a P2PFL dataset from SQuAD
p2pfl_dataset = P2PFLDataset.from_huggingface("squad")
```

### Data Access and Manipulation
Once a P2PFLDataset object is created, users can access individual data samples, apply transformations, and generate train/test splits. The dataset is structured into "train" and "test" splits, which can be specified during initialization or modified later.

Example of accessing data and applying transformations:
```python
# Access the 5th item from the training split
item = p2pfl_dataset.get(4, train=True)

# Define a transformation function to lowercase the context and question
def transform_squad_data(x):
    """
    This function converts both context and question to lowercase
    for text normalization purposes.
    """
    return {
        'context': x['context'].lower(),
        'question': x['question'].lower(),
        'answers': x['answers']  # No transformation to answers
    }

# Set the transformation for the dataset
p2pfl_dataset.set_transforms(transform_squad_data)

# Get the first item
transformed_item = p2pfl_dataset.get(4, train=True)

```

You can also generate a split between training and testing datasets:
```python
# Generate a train/test split with a 80/20 ratio
p2pfl_dataset.generate_train_test_split(test_size=0.2, seed=42)
```

Additionally, the dataset provides a method to retrieve the number of samples in either the train or test split:
```python
# Access the number of training samples
num_train_samples = p2pfl_dataset.get_num_samples(train=True)

# Access the number of test samples
num_test_samples = p2pfl_dataset.get_num_samples(train=False)
```

### Data Partitioning for Federated Learning
In federated learning, data needs to be partitioned into smaller subsets to simulate the distribution of data across multiple clients. The `P2PFLDataset` class supports partitioning the dataset into several parts based on different partitioning strategies.

Here’s how you can partition a dataset into multiple subsets:
```python
# Define a partitioning strategy (use a custom or default strategy)
num_partitions = 5
partition_strategy = MyCustomPartitionStrategy()  # Custom strategy for partitioning

# Generate partitions for federated learning
partitions = p2pfl_dataset.generate_partitions(num_partitions, strategy=partition_strategy)

# Access the first partition's training and test data
partition_1_train = partitions[0].get(0, train=True)
partition_1_test = partitions[0].get(0, train=False)
```


### Data Export
Once data is prepared, you may need to export it using a custom export strategy. The P2PFLDataset class provides an easy interface for exporting data in different formats, depending on your requirements.

Example of exporting data:
```python
# Define the export strategy (e.g., CSV export strategy)
export_strategy = MyCustomExportStrategy()

# Export the training split data using the custom strategy
exported_train_data = p2pfl_dataset.export(strategy=export_strategy, train=True)
```

### Full Code Example for SQuAD Dataset Transformation
```python
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

# Step 1: Load the SQuAD dataset from Hugging Face
p2pfl_dataset = P2PFLDataset.from_huggingface("squad")

# Access the 5th item from the training split
item = p2pfl_dataset.get(4, train=True)

# Step 2: Define a transformation function to lowercase the context and question
def transform_squad_data(x):
    """
    This function converts both context and question to lowercase
    for text normalization purposes.
    """
    return {
        'context': x['context'].lower(),  # Convert context to lowercase
        'question': x['question'].lower(),  # Convert question to lowercase
        'answers': x['answers']  # Leave answers unchanged
    }

# Step 3: Set the transformation for the dataset
p2pfl_dataset.set_transforms(transform_squad_data)

# Step 4: Apply the transformation to the first item in the dataset (training split)
transformed_item = p2pfl_dataset.get(4, train=True)

# Generate a train/test split with a 80/20 ratio
p2pfl_dataset.generate_train_test_split(test_size=0.2, seed=42)

# Access the number of training samples
num_train_samples = p2pfl_dataset.get_num_samples(train=True)

# Access the number of test samples
num_test_samples = p2pfl_dataset.get_num_samples(train=False)

```

🌟 Ready? **You can view next**: > [Simulations](docs-simulation.md)

<div style="position: fixed; bottom: 10px; right: 10px; font-size: 0.9em; padding: 10px; border: 1px solid #ccc; border-radius: 5px;"> 🌟 You Can View Next: <a href="docs-simulation.md">Simulations</a> </div>