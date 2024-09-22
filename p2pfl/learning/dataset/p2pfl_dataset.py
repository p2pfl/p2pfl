#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""P2PFL dataset abstraction."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Type, Union

import pandas as pd  # type: ignore
from datasets import Dataset, DatasetDict, load_dataset  # type: ignore

from p2pfl.learning.dataset.partition_strategies import DataPartitionStrategy

# Define the DataFiles type for clarity
DataFilesType = Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]]

# METER EN TESTS TANTO LA CARGA POR SPLITS COMO LA CARGA DE UN SOLO DATASET


class DataExportStrategy(ABC):
    """Abstract base class for export strategies."""

    @staticmethod
    @abstractmethod
    def export(data: Dataset, transforms: Optional[Callable] = None, **kwargs) -> Any:
        """
        Export the data using the specific strategy.

        Args:
            data: The data to export.
            transforms: The transforms to apply to the data.
            **kwargs: Additional keyword arguments for the export strategy.

        Return:
            The exported data.

        """
        pass


class P2PFLDataset:
    """
    Handle various data sources for Peer-to-Peer Federated Learning (P2PFL).

    This class uses Hugging Face's `datasets.Dataset` as the intermediate representation for its flexibility and
    optimizations.

    Supported data sources:

    - CSV files
    - JSON files
    - Parquet files
    - Python dictionaries
    - Python lists
    - Pandas DataFrames
    - Hugging Face datasets
    - SQL databases

    In order to load different data sources, we recomend to directly instantiate the `datasets.Dataset` object and pass
    it to the `P2PFLDataset` constructor.

    For example, if you want to load data from different sources:

    ´´´python
    from datasets import load_dataset, DatasetDict, concatenate_datasets

    # Load data from a CSV file
    dataset_csv = load_dataset("csv", data_files="data.csv")

    # Load from the Hub
    dataset_hub = load_dataset("squad", split="train")

    # Create the final dataset object
    p2pfl_dataset = P2PFLDataset(
        DatasetDict({
            "train": concatenate_datasets([dataset_csv, dataset_hub]),
            "test": dataset_json
        })
    )
    ´´´

    .. todo:: Add more complex integrations (databricks, etc...)
    """

    def __init__(
        self,
        data: Union[Dataset, DatasetDict],
        train_split_name: str = "train",
        test_split_name: str = "test",
        transforms: Optional[Callable] = None,
    ):
        """
        Initialize the P2PFLDataset object.

        Args:
            data: The dataset to use.
            train_split_name: The name of the training split.
            test_split_name: The name of the test split.
            transforms: The transforms to apply to the data.

        """
        self._data = data
        self._train_split_name = train_split_name
        self._test_split_name = test_split_name
        self._transforms = transforms

    def get(self, idx, train: bool = True) -> Dict[str, Any]:
        """
        Get the item at the given index.

        Args:
            idx: The index of the item to retrieve.
            train: If True, get the item from the training split. Otherwise, get the item from the test split.

        Returns:
            The item at the given index.

        """
        if isinstance(self._data, Dataset):
            data = self._data[idx]
        elif isinstance(self._data, DatasetDict):
            split = self._train_split_name if train else self._test_split_name
            data = self._data[split][idx]
        return data

    def set_transforms(self, transforms: Callable) -> None:
        """
        Set the transforms to apply to the data.

        Args:
            transforms: The transforms to apply to the data.

        """
        self._transforms = transforms

    def generate_train_test_split(self, test_size: float = 0.2, seed: int = 42, shuffle: bool = True, **kwargs) -> None:
        """
        Generate a train/test split of the dataset.

        Args:
            test_size: The proportion of the dataset to include in the test split.
            seed: The random seed to use for reproducibility.
            shuffle: Whether to shuffle the data before splitting.
            **kwargs: Additional keyword arguments to pass to the train_test_split method.

        """
        if isinstance(self._data, Dataset):
            self._data = self._data.train_test_split()
        else:
            raise ValueError("Unsupported data type.")

    def get_num_samples(self, train: bool = True) -> int:
        """
        Get the number of samples in the dataset.

        Args:
            train: If True, get the number of samples in the training split. Otherwise, get the number of samples in the test split.

        Returns:
            The number of samples in the dataset.

        """
        if isinstance(self._data, Dataset):
            return len(self._data)
        elif isinstance(self._data, DatasetDict):
            split = self._train_split_name if train else self._test_split_name
            return len(self._data[split])
        else:
            raise TypeError("Unsupported data type.")

    def generate_partitions(
        self, num_partitions: int, strategy: DataPartitionStrategy, seed: int = 666, label_tag: str = "label"
    ) -> List["P2PFLDataset"]:
        """
        Generate partitions of the dataset.

        Args:
            num_partitions: The number of partitions to generate.
            strategy: The partition strategy to use.
            seed: The random seed to use for reproducibility.
            label_tag: The tag to use for the label.

        Returns:
            An iterable of P2PFLDataset objects.

        """
        if isinstance(self._data, Dataset):
            raise ValueError("Cannot generate partitions for single datasets. ")
        train_partition_idxs, test_partition_idxs = strategy.generate_partitions(
            self._data[self._train_split_name],
            self._data[self._test_split_name],
            num_partitions,
            seed=seed,
            label_tag=label_tag,
        )
        return [
            P2PFLDataset(
                DatasetDict(
                    {
                        self._train_split_name: self._data[self._train_split_name].select(train_partition_idxs[i]),
                        self._test_split_name: self._data[self._test_split_name].select(test_partition_idxs[i]),
                    }
                )
            )
            for i in range(num_partitions)
        ]

    def export(
        self,
        strategy: Type[DataExportStrategy],
        train: bool = True,
        **kwargs,
    ) -> Any:
        """
        Export the dataset using the given strategy.

        Args:
            strategy: The export strategy to use.
            train: If True, export the training data. Otherwise, export the test data.
            **kwargs: Additional keyword arguments for the export strategy.

        Returns:
            The exported data.

        """
        # Checks
        if isinstance(self._data, Dataset):
            raise ValueError("Cannot export single datasets. Need to generate train/test splits first.")

        # Export
        split = self._train_split_name if train else self._test_split_name
        return strategy.export(self._data[split], transforms=self._transforms, **kwargs)

    @classmethod
    def from_csv(cls, data_files: DataFilesType, **kwargs) -> "P2PFLDataset":
        """
        Create a P2PFLDataset from a CSV file.

        Args:
            data_files: The path to the CSV file or a list of paths to CSV files.
            **kwargs: Keyword arguments to pass to datasets.load_dataset.

        Return:
            A P2PFLDataset object.

        """
        dataset = load_dataset("csv", data_files=data_files, **kwargs)
        return cls(dataset)

    @classmethod
    def from_json(cls, data_files: DataFilesType, **kwargs) -> "P2PFLDataset":
        """
        Create a P2PFLDataset from a JSON file.

        Args:
            data_files: The path to the JSON file or a list of paths to JSON files.
            **kwargs: Keyword arguments to pass to datasets.load_dataset.

        Return:
            A P2PFLDataset object.

        """
        dataset = load_dataset("json", data_files=data_files, **kwargs)
        return cls(dataset)

    @classmethod
    def from_parquet(cls, data_files: DataFilesType, **kwargs) -> "P2PFLDataset":
        """
        Create a P2PFLDataset from a Parquet file or files.

        Args:
            data_files: The path to the Parquet file or a list of paths to Parquet files.
            **kwargs: Keyword arguments to pass to datasets.load_dataset.

        Return:
            A P2PFLDataset object.

        """
        dataset = load_dataset("parquet", data_files=data_files, **kwargs)
        return cls(dataset)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "P2PFLDataset":
        """
        Create a P2PFLDataset from a Pandas DataFrame.

        Args:
            df: A Pandas DataFrame containing the data.

        Returns:
            A P2PFLDataset object.

        """
        dataset = Dataset.from_pandas(df)
        return cls(dataset)

    @classmethod
    def from_huggingface(cls, dataset_name: str, **kwargs) -> "P2PFLDataset":
        """
        Create a P2PFLDataset from a Hugging Face dataset.

        Args:
            dataset_name: The name of the Hugging Face dataset.
            **kwargs: Keyword arguments to pass to datasets.load_dataset.

        Returns:
            A P2PFLDataset object.

        """
        dataset = load_dataset(dataset_name, **kwargs)
        return cls(dataset)

    @classmethod
    def from_generator(cls, generator: Callable[[], Iterable[Dict[str, Any]]]) -> "P2PFLDataset":
        """
        Create a P2PFLDataset from a generator function.

        Args:
            generator: A generator function that yields dictionaries.

        Returns:
            A P2PFLDataset object.

        """
        dataset = Dataset.from_generator(generator)
        return cls(dataset)
