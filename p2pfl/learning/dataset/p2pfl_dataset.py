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
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import pandas as pd  # type: ignore
from datasets import Dataset, DatasetDict, load_dataset  # type: ignore

from p2pfl.learning.dataset.partition_strategies import DataPartitionStrategy

# Define the DataFiles type for clarity
DataFilesType = str | Sequence[str] | Mapping[str, str | Sequence[str]] | None

# METER EN TESTS TANTO LA CARGA POR SPLITS COMO LA CARGA DE UN SOLO DATASET


class DataExportStrategy(ABC):
    """Abstract base class for export strategies."""

    @staticmethod
    @abstractmethod
    def export(data: Dataset, batch_size: int | None = None, **kwargs) -> Any:
        """
        Export the data using the specific strategy.

        Args:
            data: The data to export. Transforms should already be applied to the dataset.
            batch_size: The batch size for the export.
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

    To load different data sources, it is recommended to directly instantiate the `datasets.Dataset` object
    and pass it to the `P2PFLDataset` constructor.

    Example:
        Load data from various sources and create a `P2PFLDataset` object:

        .. code-block:: python

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

    .. todo::
        Add more complex integrations (databricks, etc.)

    """

    def __init__(
        self,
        data: Dataset | DatasetDict,
        train_split_name: str = "train",
        test_split_name: str = "test",
        batch_size: int = 1,
        dataset_name: str | None = None,
    ):
        """
        Initialize the P2PFLDataset object.

        Args:
            data: The dataset to use.
            train_split_name: The name of the training split.
            test_split_name: The name of the test split.
            batch_size: The batch size for the dataset.
            dataset_name: The name of the dataset.

        """
        self._data = data
        self._train_split_name = train_split_name
        self._test_split_name = test_split_name
        self.batch_size = batch_size
        self.dataset_name = dataset_name

    def get(self, idx, train: bool = True) -> dict[str, Any]:
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

    def set_transforms(self, transforms: Callable | dict[str, Callable]) -> None:
        """
        Set the transforms to apply to the data, delegating to the Hugging Face dataset.

        Args:
            transforms: The transforms to apply to the data.

        """
        if isinstance(self._data, Dataset):
            self._data.set_transform(transforms)
        elif isinstance(self._data, DatasetDict) and callable(transforms):
            for split in self._data:
                self._data[split].set_transform(transforms)
        elif isinstance(self._data, DatasetDict) and isinstance(transforms, dict):
            for split in self._data:
                self._data[split].set_transform(transforms[split])
        else:
            raise ValueError("Unsupported data type.")

    def set_batch_size(self, batch_size: int) -> None:
        """
        Set the batch size for the dataset.

        Args:
            batch_size: The batch size for the dataset.

        """
        self.batch_size = batch_size

    def generate_train_test_split(self, **kwargs) -> None:
        """
        Generate a train/test split of the dataset.

        Args:
            **kwargs: Additional keyword arguments to pass to the train_test_split method.

        """
        if isinstance(self._data, Dataset):
            self._data = self._data.train_test_split(**kwargs)
        elif isinstance(self._data, DatasetDict):
            if self._train_split_name in self._data and self._test_split_name in self._data:
                raise ValueError("Train and test splits already exist. Use a different name or clear the dataset.")
            train_test_split = self._data[self._train_split_name].train_test_split(**kwargs)
            self._data = DatasetDict(
                {
                    self._train_split_name: train_test_split[self._train_split_name],
                    self._test_split_name: train_test_split[self._test_split_name],
                }
            )
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
        self, num_partitions: int, strategy: DataPartitionStrategy, seed: int = 666, label_tag: str = "label", **kwargs
    ) -> list["P2PFLDataset"]:
        """
        Generate partitions of the dataset.

        Args:
            num_partitions: The number of partitions to generate.
            strategy: The partition strategy to use.
            seed: The random seed to use for reproducibility.
            label_tag: The tag to use for the label.
            **kwargs: Additional keyword arguments for the partition strategy.

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
            **kwargs,
        )
        return [
            P2PFLDataset(
                DatasetDict(
                    {
                        self._train_split_name: self._data[self._train_split_name].select(train_partition_idxs[i]),
                        self._test_split_name: self._data[self._test_split_name].select(test_partition_idxs[i]),
                    }
                ),
                train_split_name=self._train_split_name,
                test_split_name=self._test_split_name,
                batch_size=self.batch_size,
                dataset_name=self.dataset_name,
            )
            for i in range(num_partitions)
        ]

    def export(
        self,
        strategy: type[DataExportStrategy],
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
        return strategy.export(self._data[split], batch_size=self.batch_size, **kwargs)

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
        return cls(dataset, dataset_name=dataset_name)

    @classmethod
    def from_generator(cls, generator: Callable[[], Iterable[dict[str, Any]]]) -> "P2PFLDataset":
        """
        Create a P2PFLDataset from a generator function.

        Args:
            generator: A generator function that yields dictionaries.

        Returns:
            A P2PFLDataset object.

        """
        dataset = Dataset.from_generator(generator)
        return cls(dataset)
