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

"""Data partitioning strategies for P2PFL Datasets."""

import random
from abc import abstractmethod
from typing import List, Tuple

from datasets import Dataset  # type: ignore


class DataPartitionStrategy:
    """
    Abstract class for defining data partitioning strategies in federated learning.

    This class provides a common interface for generating partitions of a dataset, which can be used
    to simulate different data distributions across clients.
    """

    @staticmethod
    @abstractmethod
    def generate_partitions(
        train_data: Dataset, test_data: Dataset, num_partitions: int, **kwargs
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Generate partitions of the dataset based on the specific strategy.

        Args:
            train_data: The training Dataset object to partition.
            test_data: The test Dataset object to partition.
            num_partitions: The number of partitions to create.
            **kwargs: Additional keyword arguments that may be required by specific strategies.

        Returns:
            A tuple containing two lists of lists:
                - The first list contains lists of indices for the training data partitions.
                - The second list contains lists of indices for the test data partitions.

        """
        pass


class RandomIIDPartitionStrategy(DataPartitionStrategy):
    """Partition the dataset randomly, resulting in an IID distribution of data across clients."""

    @staticmethod
    def generate_partitions(
        train_data: Dataset, test_data: Dataset, num_partitions: int, seed: int = 666, **kwargs
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Generate partitions of the dataset using random sampling.

        Args:
            train_data: The training Dataset object to partition.
            test_data: The test Dataset object to partition.
            num_partitions: The number of partitions to create.
            seed: The random seed to use for reproducibility.
            **kwargs: Additional keyword arguments that may be required by specific strategies.

        Returns:
            A tuple containing two lists of lists:
                - The first list contains lists of indices for the training data partitions.
                - The second list contains lists of indices for the test data partitions.

        """
        return (
            RandomIIDPartitionStrategy.__partition_data(train_data, seed, num_partitions),
            RandomIIDPartitionStrategy.__partition_data(test_data, seed, num_partitions),
        )

    @staticmethod
    def __partition_data(data: Dataset, seed: int, num_partitions: int) -> List[List[int]]:
        # Shuffle the indices
        indices = list(range(len(data)))
        random.seed(seed)
        random.shuffle(indices)

        # Get partition sizes
        samples_per_partition = len(data) // num_partitions
        remainder = len(data) % num_partitions

        # Partition the data using list comprehension
        # Each partition gets 'samples_per_partition' samples, and the first 'remainder' partitions get an extra sample
        return [
            indices[i * samples_per_partition + min(i, remainder) : (i + 1) * samples_per_partition + min(i + 1, remainder)]
            for i in range(num_partitions)
        ]


class LabelSkewedPartitionStrategy(DataPartitionStrategy):
    """
    Partitions the dataset by grouping samples with the same label, resulting in a non-IID distribution.

    This is generally considered the "worst-case" scenario for federated learning.
    """

    # CUANDO SE HAGA LA OTRA (NO-IID GENERALIZARLA)

    @staticmethod
    def generate_partitions(
        train_data: Dataset,
        test_data: Dataset,
        num_partitions: int,
        seed: int = 666,
        label_tag: str = "label",
        **kwargs,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Generate partitions of the dataset by grouping samples with the same label.

        Args:
            train_data: The training Dataset object to partition.
            test_data: The test Dataset object to partition.
            num_partitions: The number of partitions to create.
            seed: The random seed to use for reproducibility.
            label_tag: The name of the column containing the labels.
            **kwargs: Additional keyword arguments that may be required by specific strategies.

        Returns:
            A tuple containing two lists of lists:
                - The first list contains lists of indices for the training data partitions.
                - The second list contains lists of indices for the test data partitions.

        """
        raise NotImplementedError("LabelSkewedPartitionStrategy is not implemented yet. TEST!")
        train_partitions = []
        test_partitions = []

        # Partition the training data
        sorted_train_indices = train_data.sort("label")
        random.seed(seed)
        random.shuffle(sorted_train_indices)  # Shuffle within label groups
        train_partitions = [sorted_train_indices[i::num_partitions].tolist() for i in range(num_partitions)]

        # Partition the test data
        sorted_test_indices = test_data.sort("label").indices
        random.seed(seed)
        random.shuffle(sorted_test_indices)  # Shuffle within label groups
        test_partitions = [sorted_test_indices[i::num_partitions].tolist() for i in range(num_partitions)]

        return train_partitions, test_partitions


class DirichletPartitionStrategy(DataPartitionStrategy):
    """Not implemented yet."""

    pass


class PercentageBasedNonIIDPartitionStrategy(DataPartitionStrategy):
    """Not implemented yet."""

    pass
