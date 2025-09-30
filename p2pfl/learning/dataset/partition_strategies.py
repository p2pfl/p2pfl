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

import numpy as np
import pandas as pd
from datasets import Dataset  # type: ignore

from p2pfl.settings import Settings  # type: ignore


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
    ) -> tuple[list[list[int]], list[list[int]]]:
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
        train_data: Dataset, test_data: Dataset, num_partitions: int, **kwargs
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Generate partitions of the dataset using random sampling.

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
        return (
            RandomIIDPartitionStrategy.__partition_data(train_data, num_partitions),
            RandomIIDPartitionStrategy.__partition_data(test_data, num_partitions),
        )

    @staticmethod
    def __partition_data(data: Dataset, num_partitions: int) -> list[list[int]]:
        # Shuffle the indices
        indices = list(range(len(data)))
        random.Random(Settings.general.SEED).shuffle(indices)

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
        label_tag: str = "label",
        **kwargs,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Generate partitions of the dataset by grouping samples with the same label.

        Args:
            train_data: The training Dataset object to partition.
            test_data: The test Dataset object to partition.
            num_partitions: The number of partitions to create.
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
        random.Random(Settings.general.SEED).shuffle(sorted_train_indices)  # Shuffle within label groups
        train_partitions = [sorted_train_indices[i::num_partitions].tolist() for i in range(num_partitions)]

        # Partition the test data
        sorted_test_indices = test_data.sort("label").indices
        random.Random(Settings.general.SEED).shuffle(sorted_test_indices)  # Shuffle within label groups
        test_partitions = [sorted_test_indices[i::num_partitions].tolist() for i in range(num_partitions)]

        return train_partitions, test_partitions


class DirichletPartitionStrategy(DataPartitionStrategy):
    """
    Data partition strategy based on the Dirichlet distribution.

    It assigns data to different partitions (clients) so that the
    distribution of classes in each partition follows a Dirichlet distribution,
    where alpha determines the concentration of the distribution.

    Inspired by the implementation of flower. Thank you so much for taking FL to another level :)
    Original implementation: https://github.com/adap/flower/blob/main/datasets/flwr_datasets/partitioner/dirichlet_partitioner.py

    """

    @staticmethod
    def _preprocess_alpha(alpha: int | float | list[float], num_partitions: int) -> list[float]:
        """
        Convert alpha to the used format in the code (ndaarray).

        The alpha can be provided in constructor can be in different format for user
        convenience. The format into which it's transformed here is used throughout the
        code for computation.

        Args:
            alpha : Concentration parameter to the Dirichlet distribution
            num_partitions : Number of partitions to create

        Returns:
            alpha : concentration parameter in a format ready to used in computation.

        """
        if isinstance(alpha, int):
            alpha = [float(alpha)] * num_partitions
        elif isinstance(alpha, float):
            alpha = [alpha] * num_partitions
        elif isinstance(alpha, list):
            if len(alpha) != num_partitions:
                raise ValueError("If passing alpha as a List, it needs to be of length of equal to num_partitions.")
            alpha = [float(a) for a in alpha]
        else:
            raise ValueError("The given alpha format is not supported.")
        if not all(a > 0 for a in alpha):
            raise ValueError(f"Alpha values should be strictly greater than zero. Instead it'd be converted to {alpha}")
        return alpha

    @classmethod
    def _adapt_class_division_proportions(
        cls, class_division_proportions: list[float], active_partitions: list[bool] | None
    ) -> list[float]:
        """
        Adapt the class division proportions to the active partitions.

        Only used if self_balancing is True.

        Args:
            class_division_proportions: The proportions of the Dirichlet distribution.
            active_partitions: The partitions that are still active.

        """
        if active_partitions is None:
            return class_division_proportions
        else:
            unnormalized_result = [
                proportion * active for proportion, active in zip(class_division_proportions, active_partitions, strict=False)
            ]
            return [element / sum(unnormalized_result) for element in unnormalized_result]

    @classmethod
    def _calculate_assigned_proportion(cls, result: pd.DataFrame, class_proportions: dict[str | int, float]):
        return sum(class_proportions[label] * result[label] for label in result.columns)

    @classmethod
    def _recalculate_active_partitions(cls, result: pd.DataFrame, class_proportions: dict[str | int, float]) -> list[bool]:
        """
        Update the active_partitions based on how much proportion of the dataset is already assigned to each partition.

        Only used if self_balancing is True.

        Args:
            result: The proportions of the Dirichlet distribution.
            class_proportions: The proportions of the classes in the dataset

        """
        proportion_already_assigned = cls._calculate_assigned_proportion(result, class_proportions)
        partitions_to_keep = pd.Series(index=result.index, data=False)
        partitions_to_keep.loc[proportion_already_assigned < 1 / len(result)] = True
        active_partitions = partitions_to_keep.to_list()

        return active_partitions

    @classmethod
    def _generate_proportions(
        cls,
        num_partitions,
        class_proportions: dict[str | int, float],
        min_partition_proportion: float,
        alpha: list[float],
        random_generator: np.random.Generator,
        balancing: bool,
        max_tries: int = 10,
    ) -> pd.DataFrame:
        """
        Determine the proportions of the Dirichlet distribution.

        Args:
            num_partitions: The number of partitions to create.
            class_proportions: The proportions of the classes in the dataset.
            min_partition_proportion: The minimum partition size allowed.
            alpha: The alpha parameters of the Dirichlet distribution.
            random_generator: The random number generator.
            balancing: Whether the partitions should be balanced or not.
            max_tries: The maximum number of tries to find a valid partitioning.

        """
        if not abs(sum(class_proportions.values()) - 1.0) < 1e-10:
            raise ValueError("The sum of the class proportions must be 1")

        for _ in range(max_tries):
            result = pd.DataFrame(index=pd.RangeIndex(start=0, stop=num_partitions, name="partition"))
            active_partitions = [True] * num_partitions if balancing else None

            for class_label in class_proportions:
                division_proportions = cls._adapt_class_division_proportions(
                    class_division_proportions=list(random_generator.dirichlet(alpha)), active_partitions=active_partitions
                )
                result[class_label] = division_proportions

                if active_partitions is not None:
                    active_partitions = cls._recalculate_active_partitions(result, class_proportions)

            if min(cls._calculate_assigned_proportion(result, class_proportions)) >= min_partition_proportion:
                return result

            # Here it should be implemented a warning saying that the min condition is not satisfied and that
            # the sampling will be repeated. This is not a desired behavior.
        raise ValueError("Could not find a valid partitioning after max_tries. Try with other parameters.")

    @classmethod
    def _apply_proportions(
        cls,
        index_list: list[int],
        proportions: pd.Series,
        generator: np.random.Generator,
    ):
        """
        Use the proportions to get the list of indexes for each partition.

        Args:
            index_list: The list of indexes to partition.
            proportions: The proportions of the Dirichlet distribution.
            generator: The random number generator to shuffle the indexes.

        """
        right_sides = (proportions.cumsum() * len(index_list)).round().astype(int)
        generator.shuffle(index_list)
        return [index_list[: right_sides[0]]] + [index_list[right_sides[i - 1] : right_sides[i]] for i in range(1, len(right_sides))]

    @classmethod
    def _partition_data(
        cls,
        data: Dataset,
        label_tag: str,
        num_partitions: int,
        min_partition_size: int,
        alpha: list[float],
        random_generator: np.random.Generator,
        balancing: bool,
        max_tries: int = 10,
    ) -> list[list[int]]:
        """
        Partition the data and return the list of indexes.

        Args:
            data: The dataset to partition.
            label_tag: The name of the column containing the labels.
            num_partitions: The number of partitions to create.
            min_partition_size: The minimum partition size allowed.
            alpha: The alpha parameters of the Dirichlet distribution.
            random_generator: The random number generator.
            balancing: Whether the partitions should be balanced or not.
            max_tries: The maximum number of tries to find a valid partitioning.

        """
        class_proportions = {label: data[label_tag].count(label) / len(data[label_tag]) for label in set(data[label_tag])}
        proportions = cls._generate_proportions(
            num_partitions=num_partitions,
            class_proportions=class_proportions,
            min_partition_proportion=min_partition_size / len(data),
            alpha=alpha,
            random_generator=random_generator,
            balancing=balancing,
            max_tries=max_tries,
        )

        result: list[list[int]] = [[] for _ in range(num_partitions)]

        for label in class_proportions:
            label_index = [idx for idx, lab in enumerate(data[label_tag]) if lab == label]
            for partition, index_list in enumerate(cls._apply_proportions(label_index, proportions[label], random_generator)):
                result[partition].extend(index_list)

        return result

    @classmethod
    def generate_partitions(
        cls,
        train_data: Dataset,
        test_data: Dataset,
        num_partitions: int,
        label_tag: str = "label",
        alpha: int | float | list[float] = 1,
        min_partition_size: int = 2,
        self_balancing: bool = False,
        **kwargs,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Generate partitions of the dataset using Dirichlet distribution.

        It divides the data into partitions so that the distribution of classes in each partition
        follows a Dirichlet distribution controlled by the alpha parameter.

        Args:
            train_data: The training Dataset object to partition.
            test_data: The test Dataset object to partition.
            num_partitions: The number of partitions to create.
            label_tag: The name of the column containing the labels.
            alpha: The alpha parameters of the dirichlet distribution
            min_partition_size: The minimum partition size allowed in train and test.
            self_balancing: Whether the partitions should be balanced or not.
                The balancing is done by not allowing some label values to go
                in partitions that are already overly big.
            shuffle: Whether to shuffle the indexes or not
            **kwargs: Additional keyword arguments that may be required by specific strategies.

        Returns:
            A tuple containing two lists of lists:
                - The first list contains lists of indices for the training data partitions.
                - The second list contains lists of indices for the test data partitions.

        """
        alpha = cls._preprocess_alpha(alpha, num_partitions)
        cls._check_num_partitions(num_partitions=num_partitions, len_smallest_dataset=min(len(train_data), len(test_data)))

        random_generator = np.random.default_rng(Settings.general.SEED)
        return cls._partition_data(
            data=train_data,
            label_tag=label_tag,
            num_partitions=num_partitions,
            min_partition_size=min_partition_size,
            alpha=alpha,
            random_generator=random_generator,
            balancing=self_balancing,
        ), cls._partition_data(
            data=test_data,
            label_tag=label_tag,
            num_partitions=num_partitions,
            min_partition_size=min_partition_size,
            alpha=alpha,
            random_generator=random_generator,
            balancing=self_balancing,
        )

    @classmethod
    def _check_num_partitions(cls, num_partitions, len_smallest_dataset) -> None:
        """Test num_partitions."""
        if num_partitions > len_smallest_dataset:
            raise ValueError("The number of partitions needs to be smaller than the number of samples in the smallest dataset.")
        if not num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")
        if int(num_partitions) != num_partitions:
            raise ValueError("The number of partitions needs to be an integer")


class PercentageBasedNonIIDPartitionStrategy(DataPartitionStrategy):
    """Not implemented yet."""

    pass
