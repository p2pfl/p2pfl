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
from typing import List, Tuple, Union

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
    """
    Explanation.

    Explanation.
    """

    @staticmethod
    def generate_partitions(
        train_data: Dataset,
        test_data: Dataset,
        num_partitions: int,
        seed: int = 666,
        label_tag: str = "label",
        alpha: Union[int, float, list[float]] = 1,
        min_partition_size: int = 10,
        self_balancing: bool = False,
        shuffle: bool = True,
        **kwargs,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Generate partitions of the dataset using Dirichlet.

        Args:
            train_data: The training Dataset object to partition.
            test_data: The test Dataset object to partition.
            num_partitions: The number of partitions to create.
            seed: The random seed to use for reproducibility.
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
        return (
            [], [] # train and test list of indexes
        )

    def __partition_data():
        # Attributes based on the constructor
        self._num_partitions = num_partitions
        self._check_num_partitions_greater_than_zero()
        self._alpha: NDArrayFloat = self._initialize_alpha(alpha)
        self._partition_by = partition_by
        self._min_partition_size: int = min_partition_size
        self._self_balancing = self_balancing
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)  # NumPy random generator

        # Utility attributes
        # The attributes below are determined during the first call to load_partition
        self._avg_num_of_samples_per_partition: Optional[float] = None
        self._unique_classes: Optional[Union[list[int], list[str]]] = None
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_indices_determined = False


    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single partition of a dataset
        """
        # The partitioning is done lazily - only when the first partition is
        # requested. Only the first call creates the indices assignments for all the
        # partition indices.
        self._check_num_partitions()
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])


    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._check_num_partitions()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    def _initialize_alpha(
        self, alpha: Union[int, float, list[float], NDArrayFloat]
    ) -> NDArrayFloat:
        """Convert alpha to the used format in the code a NDArrayFloat.

        The alpha can be provided in constructor can be in different format for user
        convenience. The format into which it's transformed here is used throughout the
        code for computation.

        Parameters
        ----------
            alpha : Union[int, float, List[float], NDArrayFloat]
                Concentration parameter to the Dirichlet distribution

        Returns
        -------
        alpha : NDArrayFloat
            Concentration parameter in a format ready to used in computation.
        """
        if isinstance(alpha, int):
            alpha = np.array([float(alpha)], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, float):
            alpha = np.array([alpha], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, list):
            if len(alpha) != self._num_partitions:
                raise ValueError(
                    "If passing alpha as a List, it needs to be of length of equal to "
                    "num_partitions."
                )
            alpha = np.asarray(alpha)
        elif isinstance(alpha, np.ndarray):
            # pylint: disable=R1720
            if alpha.ndim == 1 and alpha.shape[0] != self._num_partitions:
                raise ValueError(
                    "If passing alpha as an NDArray, its length needs to be of length "
                    "equal to num_partitions."
                )
            elif alpha.ndim == 2:
                alpha = alpha.flatten()
                if alpha.shape[0] != self._num_partitions:
                    raise ValueError(
                        "If passing alpha as an NDArray, its size needs to be of length"
                        " equal to num_partitions."
                    )
        else:
            raise ValueError("The given alpha format is not supported.")
        if not (alpha > 0).all():
            raise ValueError(
                f"Alpha values should be strictly greater than zero. "
                f"Instead it'd be converted to {alpha}"
            )
        return alpha

    def _determine_partition_id_to_indices_if_needed(
        self,
    ) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return

        # Generate information needed for Dirichlet partitioning
        self._unique_classes = self.dataset.unique(self._partition_by)
        assert self._unique_classes is not None
        # This is needed only if self._self_balancing is True (the default option)
        self._avg_num_of_samples_per_partition = (
            self.dataset.num_rows / self._num_partitions
        )

        # Change targets list data type to numpy
        targets = np.array(self.dataset[self._partition_by])

        # Repeat the sampling procedure based on the Dirichlet distribution until the
        # min_partition_size is reached.
        sampling_try = 0
        while True:
            # Prepare data structure to store indices assigned to partition ids
            partition_id_to_indices: dict[int, list[int]] = {}
            for nid in range(self._num_partitions):
                partition_id_to_indices[nid] = []

            # Iterated over all unique labels (they are not necessarily of type int)
            for k in self._unique_classes:
                # Access all the indices associated with class k
                indices_representing_class_k = np.nonzero(targets == k)[0]
                # Determine division (the fractions) of the data representing class k
                # among the partitions
                class_k_division_proportions = self._rng.dirichlet(self._alpha)
                nid_to_proportion_of_k_samples = {}
                for nid in range(self._num_partitions):
                    nid_to_proportion_of_k_samples[nid] = class_k_division_proportions[
                        nid
                    ]
                # Balancing (not mentioned in the paper but implemented)
                # Do not assign additional samples to the partition if it already has
                # more than the average numbers of samples per partition. Note that it
                # might especially affect classes that are later in the order. This is
                # the reason for more sparse division that the alpha might suggest.
                if self._self_balancing:
                    assert self._avg_num_of_samples_per_partition is not None
                    for nid in nid_to_proportion_of_k_samples.copy():
                        if (
                            len(partition_id_to_indices[nid])
                            > self._avg_num_of_samples_per_partition
                        ):
                            nid_to_proportion_of_k_samples[nid] = 0

                    # Normalize the proportions such that they sum up to 1
                    sum_proportions = sum(nid_to_proportion_of_k_samples.values())
                    for nid, prop in nid_to_proportion_of_k_samples.copy().items():
                        nid_to_proportion_of_k_samples[nid] = prop / sum_proportions

                # Determine the split indices
                cumsum_division_fractions = np.cumsum(
                    list(nid_to_proportion_of_k_samples.values())
                )
                cumsum_division_numbers = cumsum_division_fractions * len(
                    indices_representing_class_k
                )
                # [:-1] is because the np.split requires the division indices but the
                # last element represents the sum = total number of samples
                indices_on_which_split = cumsum_division_numbers.astype(int)[:-1]

                split_indices = np.split(
                    indices_representing_class_k, indices_on_which_split
                )

                # Append new indices (coming from class k) to the existing indices
                for nid, indices in partition_id_to_indices.items():
                    indices.extend(split_indices[nid].tolist())

            # Determine if the indices assignment meets the min_partition_size
            # If it does not mean the requirement repeat the Dirichlet sampling process
            # Otherwise break the while loop
            min_sample_size_on_client = min(
                len(indices) for indices in partition_id_to_indices.values()
            )
            if min_sample_size_on_client >= self._min_partition_size:
                break
            sample_sizes = [
                len(indices) for indices in partition_id_to_indices.values()
            ]
            alpha_not_met = [
                self._alpha[i]
                for i, ss in enumerate(sample_sizes)
                if ss == min(sample_sizes)
            ]
            mssg_list_alphas = (
                (
                    "Generating partitions by sampling from a list of very wide range "
                    "of alpha values can be hard to achieve. Try reducing the range "
                    f"between maximum ({max(self._alpha)}) and minimum alpha "
                    f"({min(self._alpha)}) values or increasing all the values."
                )
                if len(self._alpha.flatten().tolist()) > 0
                else ""
            )
            warnings.warn(
                f"The specified min_partition_size ({self._min_partition_size}) was "
                f"not satisfied for alpha ({alpha_not_met}) after "
                f"{sampling_try} attempts at sampling from the Dirichlet "
                f"distribution. The probability sampling from the Dirichlet "
                f"distribution will be repeated. Note: This is not a desired "
                f"behavior. It is recommended to adjust the alpha or "
                f"min_partition_size instead. {mssg_list_alphas}",
                stacklevel=1,
            )
            if sampling_try == 10:
                raise ValueError(
                    "The max number of attempts (10) was reached. "
                    "Please update the values of alpha and try again."
                )
            sampling_try += 1

        # Shuffle the indices not to have the datasets with targets in sequences like
        # [00000, 11111, ...]) if the shuffle is True
        if self._shuffle:
            for indices in partition_id_to_indices.values():
                # In place shuffling
                self._rng.shuffle(indices)
        self._partition_id_to_indices = partition_id_to_indices
        self._partition_id_to_indices_determined = True

    def _check_num_partitions(num_partitions, min_rows) -> None:
        """Test num_partitions."""
        if num_partitions > min_rows:
            raise ValueError(
                "The number of partitions needs to be smaller than the number of "
                "samples in the smallest dataset."
            )
        if not num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")
        if int(num_partitions) != num_partitions:
            raise ValueError("The number of partitions needs to be an integer")





class PercentageBasedNonIIDPartitionStrategy(DataPartitionStrategy):
    """Not implemented yet."""

    pass
