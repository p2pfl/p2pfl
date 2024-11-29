#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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
"""P2PFL dataset tests."""

import numpy as np
import pytest
from datasets import DatasetDict, load_dataset  # type: ignore

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import DirichletPartitionStrategy, RandomIIDPartitionStrategy


@pytest.fixture
def mnist_dataset() -> P2PFLDataset:
    """Load a small subset of the MNIST dataset."""
    hg_ds = DatasetDict(
        {
            "train": load_dataset("p2pfl/MNIST", split="train[:100]"),
            "test": load_dataset("p2pfl/MNIST", split="test[:10]"),
        }
    )

    return P2PFLDataset(hg_ds)


def __test_mnist_sample(item):
    assert "image" in item
    assert "label" in item

    assert item["image"].size == (28, 28)
    assert item["label"] in range(10)


def test_mnist_sample(mnist_dataset):
    """Test the data loading for the MNIST dataset."""
    assert mnist_dataset.get_num_samples(train=True) == 100
    assert mnist_dataset.get_num_samples(train=False) == 10

    item = mnist_dataset.get(0, train=True)

    __test_mnist_sample(item)


def test_generate_train_test_split():
    """Test the generation of train-test splits."""
    dataset = P2PFLDataset.from_huggingface("p2pfl/MNIST", split="train[:100]")
    dataset.generate_train_test_split(test_size=0.2, seed=42)
    assert isinstance(dataset._data, DatasetDict)
    assert "train" in dataset._data
    assert "test" in dataset._data


def test_partition_without_split():
    """Test the export of the dataset without a split."""
    dataset = P2PFLDataset(load_dataset("p2pfl/MNIST", split="train[:100]"))

    # Check that assert
    with pytest.raises(ValueError):
        dataset.generate_partitions(num_partitions=3, strategy=None)


def test_export_without_split_partition():
    """Test the export of the dataset without a split."""
    dataset = P2PFLDataset(load_dataset("p2pfl/MNIST", split="train[:100]"))

    # Check that assert (no split)
    with pytest.raises(ValueError):
        dataset.export(None)


@pytest.mark.parametrize(
    "strategy",
    [
        DirichletPartitionStrategy,
        RandomIIDPartitionStrategy,
    ],
)
def test_generate_partitions(mnist_dataset, strategy):
    """Test the generation of partitions."""
    num_partitions = 3
    partitions = mnist_dataset.generate_partitions(num_partitions=num_partitions, strategy=strategy)

    # check
    assert len(partitions) == 3

    train_size = mnist_dataset.get_num_samples(train=True)
    test_size = mnist_dataset.get_num_samples(train=False)

    # Check that all the indexes are unique and that they are all in the dataset
    partitions_train_samples = sum([partition.get_num_samples(train=True) for partition in partitions])
    partitions_test_samples = sum([partition.get_num_samples(train=False) for partition in partitions])
    assert partitions_train_samples == train_size
    assert partitions_test_samples == test_size

    # Check item
    item = partitions[0].get(0, train=True)
    __test_mnist_sample(item)


@pytest.mark.parametrize(
    "num_partitions, class_proportions, min_partition_proportion, alpha, balancing, expected_A, expected_B",
    [
        (3, {"A": 0.9, "B": 0.1}, 0.05, [1, 1, 3], True, [0.2, 0.2, 0.6], [0.5, 0.5, 0.0]),
        (3, {"A": 0.9, "B": 0.1}, 0.05, [1, 1, 3], False, [0.2, 0.2, 0.6], [0.2, 0.2, 0.6]),
        (3, {"A": 0.5, "B": 0.5}, 0.05, [1, 1, 3], True, [0.2, 0.2, 0.6], [0.2, 0.2, 0.6]),
        (3, {"A": 0.5, "B": 0.5}, 0.1, [0.09, 0.5, 0.41], False, None, None),
    ],
)
def test_dirichlet_generate_proportions(
    num_partitions, class_proportions, min_partition_proportion, alpha, balancing, expected_A, expected_B
):
    """Test to check proportions of dirichlet sampling."""
    random_generator = np.random.default_rng(seed=1)
    M = 10**10
    alpha = [a * M for a in alpha]

    if expected_A is None:
        with pytest.raises(ValueError):
            DirichletPartitionStrategy._generate_proportions(
                num_partitions=num_partitions,
                class_proportions=class_proportions,
                min_partition_proportion=min_partition_proportion,
                alpha=alpha,
                random_generator=random_generator,
                balancing=balancing,
            )
        return

    result = DirichletPartitionStrategy._generate_proportions(
        num_partitions=num_partitions,
        class_proportions=class_proportions,
        min_partition_proportion=min_partition_proportion,
        alpha=alpha,
        random_generator=random_generator,
        balancing=balancing,
    )

    assert np.isclose(result["A"].to_list(), expected_A, rtol=1e-3).all()
    assert np.isclose(result["B"].to_list(), expected_B, rtol=1e-3).all()
