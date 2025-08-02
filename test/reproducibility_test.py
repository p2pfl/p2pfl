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
"""

Reproducibility tests.

These tests are not super important, but they are a good way to check that the code is reproducible.

Not recomended to run always, as they are slow.
"""

import contextlib

import numpy as np
import pytest  # noqa: E402, I001
from datasets import DatasetDict, load_dataset  # noqa: E402, I001

from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset  # noqa: E402
from p2pfl.learning.dataset.partition_strategies import DirichletPartitionStrategy, RandomIIDPartitionStrategy
from p2pfl.learning.frameworks.learner_factory import LearnerFactory
from p2pfl.management.logger import logger
from p2pfl.node import Node  # noqa: E402
from p2pfl.settings import Settings
from p2pfl.utils.check_ray import ray_installed
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import (  # noqa: E402
    set_standalone_settings,
    wait_convergence,
    wait_to_finish,
)

with contextlib.suppress(ImportError):
    from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn as model_build_fn_tensorflow

with contextlib.suppress(ImportError):
    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_pytorch

set_standalone_settings()

##
# Schocastic P2PFL processes (voting for now)
##


def __fl_without_training(seed):
    # Set seed
    Settings.general.SEED = seed

    # Create X nodes (model and data do not matter)
    nodes = [
        Node(model_build_fn_pytorch(), P2PFLDataset.from_huggingface("p2pfl/MNIST"), protocol=MemoryCommunicationProtocol())
        for _ in range(5)
    ]

    try:
        [node.start() for node in nodes]

        # Connect the nodes
        adjacency_matrix = TopologyFactory.generate_matrix(TopologyType.STAR, len(nodes))
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)

        # Wait for convergence
        wait_convergence(nodes, len(nodes) - 1, only_direct=False)

        # Start learning for 6 rounds, no training
        nodes[0].set_start_learning(rounds=2, epochs=0)

        # Wait to finish
        wait_to_finish(nodes, timeout=240)

    finally:
        [node.stop() for node in nodes]

    # Return the stages of the nodes.
    nodes = sorted(nodes, key=lambda x: x.addr)
    print([node.addr for node in nodes])

    return [node.learning_workflow.history for node in nodes]


# TODO: Merge this test with the global training one
def test_voting_reproducibility():
    """Test that seed ensures reproducible voting results."""
    nodes_stages_1 = __fl_without_training(666)
    nodes_stages_2 = __fl_without_training(666)
    nodes_stages_3 = __fl_without_training(777)
    assert nodes_stages_1 == nodes_stages_2
    assert nodes_stages_1 != nodes_stages_3


##
# Dataset Partitioning
##


@pytest.mark.parametrize(
    "strategy, strategy_kwargs",
    [
        (RandomIIDPartitionStrategy, {}),
        (DirichletPartitionStrategy, {"alpha": 0.5}),
    ],
)
def test_set_dataset_partition_reproducibility(strategy, strategy_kwargs):
    """Test that seed ensures reproducibility for partitioning strategies."""
    # Dataset
    mnist_dataset = P2PFLDataset(
        DatasetDict(
            {
                "train": load_dataset("p2pfl/MNIST", split="train[:100]"),
                "test": load_dataset("p2pfl/MNIST", split="test[:10]"),
            }
        )
    )
    # Test 1: Same global seed, same strategy seed -> same partitions
    Settings.general.SEED = 666
    partitions1 = mnist_dataset.generate_partitions(num_partitions=3, strategy=strategy, **strategy_kwargs)
    Settings.general.SEED = 666
    partitions2 = mnist_dataset.generate_partitions(num_partitions=3, strategy=strategy, **strategy_kwargs)

    # Verify partitions are the same
    for i in range(3):
        # Compare train data indices
        assert (
            partitions1[i]._data[partitions1[i]._train_split_name]._indices
            == partitions2[i]._data[partitions2[i]._train_split_name]._indices
        )

        # Compare test data indices
        assert (
            partitions1[i]._data[partitions1[i]._test_split_name]._indices == partitions2[i]._data[partitions2[i]._test_split_name]._indices
        )

    # Test 2: Different strategy seed -> different partitions
    Settings.general.SEED = 777
    partitions3 = mnist_dataset.generate_partitions(num_partitions=3, strategy=strategy, **strategy_kwargs)

    # Verify at least one partition is different
    different_strategy_seed = False
    for i in range(3):
        if (
            partitions1[i]._data[partitions1[i]._train_split_name]._indices
            != partitions3[i]._data[partitions3[i]._train_split_name]._indices
        ):
            different_strategy_seed = True
            break

    assert different_strategy_seed, "Partitions should be different with different strategy seeds"


##
# Model
##


@pytest.mark.parametrize("model_build_fn", [model_build_fn_pytorch, model_build_fn_tensorflow])  # , model_build_fn_flax])
def test_model_initialization_reproducibility(model_build_fn):
    """Test that seed ensures reproducible model initialization."""
    try:
        # First initialization with seed
        Settings.general.SEED = 666
        params1 = model_build_fn().get_parameters()

        # Second initialization with same seed
        Settings.general.SEED = 666
        params2 = model_build_fn().get_parameters()

        # Assert parameters are identical
        for p1, p2 in zip(params1, params2, strict=False):
            assert np.array_equal(p1, p2), "Model parameters differ despite using the same seed"

        # Different seed should produce different parameters
        Settings.general.SEED = 777
        params3 = model_build_fn().get_parameters()

        # At least one parameter should be different
        any_different = False
        for p1, p3 in zip(params1, params3, strict=False):
            if not np.array_equal(p1, p3):
                any_different = True
                break
        assert any_different, "Different seeds produced identical model parameters"

    except ImportError:
        pytest.skip("PyTorch not available")


##
# Training
##


@pytest.mark.skip(reason="Working but slow....")
@pytest.mark.parametrize("model_build_fn", [model_build_fn_pytorch, model_build_fn_tensorflow])  # model_build_fn_flax
def test_local_training_reproducibility(model_build_fn):
    """Test that seed ensures reproducible training results."""
    try:
        # Create a small dataset for testing
        dataset = P2PFLDataset(
            DatasetDict({"train": load_dataset("p2pfl/MNIST", split="train[:20]"), "test": load_dataset("p2pfl/MNIST", split="test[:10]")})
        )

        # First training run with seed
        Settings.general.SEED = 666
        model1 = model_build_fn(lr_rate=2.0)  # High learning rate to ensure that schocastic differences are visible
        learner1 = LearnerFactory.create_learner(model1)()
        learner1.set_data(dataset)
        learner1.set_model(model1)
        learner1.set_addr("test-node-1")
        learner1.set_epochs(1)
        learner1.fit()
        params1 = model1.get_parameters()
        eval1 = learner1.evaluate()

        Settings.general.SEED = 666
        model1_1 = model_build_fn(lr_rate=2.0)  # High learning rate to ensure that schocastic differences are visible
        learner1_1 = LearnerFactory.create_learner(model1_1)()
        learner1_1.set_data(dataset)
        learner1_1.set_model(model1_1)
        learner1_1.set_addr("test-node-1")
        learner1_1.set_epochs(1)
        learner1_1.fit()
        params1_1 = model1_1.get_parameters()
        eval1_1 = learner1_1.evaluate()

        # Assert parameters and evaluation metrics are identical or very close
        for p1, p1_1 in zip(params1, params1_1, strict=False):
            assert np.array_equal(p1, p1_1), "Model parameters are not identical despite using equal seeds"

        for metric in eval1:
            assert np.array_equal(eval1[metric], eval1_1[metric]), f"Evaluation metric {metric} is not identical despite using equal seeds"

        # Second training run with same seed
        Settings.general.SEED = 777
        model2 = model_build_fn(lr_rate=2.0)  # High learning rate to ensure that schocastic differences are visible
        learner2 = LearnerFactory.create_learner(model2)()
        learner2.set_data(dataset)
        learner2.set_model(model2)
        learner2.set_addr("test-node-2")
        learner2.set_epochs(1)
        learner2.fit()
        params2 = model2.get_parameters()

        # Assert parameters and evaluation metrics are identical or very close
        for p1, p2 in zip(params1, params2, strict=False):
            assert not np.array_equal(p1, p2), "Model parameters not differ despite using different seed"

    except ImportError:
        pytest.skip("PyTorch not available")


def __train_with_seed(s, n, r, model_build_fn, disable_ray: bool = False):
    # Ray
    if disable_ray:
        Settings.general.DISABLE_RAY = True
    else:
        Settings.general.DISABLE_RAY = False

    assert ray_installed() != disable_ray

    # Seed
    Settings.general.SEED = s

    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(n * 50, RandomIIDPartitionStrategy)

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(model_build_fn(), partitions[i])
        node.start()
        nodes.append(node)

    # Connect the nodes
    adjacency_matrix = TopologyFactory.generate_matrix(TopologyType.STAR, len(nodes))
    TopologyFactory.connect_nodes(adjacency_matrix, nodes)

    # Start Learning
    exp_name = nodes[0].set_start_learning(rounds=r, epochs=1)

    # Wait
    wait_to_finish(nodes, timeout=240)

    # Stop Nodes
    [n.stop() for n in nodes]

    return exp_name


def __get_results(exp_name):
    # Get global metrics
    global_metrics = logger.get_global_logs()[exp_name]
    print(global_metrics)
    # Sort by node name
    global_metrics = dict(sorted(global_metrics.items(), key=lambda item: item[0]))
    # Get only the metrics
    global_metrics = list(global_metrics.values())

    # Get local metrics
    local_metrics = list(logger.get_local_logs()[exp_name].values())
    # Sort by node name and remove it
    local_metrics = [list(dict(sorted(r.items(), key=lambda item: item[0])).values()) for r in local_metrics]

    # Assert if empty
    if len(local_metrics) == 0:
        raise ValueError("No local metrics found")
    if len(global_metrics) == 0:
        raise ValueError("No global metrics found")

    # Return metrics
    return global_metrics, local_metrics


def __flatten_results(item):
    """
    Recursively flattens a nested structure and extracts numerical values.

    Args:
        item: The item to flatten, can be a list, dict, tuple, or number.

    Returns:
        A list of numerical values found in the item.

    """
    if isinstance(item, int | float):
        return [item]  # Base case: if it's a number, return it in a list
    elif isinstance(item, list):
        return [sub_item for element in item for sub_item in __flatten_results(element)]
    elif isinstance(item, dict):
        return [sub_item for value in item.values() for sub_item in __flatten_results(value)]
    elif isinstance(item, tuple):
        return [sub_item for element in item for sub_item in __flatten_results(element)]
    else:
        return []


@pytest.mark.skip(reason="Working but slow....")
@pytest.mark.parametrize(
    "input",
    [
        (model_build_fn_tensorflow, True),
        (model_build_fn_pytorch, False),
        (model_build_fn_tensorflow, False),
    ],
)
def test_global_training_reproducibility(input):
    """Test that seed ensures reproducible global training results."""
    model_build_fn, disable_ray = input
    n, r = 10, 1

    exp_name1 = __train_with_seed(666, n, r, model_build_fn, disable_ray)
    exp_name2 = __train_with_seed(666, n, r, model_build_fn, disable_ray)
    exp_name3 = __train_with_seed(777, n, r, model_build_fn, disable_ray)

    # Check if metrics are the same in the 2 trainings -> set seed works
    assert np.allclose(__flatten_results(__get_results(exp_name1)), __flatten_results(__get_results(exp_name2)))
    assert not np.allclose(__flatten_results(__get_results(exp_name2)), __flatten_results(__get_results(exp_name3)))
