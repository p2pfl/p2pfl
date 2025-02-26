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
"""Node tests."""

# Disable Ray on the experiment (if installed)
from p2pfl.settings import Settings

Settings.general.DISABLE_RAY = True

import contextlib  # noqa: E402, I001
import time  # noqa: E402
import pytest  # noqa: E402
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset  # noqa: E402
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy  # noqa: E402
from p2pfl.management.logger import logger  # noqa: E402
from p2pfl.node import Node  # noqa: E402
from p2pfl.utils.utils import (  # noqa: E402
    check_equal_models,
    set_standalone_settings,
    wait_convergence,
    wait_to_finish,
)

with contextlib.suppress(ImportError):
    from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn as model_build_fn_tensorflow


with contextlib.suppress(ImportError):
    import jax
    import jax.numpy as jnp

    from p2pfl.examples.mnist.model.mlp_flax import MLP as MLP_FLAX
    from p2pfl.learning.frameworks.flax.flax_model import FlaxModel

with contextlib.suppress(ImportError):
    from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn as model_build_fn_pytorch

set_standalone_settings()


@pytest.fixture
def two_nodes():
    """Create two nodes and start them. Yield the nodes. After the test, stop the nodes."""
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    n1 = Node(model_build_fn_pytorch(), data)
    n2 = Node(model_build_fn_pytorch(), data)
    n1.start()
    n2.start()

    yield n1, n2

    n1.stop()
    n2.stop()


########################
#    Tests Learning    #
########################


# TODO: Add more frameworks and aggregators
@pytest.mark.parametrize("x", [(2, 2), (6, 3)])
def test_convergence(x):
    """Test convergence (on learning) of two nodes."""
    n, r = x

    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(n * 50, RandomIIDPartitionStrategy)

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(model_build_fn_pytorch(), partitions[i])
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes) - 1):
        nodes[i + 1].connect(nodes[i].addr)
        time.sleep(0.1)
    wait_convergence(nodes, n - 1, only_direct=False)

    # Start Learning
    nodes[0].set_start_learning(rounds=r, epochs=0)

    # Wait
    wait_to_finish(nodes, timeout=240)

    # Check if execution is correct
    for node in nodes:
        # History
        history = node.learning_workflow.history
        assert history[0] == "StartLearningStage"
        history = history[1:]
        # Pattern
        stage_pattern = ["VoteTrainSetStage", ["TrainStage", "WaitAggregatedModelsStage"], "GossipModelStage", "RoundFinishedStage"]
        # Get batches (len(stage_pattern))
        assert int(len(history) / len(stage_pattern)) == r
        # Check pattern
        for i in range(r):
            for gt, st in zip(stage_pattern, history[i * len(stage_pattern) : (i + 1) * len(stage_pattern)]):
                if isinstance(gt, list):
                    assert st in gt
                else:
                    assert st == gt

    check_equal_models(nodes)

    # Get accuracies
    accuracies = [metrics["test_metric"] for metrics in list(logger.get_global_logs().values())[0].values()]
    # Get last round accuracies
    last_round_accuracies = [acc for node_acc in accuracies for r, acc in node_acc if r == 1]
    # Assert that the accuracies are higher than 0.5
    assert all(acc > 0.5 for acc in last_round_accuracies)

    # Stop Nodes
    [n.stop() for n in nodes]


# DISABLED! NOT IMPLEMENTED BY RAY/TF
def _test_interrupt_train(two_nodes):
    """Test interrupting training of a node."""
    n1, n2 = two_nodes
    n1.connect(n2.addr)
    wait_convergence([n1, n2], 1, only_direct=True)

    n1.set_start_learning(100, 100)

    time.sleep(1)  # Wait because of asincronity

    n1.set_stop_learning()

    wait_to_finish([n1, n2])

    # Check if execution is incorrect
    assert "RoundFinishedStage" not in n1.learning_workflow.history
    assert "RoundFinishedStage" not in n2.learning_workflow.history


##############################
#    Fault Tolerace Tests    #
##############################

"""
-> Énfasis on the trainset inconsistency
"""


@pytest.mark.parametrize("n", [2, 4])
def _test_node_down_on_learning(n):
    """Test node down on learning."""
    # Node Creation
    nodes = []
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    for _ in range(n):
        node = Node(model_build_fn_pytorch(), data)
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes) - 1):
        nodes[i + 1].connect(nodes[i].addr)
        time.sleep(0.1)
    wait_convergence(nodes, n - 1, only_direct=False)

    # Start Learning
    nodes[0].set_start_learning(rounds=2, epochs=0)

    # Stopping node
    time.sleep(1)
    nodes[-1].stop()

    wait_to_finish(nodes)

    # Check if execution is incorrect
    assert "RoundFinishedStage" not in nodes[-1].learning_workflow.history
    for node in nodes[:-1]:
        assert "RoundFinishedStage" in node.learning_workflow.history

    for node in nodes[:-1]:
        node.stop()


#####
# Training with other frameworks
#####


def test_tensorflow_node():
    """Test a TensorFlow node."""
    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(400, RandomIIDPartitionStrategy)

    # Create the model
    p2pfl_model = model_build_fn_tensorflow()

    # Nodes
    n1 = Node(p2pfl_model, partitions[0])
    n2 = Node(p2pfl_model.build_copy(), partitions[1])

    # Start
    n1.start()
    n2.start()

    # Connect
    n2.connect(n1.addr)
    wait_convergence([n1, n2], 1, only_direct=True)

    # Start Learning
    n1.set_start_learning(rounds=1, epochs=1)

    # Wait
    wait_to_finish([n1, n2], timeout=120)

    # Check if execution is correct
    for node in [n1, n2]:
        assert "RoundFinishedStage" in node.learning_workflow.history

    check_equal_models([n1, n2])

    # Stop
    n1.stop()
    n2.stop()


def test_torch_node():
    """Test a TensorFlow node."""
    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(400, RandomIIDPartitionStrategy)

    # Create the model
    p2pfl_model = model_build_fn_pytorch()

    # Nodes
    n1 = Node(p2pfl_model, partitions[0])
    n2 = Node(p2pfl_model.build_copy(), partitions[1])

    # Start
    n1.start()
    n2.start()

    # Connect
    n2.connect(n1.addr)
    wait_convergence([n1, n2], 1, only_direct=True)

    # Start Learning
    n1.set_start_learning(rounds=1, epochs=1)

    # Wait
    wait_to_finish([n1, n2], timeout=120)

    # Check if execution is correct
    for node in [n1, n2]:
        assert "RoundFinishedStage" in node.learning_workflow.history

    check_equal_models([n1, n2])

    # Stop
    n1.stop()
    n2.stop()


def test_flax_node():
    """Test a Flax node."""
    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(400, RandomIIDPartitionStrategy)
    model = MLP_FLAX()
    seed = jax.random.PRNGKey(0)
    model_params = model.init(seed, jnp.ones((1, 28, 28)))["params"]
    p2pfl_model = FlaxModel(model, model_params)
    # Nodes
    n1 = Node(p2pfl_model, partitions[0])
    n2 = Node(p2pfl_model.build_copy(), partitions[1])
    # Start
    n1.start()
    n2.start()
    # Connect
    n2.connect(n1.addr)
    wait_convergence([n1, n2], 1, only_direct=True)
    # Start Learning
    n1.set_start_learning(rounds=1, epochs=1)
    # Wait
    wait_to_finish([n1, n2], timeout=120)
    # Check if execution is correct
    for node in [n1, n2]:
        assert "RoundFinishedStage" in node.learning_workflow.history
    check_equal_models([n1, n2])
    # Stop
    n1.stop()
    n2.stop()


def __test_model_is_learning():
    pass
