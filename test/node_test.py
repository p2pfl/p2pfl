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

import time

import pytest
import torch

from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.learning.pytorch.lightning_learner import LightningModel
from p2pfl.learning.pytorch.torch_model import MLP
from p2pfl.node import Node
from p2pfl.utils import (
    check_equal_models,
    set_test_settings,
    wait_to_finish,
    wait_convergence,
)

set_test_settings()


@pytest.fixture
def two_nodes():
    """Create two nodes and start them. Yield the nodes. After the test, stop the nodes."""
    data = P2PFLDataset.from_huggingface("p2pfl/mnist")
    n1 = Node(LightningModel(MLP()), data)
    n2 = Node(LightningModel(MLP()), data)
    n1.start()
    n2.start()

    yield n1, n2

    n1.stop()
    n2.stop()


@pytest.fixture
def four_nodes():
    """Create four nodes and start them. Yield the nodes. After the test, stop the nodes."""
    data = P2PFLDataset.from_huggingface("p2pfl/mnist")
    partitions = data.generate_partitions(20, RandomIIDPartitionStrategy)
    n1 = Node(LightningModel(MLP()), partitions[0])
    n2 = Node(LightningModel(MLP()), partitions[1])
    n3 = Node(LightningModel(MLP()), partitions[2])
    n4 = Node(LightningModel(MLP()), partitions[3])
    nodes = [n1, n2, n3, n4]
    [n.start() for n in nodes]

    yield (n1, n2, n3, n4)

    [n.stop() for n in nodes]


########################
#    Tests Learning    #
########################

@pytest.mark.parametrize("x", [(2, 1), (2, 2)])
def test_convergence(x):
    """Test convergence (on learning) of two nodes."""
    n, r = x

    # Node Creation
    nodes = []
    for _ in range(n):
        node = Node(
            LightningModel(MLP()),
            P2PFLDataset.from_huggingface("p2pfl/MNIST")
        )
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
    wait_to_finish(nodes)

    # Check if execution is correct
    for node in nodes:
        # gt
        round_stages = ['VoteTrainSetStage', 'TrainStage', 'GossipModelStage', 'RoundFinishedStage'] * r
        assert node.learning_workflow.history == ['StartLearningStage'] + round_stages

    check_equal_models(nodes)

    # Stop Nodes
    [n.stop() for n in nodes]


def test_interrupt_train(two_nodes):
    """Test interrupting training of a node."""
    n1, n2 = two_nodes
    n1.connect(n2.addr)
    wait_convergence([n1, n2], 1, only_direct=True)

    n1.set_start_learning(100, 100)

    time.sleep(1)  # Wait because of asincronity

    n1.set_stop_learning()

    wait_to_finish([n1, n2])

    # Check if execution is incorrect
    assert 'RoundFinishedStage' not in n1.learning_workflow.history
    assert 'RoundFinishedStage' not in n2.learning_workflow.history


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
    data = P2PFLDataset.from_huggingface("p2pfl/mnist")
    for _ in range(n):
        node = Node(LightningModel(MLP()), data)
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
    assert 'RoundFinishedStage' not in nodes[-1].learning_workflow.history
    for node in nodes[:-1]:
        assert 'RoundFinishedStage' in node.learning_workflow.history

    for node in nodes[:-1]:
        node.stop()
