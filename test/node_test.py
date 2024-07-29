#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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

from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import (
    MnistFederatedDM,
)
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
from p2pfl.utils import (
    check_equal_models,
    set_test_settings,
    wait_4_results,
    wait_convergence,
)

set_test_settings()


@pytest.fixture
def two_nodes():
    """Create two nodes and start them. Yield the nodes. After the test, stop the nodes."""
    n1 = Node(MLP(), MnistFederatedDM())
    n2 = Node(MLP(), MnistFederatedDM())
    n1.start()
    n2.start()

    yield n1, n2

    n1.stop()
    n2.stop()


@pytest.fixture
def four_nodes():
    """Create four nodes and start them. Yield the nodes. After the test, stop the nodes."""
    n1 = Node(MLP(), MnistFederatedDM())
    n2 = Node(MLP(), MnistFederatedDM())
    n3 = Node(MLP(), MnistFederatedDM())
    n4 = Node(MLP(), MnistFederatedDM())
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
        node = Node(MLP(), MnistFederatedDM())
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes) - 1):
        nodes[i + 1].connect(nodes[i].addr)
        time.sleep(0.1)
    wait_convergence(nodes, n - 1, only_direct=False)

    # Start Learning
    nodes[0].set_start_learning(rounds=r, epochs=0)

    # Wait and check
    wait_4_results(nodes)
    check_equal_models(nodes)

    # Stop Nodes
    [n.stop() for n in nodes]


def test_interrupt_train(two_nodes):
    """Test interrupting training of a node."""
    if (
        __name__ == "__main__"
    ):  # To avoid creating new process when current has not finished its bootstrapping phase
        n1, n2 = two_nodes
        n1.connect(n2.addr)
        wait_convergence([n1, n2], 1, only_direct=True)

        n1.set_start_learning(100, 100)

        time.sleep(1)  # Wait because of asincronity

        n1.set_stop_learning()

        wait_4_results([n1, n2])


##############################
#    Fault Tolerace Tests    #
##############################


@pytest.mark.parametrize("n", [2, 4])
def test_node_down_on_learning(n):
    """Test node down on learning."""
    # Node Creation
    nodes = []
    for _ in range(n):
        node = Node(MLP(), MnistFederatedDM())
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
    time.sleep(0.3)
    nodes[-1].stop()

    wait_4_results(nodes)

    for node in nodes[:-1]:
        node.stop()


def test_wrong_model():
    """Test sending a wrong model."""
    n1 = Node(MLP(), MnistFederatedDM())
    n2 = Node(CNN(), MnistFederatedDM())

    n1.start()
    n2.start()

    n1.connect(n2.addr)
    time.sleep(0.1)

    n1.set_start_learning(rounds=2, epochs=0)
    time.sleep(0.1)

    wait_4_results([n1, n2])

    # CHANGE THIS WHEN STOP NODE CHANGES TO DISCONECTION
    try:
        n1.stop()
        n2.stop()
    except BaseException:
        pass
