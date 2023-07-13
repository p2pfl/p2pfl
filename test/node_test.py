#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/federated_learning_p2p).
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

from test.utils import (
    check_equal_models,
    set_test_settings,
    wait_4_results,
    wait_convergence,
)
set_test_settings()
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import time
import pytest


@pytest.fixture
def two_nodes():
    n1 = Node(MLP(), MnistFederatedDM())
    n2 = Node(MLP(), MnistFederatedDM())
    n1.start()
    n2.start()

    yield n1, n2

    n1.stop()
    n2.stop()


@pytest.fixture
def four_nodes():
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

"""
@pytest.mark.parametrize("x", [(2, 1), (2, 2)])
def test_convergence(x):
    n, r = x

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(MLP(), MnistFederatedDM())
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes) - 1):
        nodes[i + 1].connect(nodes[i].addr)
        time.sleep(0.1)
    wait_convergence(nodes, n-1, only_direct=True)

    # Start Learning
    nodes[0].set_start_learning(rounds=r, epochs=0)

    # Wait and check
    wait_4_results(nodes)
    check_equal_models(nodes)

    # Stop Nodes
    [n.stop() for n in nodes]

def test_interrupt_train(two_nodes):
    if (
        __name__ == "__main__"
    ):  # To avoid creating new process when current has not finished its bootstrapping phase
        n1, n2 = two_nodes
        n1.connect(n2.addr)
        wait_convergence([n1,n2], 1, only_direct=True)

        n1.set_start_learning(100, 100)

        time.sleep(1)  # Wait because of asincronity

        n1.set_stop_learning()

        wait_4_results([n1, n2])
"""

def test_connect_while_training(four_nodes):
    n1, n2, n3, n4 = four_nodes

    # Connect Nodes (unless the n4)
    n1.connect(n2.addr)
    n3.connect(n1.addr)
    wait_convergence([n1], 2, only_direct=True)
    wait_convergence([n2,n3], 1, only_direct=True)

    # Start Learning
    n1.set_start_learning(2, 1)
    time.sleep(4)

    # Try to connect
    assert n1.connect(n4.addr) == False
    assert n4.connect(n1.addr) == False
    time.sleep(1)
    assert len(n1.get_neighbors(only_direct=True)) == 2
    assert len(n4.get_neighbors(only_direct=True)) == 0

##############################
#    Fault Tolerace Tests    #
##############################

"""

@pytest.mark.parametrize("n", [2, 4])
def test_node_down_on_learning(n):

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(MLP(), MnistFederatedDM())
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes) - 1):
        nodes[i + 1].connect_to(nodes[i].host, nodes[i].port, full=True)
        time.sleep(1)

    # Check if they are connected
    for node in nodes:
        assert len(node.get_neighbors()) == n - 1

    # Start Learning
    nodes[0].set_start_learning(rounds=2, epochs=0)

    # Stopping node
    time.sleep(0.3)
    nodes[1].stop()

    wait_4_results(nodes)

    for node in nodes:
        node.stop()


def test_bad_binary_model(two_nodes):
    n1, n2 = two_nodes
    n1.connect_to(n2.host, n2.port, full=True)
    time.sleep(0.1)

    # Start Learning
    n1.set_start_learning(rounds=2, epochs=0)

    # Adding noise to the buffer
    for _ in range(200):
        n1.get_neighbors()[0]._NodeConnection__param_bufffer += "noise".encode("utf-8")

    wait_4_results([n1, n2])


def test_wrong_model():
    n1 = Node(MLP(), MnistFederatedDM())
    n2 = Node(CNN(), MnistFederatedDM())

    n1.start()
    n2.start()
    n1.connect_to(n2.host, n2.port)
    time.sleep(0.1)

    n1.set_start_learning(rounds=2, epochs=0)
    time.sleep(0.1)

    wait_4_results([n1, n2])

    n1.stop()
    n2.stop()


#############################
#    Node Encrypted Test    #
#############################


@pytest.mark.parametrize("x", [(2, 1), (2, 2)])
def test_encrypted_convergence(x):
    n, r = x

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(MLP(), MnistFederatedDM(), simulation=False)
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes) - 1):
        nodes[i + 1].connect_to(nodes[i].host, nodes[i].port, full=True)
        time.sleep(0.1)

    # Overhead at handshake because of the encryption
    time.sleep(3)

    # Check if they are connected
    time.sleep(3)
    for node in nodes:
        assert len(node.get_neighbors()) == n - 1

    # Start Learning
    nodes[0].set_start_learning(rounds=r, epochs=0)

    # Wait and check
    wait_4_results(nodes)
    check_equal_models(nodes)

    # Close
    [n.stop() for n in nodes]


#######################################
#    Non Full Connected Topologies    #
#######################################


def test_ring_network_learning(four_nodes):
    n1, n2, n3, n4 = four_nodes

    # Ring network
    n1.connect_to(n2.host, n2.port, full=False)
    n2.connect_to(n3.host, n3.port, full=False)
    n3.connect_to(n4.host, n4.port, full=False)
    n4.connect_to(n1.host, n1.port, full=False)

    wait_network_nodes(four_nodes)

    # Verify topology
    assert (
        len(n1.get_neighbors())
        == len(n2.get_neighbors())
        == len(n3.get_neighbors())
        == len(n4.get_neighbors())
        == 2
    )

    __test_learning((n1, n2, n3, n4))

    # Verify topology after learning
    assert (
        len(n1.get_neighbors())
        == len(n2.get_neighbors())
        == len(n3.get_neighbors())
        == len(n4.get_neighbors())
        == 2
    )

    # Stop Nodes
    [n.stop() for n in four_nodes]


def test_star_network_learning(four_nodes):
    n1, n2, n3, n4 = four_nodes

    # Star Network
    n1.connect_to(n4.host, n4.port, full=False)
    n2.connect_to(n4.host, n4.port, full=False)
    n3.connect_to(n4.host, n4.port, full=False)

    wait_network_nodes(four_nodes)

    # Verify topology after learning
    assert (
        len(n1.get_neighbors())
        == len(n2.get_neighbors())
        == len(n3.get_neighbors())
        == 1
    )
    assert len(n4.get_neighbors()) == 3

    __test_learning((n1, n2, n3, n4))

    # Verify topology after learning
    assert (
        len(n1.get_neighbors())
        == len(n2.get_neighbors())
        == len(n3.get_neighbors())
        == 1
    )
    assert len(n4.get_neighbors()) == 3

    # Stop Nodes
    [n.stop() for n in four_nodes]


def test_line_network_learning(four_nodes):
    n1, n2, n3, n4 = four_nodes

    # Star Network
    n1.connect_to(n2.host, n2.port, full=False)
    n2.connect_to(n3.host, n3.port, full=False)
    n3.connect_to(n4.host, n4.port, full=False)

    wait_network_nodes(four_nodes)

    # Verify topology
    assert len(n1.get_neighbors()) == len(n4.get_neighbors()) == 1
    assert len(n2.get_neighbors()) == len(n3.get_neighbors()) == 2

    __test_learning((n1, n2, n3, n4))

    # Verify topology after learning
    assert len(n1.get_neighbors()) == len(n4.get_neighbors()) == 1
    assert len(n2.get_neighbors()) == len(n3.get_neighbors()) == 2

    # Stop Nodes
    [n.stop() for n in four_nodes]


def __test_learning(nodes):
    n1, n2, n3, n4 = nodes
    n1.set_start_learning(rounds=2, epochs=0)
    time.sleep(1)
    wait_4_results(nodes)

    # Verify network nodes
    assert (
        len(n1.get_network_nodes())
        == len(n2.get_network_nodes())
        == len(n3.get_network_nodes())
        == len(n4.get_network_nodes())
        == 4
    )
"""
