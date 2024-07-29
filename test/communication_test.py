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
"""Communication tests."""

import time

import pytest

from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils import full_connection, set_test_settings, wait_convergence

set_test_settings()


@pytest.fixture
def two_nodes():
    """Create two nodes and start them. Yield the nodes. After the test, stop the nodes."""
    n1 = Node(None, None)
    n2 = Node(None, None)
    n1.start()
    n2.start()

    yield n1, n2

    n1.stop()
    n2.stop()


@pytest.fixture
def four_nodes():
    """Create four nodes and start them. Yield the nodes. After the test, stop the nodes."""
    n1 = Node(None, None)
    n2 = Node(None, None)
    n3 = Node(None, None)
    n4 = Node(None, None)
    n1.start()
    n2.start()
    n3.start()
    n4.start()

    return n1, n2, n3, n4


###########################
#  Tests Infraestructure  #
###########################


def test_connect_invalid_node():
    """Test that a node can't connect to an invalid node."""
    n = Node(None, None)
    n.start()
    n.connect("google.es:80")
    n.connect("holadani.holaenrique")
    assert len(n.get_neighbors()) == 0
    n.stop()


def test_basic_node_paring(two_nodes):
    """Test that two nodes can connect and disconnect."""
    n1, n2 = two_nodes

    # Connect
    n1.connect(n2.addr)
    wait_convergence(two_nodes, 1, only_direct=True)
    assert len(n1.get_neighbors(only_direct=True)) == len(n2.get_neighbors(only_direct=True)) == 1

    # Disconnect
    n2.disconnect(n1.addr)
    time.sleep((Settings.HEARTBEAT_PERIOD * 2) + 1)
    assert len(n1.get_neighbors(only_direct=True)) == len(n2.get_neighbors(only_direct=True)) == 0


def test_full_connected(four_nodes):
    """Test that four nodes can connect and disconnect (full connected)."""
    n1, n2, n3, n4 = four_nodes

    # Connect n1 n2
    n1.connect(n2.addr)
    wait_convergence([n1, n2], 1, only_direct=True)

    # Connect n3
    full_connection(n3, [n1, n2])
    wait_convergence([n1, n2, n3], 2, only_direct=True)

    # Connect n4
    full_connection(n4, [n1, n2, n3])
    wait_convergence(four_nodes, 3, only_direct=True)

    # Disconnect n1
    n1.stop()
    wait_convergence([n2, n3, n4], 2, only_direct=True)

    # Disconnect n2
    n2.stop()
    wait_convergence([n3, n4], 1, only_direct=True)

    # Disconnect n3
    n3.stop()
    wait_convergence([n4], 0, only_direct=True)

    # Disconnect n4
    n4.stop()


def test_network_neightbors(four_nodes):
    """Test that four nodes can connect and disconnect (non-full connected)."""
    n1, n2, n3, n4 = four_nodes

    # Connect n1 n2
    n1.connect(n2.addr)
    wait_convergence([n1, n2], 1, only_direct=False)

    # Connect n3 n1
    n3.connect(n1.addr)
    wait_convergence([n1, n2, n3], 2, only_direct=False)

    # Connect n4 n1
    n4.connect(n1.addr)
    wait_convergence(four_nodes, 3, only_direct=False)

    # Disconnect n4
    n4.stop()
    wait_convergence([n1, n2, n3], 2, only_direct=False)

    # Disconnect n3
    n3.stop()
    wait_convergence([n1, n2], 1, only_direct=False)

    # Disconnect n2
    n2.stop()
    wait_convergence([n1], 0, only_direct=False)

    # Disconnect n1
    n1.stop()


##############################
#    Test Fault Tolerance    #
##############################


def test_bad_msg(two_nodes):
    """Test that a node can't connect to an invalid node."""
    n1, n2 = two_nodes

    # Conexión
    n1.connect(n2.addr)
    time.sleep(0.1)

    # Create an error message
    n1._communication_protocol.broadcast(n1._communication_protocol.build_msg("BAD_MSG"))
    time.sleep(1)
    assert len(n1.get_neighbors()) == len(n2.get_neighbors()) == 0


def test_node_abrupt_down(four_nodes):
    """Test fault tolerance of a node."""
    n1, n2, n3, n4 = four_nodes

    # Connect n1 n2
    n1.connect(n2.addr)
    wait_convergence([n1, n2], 1, only_direct=True)

    # Connect n3
    full_connection(n3, [n1, n2])
    wait_convergence([n1, n2, n3], 2, only_direct=True)

    # Connect n4
    full_connection(n4, [n1, n2, n3])
    wait_convergence(four_nodes, 3, only_direct=True)

    # n1 stops heartbeater
    n1._communication_protocol._heartbeater.stop()
    wait_convergence([n2, n3, n4], 2, only_direct=True, wait=10)
    n1.stop()

    # n2 stop server
    n2._communication_protocol._server.stop()
    wait_convergence([n3, n4], 1, only_direct=True, wait=10)
    n2.stop()

    # Disconnect n3 n4
    n3.stop()
    n4.stop()
