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
"""Utils tests."""

from unittest.mock import MagicMock, call

import numpy as np
import pytest

from p2pfl.utils.node_component import NodeComponent, allow_no_addr_check
from p2pfl.utils.topologies import TopologyFactory, TopologyType


# Mock Node class for testing
class MockNode:
    """Mock Node class for testing."""

    def __init__(self, addr):
        """Initialize the mock node."""
        self.addr = addr
        self.connect = MagicMock()


@pytest.mark.parametrize(
    "topology_type, expected_matrix",
    [
        (TopologyType.STAR, np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])),
        (TopologyType.FULL, np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])),
        (TopologyType.LINE, np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])),
        (TopologyType.RING, np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])),
    ],
)
def test_generate_matrix(topology_type, expected_matrix):
    """Test the generation of adjacency matrices for different topologies."""
    num_nodes = 4
    matrix = TopologyFactory.generate_matrix(topology_type, num_nodes)
    assert np.array_equal(matrix, expected_matrix)


@pytest.mark.parametrize(
    "adjacency_matrix, expected_calls",
    [
        (
            np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),  # Star topology matrix
            [
                [call("address_1"), call("address_2"), call("address_3")],
                [],
                [],
                [],
            ],
        ),
        (
            np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]),  # Full topology matrix
            [
                [call("address_1"), call("address_2"), call("address_3")],
                [call("address_2"), call("address_3")],
                [call("address_3")],
                [],
            ],
        ),
    ],
)
def test_connect_nodes(adjacency_matrix, expected_calls):
    """Test that nodes are connected according to the provided adjacency matrix."""
    num_nodes = adjacency_matrix.shape[0]  # Get num_nodes from matrix shape
    nodes = [MockNode(f"address_{i}") for i in range(num_nodes)]
    TopologyFactory.connect_nodes(adjacency_matrix, nodes)

    for i, calls in enumerate(expected_calls):
        nodes[i].connect.assert_has_calls(calls, any_order=True)
        assert nodes[i].connect.call_count == len(calls)


def test_invalid_topology_type():
    """Test that an exception is raised when an invalid topology type is passed."""
    with pytest.raises(ValueError):
        TopologyFactory.generate_matrix("invalid_type", 4)


@pytest.mark.parametrize(
    "topology_type, num_nodes",
    [
        (TopologyType.RANDOM_2, 0),
        (TopologyType.RANDOM_2, 1),
        (TopologyType.RANDOM_2, 2),
        (TopologyType.RANDOM_2, 10),
        (TopologyType.RANDOM_3, 0),
        (TopologyType.RANDOM_3, 1),
        (TopologyType.RANDOM_3, 2),
        (TopologyType.RANDOM_3, 10),
        (TopologyType.RANDOM_4, 0),
        (TopologyType.RANDOM_4, 1),
        (TopologyType.RANDOM_4, 2),
        (TopologyType.RANDOM_4, 10),
    ],
)
def test_generate_random_matrix_properties(topology_type, num_nodes):
    """Test properties of randomly generated adjacency matrices."""
    matrix = TopologyFactory.generate_matrix(topology_type, num_nodes)

    # Check shape
    assert matrix.shape == (num_nodes, num_nodes)

    # Check diagonal is zero
    assert np.all(np.diag(matrix) == 0)

    # Check symmetry
    assert np.array_equal(matrix, matrix.T)

    # Check number of edges
    if num_nodes <= 1:
        expected_num_edges = 0
    else:
        if topology_type == TopologyType.RANDOM_2:
            avg_degree = 2
        elif topology_type == TopologyType.RANDOM_3:
            avg_degree = 3
        else:  # RANDOM_4
            avg_degree = 4

        # Calculate expected number of edges based on implementation logic
        num_edges_target = round(num_nodes * avg_degree / 2)
        max_possible_edges = num_nodes * (num_nodes - 1) // 2
        expected_num_edges = min(num_edges_target, max_possible_edges)

    actual_num_edges = np.sum(matrix) // 2
    assert actual_num_edges == expected_num_edges


class MockNodeComponent(NodeComponent):
    """Mock class inheriting from NodeComponent for testing."""

    def __init__(self):
        """Initialize the mock class."""
        # super init
        super().__init__()

    def example_method(self) -> str:
        """Return the address. Example method that requires addr to be set."""
        return self.addr

    @allow_no_addr_check
    def get_default_name(self) -> str:
        """Return "Hola!". A method with no addr check."""
        return "Hola!"


def test_node_component_initialization():
    """Test initial state and setting of addr."""
    component = MockNodeComponent()
    assert component.addr == ""

    addr = "test_address"
    returned_addr = component.set_addr(addr)
    assert component.addr == addr
    assert returned_addr == addr


def test_node_component_methods():
    """Test method calls with and without addr set."""
    component = MockNodeComponent()
    assert component.get_default_name() == "Hola!"

    # Method call without addr should raise ValueError
    with pytest.raises(ValueError):
        component.example_method()

    # Method call with addr set should succeed
    addr = "test_address"
    component.set_addr(addr)
    assert component.example_method() == addr
