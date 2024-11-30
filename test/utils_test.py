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
    with pytest.raises(TypeError) as excinfo:
        TopologyFactory.generate_matrix("invalid_type", 4)  # Pass a string, not enum
    assert str(excinfo.value) == "topology_type must be a TopologyType enum member"
