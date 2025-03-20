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

"""Network topologies for the p2pfl package."""

import time
from enum import Enum
from typing import List

import numpy as np

from p2pfl.node import Node


class TopologyType(Enum):
    """Enumeration of supported network topologies."""

    STAR = "star"
    FULL = "full"
    LINE = "line"
    RING = "ring"


class TopologyFactory:
    """Factory class for generating network topologies."""

    @staticmethod
    def generate_matrix(topology_type: TopologyType | str, num_nodes: int) -> np.ndarray:
        """
        Generate the adjacency matrix for the specified topology.

        Args:
            topology_type: The type of topology to generate.
            num_nodes: The number of nodes in the network.

        """
        if isinstance(topology_type, str):
            topology_type = TopologyType(topology_type)

        matrix = np.zeros((num_nodes, num_nodes), dtype=int)  # Initialize as NumPy array

        if topology_type == TopologyType.STAR:
            matrix[0, 1:] = 1
            matrix[1:, 0] = 1
        elif topology_type == TopologyType.FULL:
            matrix[:] = 1  # Set all to 1
            np.fill_diagonal(matrix, 0)  # Set diagonal to 0
        elif topology_type == TopologyType.LINE:
            for i in range(num_nodes - 1):
                matrix[i, i + 1] = 1
                matrix[i + 1, i] = 1
        elif topology_type == TopologyType.RING:
            for i in range(num_nodes):
                matrix[i, (i + 1) % num_nodes] = 1
                matrix[(i + 1) % num_nodes, i] = 1
        else:
            raise ValueError(f"Unsupported topology type: {topology_type}")

        return matrix

    @staticmethod
    def connect_nodes(adjacency_matrix: np.ndarray, nodes: List[Node]):
        """
        Connect nodes based on the adjacency matrix.

        Args:
            adjacency_matrix: The adjacency matrix of the network.
            nodes: The list of nodes in the network.

        """
        num_nodes = len(nodes)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adjacency_matrix[i, j] == 1:
                    try:
                        nodes[i].connect(nodes[j].addr)
                        print(f"Connected nodes {nodes[i].addr} and {nodes[j].addr}")
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"Error connecting nodes {nodes[i].addr} and {nodes[j].addr}: {e}")
