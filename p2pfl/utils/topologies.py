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

import numpy as np

from p2pfl.management.logger import logger
from p2pfl.node import Node


class TopologyType(Enum):
    """Enumeration of supported network topologies."""

    STAR = "star"
    FULL = "full"
    LINE = "line"
    RING = "ring"
    RANDOM_2 = "random_2"  # Random graph with average degree 2
    RANDOM_3 = "random_3"  # Random graph with average degree 3
    RANDOM_4 = "random_4"  # Random graph with average degree 4


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
        elif topology_type in [TopologyType.RANDOM_2, TopologyType.RANDOM_3, TopologyType.RANDOM_4]:
            # Erdős–Rényi G(n, M) model: select M edges randomly
            if num_nodes <= 1:
                return matrix  # No edges possible for 0 or 1 node

            if topology_type == TopologyType.RANDOM_2:
                avg_degree = 2
            elif topology_type == TopologyType.RANDOM_3:
                avg_degree = 3
            else:  # RANDOM_4
                avg_degree = 4

            # Calculate target number of edges M = (n * avg_degree) / 2
            num_edges_target = round(num_nodes * avg_degree / 2)

            rng = np.random.default_rng()
            possible_edges = []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    possible_edges.append((i, j))

            max_possible_edges = len(possible_edges)
            if max_possible_edges == 0:
                return matrix  # Should not happen if num_nodes > 1 but good practice

            # Ensure we don't try to select more edges than possible
            num_edges_to_select = min(num_edges_target, max_possible_edges)

            if num_edges_to_select > 0:
                selected_indices = rng.choice(max_possible_edges, size=num_edges_to_select, replace=False)
                for index in selected_indices:
                    i, j = possible_edges[index]
                    matrix[i, j] = 1
                    matrix[j, i] = 1
        else:
            raise ValueError(f"Unsupported topology type: {topology_type}")

        return matrix

    @staticmethod
    def connect_nodes(adjacency_matrix: np.ndarray, nodes: list[Node]):
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
                        logger.info("", f"Connected nodes {nodes[i].addr} and {nodes[j].addr}")
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error("", f"Error connecting nodes {nodes[i].addr} and {nodes[j].addr}: {e}")
