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
from typing import List

from p2pfl.node import Node


class BaseTopology:
    """Base class for network topologies."""

    def __init__(self, nodes: List[Node]):
        """
        Initialize the network topology.

        Args:
            nodes: List of nodes in the network.

        """
        self.nodes = nodes
        self.connect()

    def connect(self):
        """
        Generate the network topology.

        Returns:
            List of nodes in the network.

        """
        raise NotImplementedError

class InLineTopology(BaseTopology):
    """In-line network topology."""

    def connect(self):
        """Generate the in-line network topology."""
        for i in range(len(self.nodes) - 1):
            self.nodes[i + 1].connect(self.nodes[i].addr)
            time.sleep(0.1)

class RingTopology(BaseTopology):
    """Ring network topology."""

    def connect(self):
        """Generate the ring network topology."""
        for i in range(len(self.nodes)):
            self.nodes[i].connect(self.nodes[(i + 1) % len(self.nodes)].addr)
            time.sleep(0.1)

class FullyConnectedTopology(BaseTopology):
    """Fully connected network topology."""

    def connect(self):
        """Generate the fully connected network topology."""
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                self.nodes[i].connect(self.nodes[j].addr)
                time.sleep(0.1)

class StarTopology(BaseTopology):
    """Star network topology."""

    def __init__(self, nodes, center_node_idx):
        """
        Generate the star network topology.

        Args:
            nodes: List of nodes in the network.
            center_node_idx: Index of the center node.

        """
        self.center_node = center_node_idx
        super().__init__(nodes)

    def connect(self):
        """Generate the star network topology."""
        for i in range(len(self.nodes)):
            if i != self.center_node:
                self.nodes[i].connect(self.nodes[self.center_node].addr)
                time.sleep(0.1)
