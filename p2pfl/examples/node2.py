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

"""
Example of a P2PFL MNIST node using a MLP model and a MnistFederatedDM.

This node will be connected to node1 and then, the federated learning process will start.
"""

import argparse
import time

from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import (
    MnistFederatedDM,
)
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node

#from p2pfl.utils import set_test_settings

#set_test_settings()


def __get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2PFL MNIST node using a MLP model and a MnistFederatedDM.")
    parser.add_argument("--port", type=int, help="The port to connect.", required=True)
    return parser.parse_args()


def node2(port: int) -> None:
    """
    Start a node2, connects to another node, start and waits the federated learning process to finish.

    Args:
        port: The port to connect.

    """
    node = Node(MLP(), MnistFederatedDM(sub_id=1, number_sub=2), address="127.0.0.1")
    node.start()

    node.connect(f"127.0.0.1:{port}")
    time.sleep(4)

    node.set_start_learning(rounds=2, epochs=1)

    # Wait 4 results

    while True:
        time.sleep(1)

        if node.state.round is None:
            break

    node.stop()


if __name__ == "__main__":
    # Get arguments
    args = __get_args()

    # Run node2
    node2(args.port)
