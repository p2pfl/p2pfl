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

This node only starts, create a node2 and connect to it in order to start the federated learning process.
"""

import argparse

from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import (
    MnistFederatedDM,
)
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node

#from p2pfl.utils import set_test_settings

#set_test_settings()

def __get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2PFL MNIST node using a MLP model and a MnistFederatedDM.")
    parser.add_argument("--port", type=int, help="The port.", required=True)
    return parser.parse_args()


def node1(port: int) -> None:
    """
    Start a node1 and waits for a key press to stop it.

    Args:
        port: The port where the node will be listening.

    """
    print(f"127.0.0.1:{port}")
    node = Node(MLP(), MnistFederatedDM(sub_id=0, number_sub=2), address=f"127.0.0.1:{port}")
    node.start()

    input("Press any key to stop\n")

    node.stop()


if __name__ == "__main__":
    args = __get_args()
    node1(args.port)
