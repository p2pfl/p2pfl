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

import sys

from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import (
    MnistFederatedDM,
)
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import argparse

"""
Example of a P2PFL MNIST node using a MLP model and a MnistFederatedDM. 
This node only starts, create a node2 and connect to it in order to start the federated learning process.
"""

def __get_args():
    parser = argparse.ArgumentParser(description="P2PFL MNIST node using a MLP model and a MnistFederatedDM.")
    parser.add_argument("port", type=int, help="The port.")
    return parser.parse_args()

def node1(port):
    node = Node(
        MLP(),
        MnistFederatedDM(sub_id=0, number_sub=2),
        port=port
    )
    node.start()

    input("Press any key to stop\n")

    node.stop()

if __name__ == "__main__":
    args = __get_args()
    node1(args.port)
