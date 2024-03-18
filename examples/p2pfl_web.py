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

from p2pfl.management.logger import logger
from p2pfl.management.p2pfl_web_services import P2pflWebServices
from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import (
    MnistFederatedDM,
)
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
import sys

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: python3 p2pfl_web.py")
        sys.exit(1)

    # Set the logger
    logger.init(P2pflWebServices("https://b3e4b102-69aa-42b0-9d2d-ca48e7d553f2.mock.pstmn.io", "1234"))
    
    # Node Creation
    node = Node(
        MLP(),
        MnistFederatedDM(sub_id=0, number_sub=2)
    )
    node.start()

    input("Press any key to stop\n")

    node.stop()
