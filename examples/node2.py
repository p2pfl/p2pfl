# 
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/federated_learning_p2p).
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

from p2pfl.node import Node
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
import time
import sys

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "Usage: python3 nodo2.py <self_port> <other_node_port>"
        )
        sys.exit(1)

    node = Node(
        MLP(),
        MnistFederatedDM(sub_id=1, number_sub=2),
        port=int(sys.argv[1]),
    )
    node.start()

    node.connect(f"127.0.0.1:{sys.argv[2]}")
    time.sleep(4)

    node.set_start_learning(rounds=2, epochs=1)

    # Wait 4 results

    while True:
        time.sleep(1)

        if node.round is None:
            break

    node.stop()