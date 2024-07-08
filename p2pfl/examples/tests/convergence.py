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
import time
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import (
    MnistFederatedDM,
)
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
from p2pfl.utils import (
    set_test_settings,
    wait_4_results,
    wait_convergence,
    check_equal_models,
)

set_test_settings()

###########################
#  Tests Infraestructure  #
###########################


def main():
    n = 2
    nodes = []
    for i in range(n):
        node = Node(MLP(), MnistFederatedDM())
        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes) - 1):
        nodes[i + 1].connect(nodes[i].addr)
        time.sleep(0.1)
    wait_convergence(nodes, n - 1, only_direct=False)

    # Start Learning
    nodes[0].set_start_learning(rounds=2, epochs=0)

    # Stopping node
    time.sleep(0.1)
    nodes[-1].stop()

    wait_4_results(nodes)

    for node in nodes[:-1]:
        node.stop()


if __name__ == "__main__":
    main()
