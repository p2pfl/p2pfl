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

from p2pfl.node import Node
import time
from p2pfl.utils import (
    check_equal_models,
    set_test_settings,
    wait_4_results,
    wait_convergence,
)
set_test_settings()
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import (
    MnistFederatedDM,
)
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
import time

def wait_convergence(nodes, n_neis, wait=5, only_direct=False):
    acum = 0
    while True:
        begin = time.time()
        if all(
            [len(n.get_neighbors(only_direct=only_direct)) == n_neis for n in nodes]
        ):
            break
        time.sleep(0.1)
        acum += time.time() - begin
        if acum > wait:
            assert False

###########################
#  Tests Infraestructure  #
###########################

def main():
    n = 4
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
    #time.sleep(0.3)
    #print("Stopping node")
    #nodes[-1].stop()

    wait_4_results(nodes)

    for node in nodes[:-1]:
        node.stop()

    [n.stop() for n in nodes]

if __name__ == "__main__":
    main()
