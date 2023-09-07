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

from test.utils import (
    wait_convergence,
    set_test_settings,
    wait_4_results,
)
from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import MnistFederatedDM
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.learning.pytorch.mnist_examples.models.cnn import CNN
from p2pfl.node import Node
import time
import matplotlib.pyplot as plt


def test_convergence(n, r, epochs=2):
    # Node Creation
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
    nodes[0].set_start_learning(rounds=r, epochs=epochs)

    # Wait and check
    wait_4_results(nodes)

    # Get logs
    logs = nodes[0].learner.get_logs()[0]
    for node_name, node_metrics in logs.items():
        for metric, values in node_metrics.items():
            v = zip(*values)
            plt.plot(*v)
            plt.title(f"{node_name} - {metric}")
            plt.show()

    # Stop Nodes
    [n.stop() for n in nodes]


if __name__ == "__main__":
    # Settings
    set_test_settings()
    # Launch experiment
    test_convergence(2, 2, epochs=0)
