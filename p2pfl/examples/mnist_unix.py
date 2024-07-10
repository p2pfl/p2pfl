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

import matplotlib.pyplot as plt

from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import (
    MnistFederatedDM,
)
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.utils import (
    set_test_settings,
    wait_4_results,
    wait_convergence,
)


def wait_convergence(nodes, n_neis, wait=5, only_direct=False):
    acum = 0
    while True:
        begin = time.time()
        if all([len(n.get_neighbors(only_direct=only_direct)) == n_neis for n in nodes]):
            break
        time.sleep(0.1)
        acum += time.time() - begin
        if acum > wait:
            assert False


def test_convergence(n, r, epochs=2):
    start_time = time.time()

    # Node Creation
    nodes = []
    for i in range(n):
        node = Node(
            MLP(),
            MnistFederatedDM(sub_id=0, number_sub=20),  # sampling for increase speed
            address=f"unix:///tmp/test{i}.sock",
        )
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

    # Local Logs
    local_logs = logger.get_local_logs()
    if local_logs != {}:
        logs = list(local_logs.items())[0][1]
        #  Plot experiment metrics
        for round_num, round_metrics in logs.items():
            for node_name, node_metrics in round_metrics.items():
                for metric, values in node_metrics.items():
                    x, y = zip(*values)
                    plt.plot(x, y, label=metric)
                    # Add a red point to the last data point
                    plt.scatter(x[-1], y[-1], color="red")
                    plt.title(f"Round {round_num} - {node_name}")
                    plt.xlabel("Epoch")
                    plt.ylabel(metric)
                    plt.legend()
                    plt.show()

    # Global Logs
    global_logs = logger.get_global_logs()
    if global_logs != {}:
        logs = list(global_logs.items())[0][1]  # Accessing the nested dictionary directly
        # Plot experiment metrics
        for node_name, node_metrics in logs.items():
            for metric, values in node_metrics.items():
                x, y = zip(*values)
                plt.plot(x, y, label=metric)
                # Add a red point to the last data point
                plt.scatter(x[-1], y[-1], color="red")
                plt.title(f"{node_name} - {metric}")
                plt.xlabel("Epoch")
                plt.ylabel(metric)
                plt.legend()
                plt.show()

    # Stop Nodes
    [n.stop() for n in nodes]

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    # Settings
    set_test_settings()
    # Launch experiment
    test_convergence(10, 10, epochs=0)
