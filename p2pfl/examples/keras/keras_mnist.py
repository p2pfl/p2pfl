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

import argparse
import time

import matplotlib.pyplot as plt

# from p2pfl.learning.pytorch.mnist_examples.mnistfederated_dm import (
#     MnistFederatedDM,
# )
import tensorflow as tf

from p2pfl.communication.grpc.grpc_communication_protocol import GrpcCommunicationProtocol
from p2pfl.communication.memory.memory_communication_protocol import InMemoryCommunicationProtocol
from p2pfl.learning.tensorflow.keras_learner import KerasLearner
from p2pfl.learning.tensorflow.mnist_examples.keras_mnist_dataset import MnistFederatedDM

# from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.learning.tensorflow.mnist_examples.keras_sample_model import MLP
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.utils import (
    set_test_settings,
    wait_4_results,
    wait_convergence,
)

"""
Example of a P2PFL MNIST experiment, using a MLP model and a MnistFederatedDM.
"""


def __parse_args():
    parser = argparse.ArgumentParser(description="P2PFL MNIST experiment using the Web Logger.")
    parser.add_argument("--nodes", type=int, help="The number of nodes.", default=2)
    parser.add_argument("--rounds", type=int, help="The number of rounds.", default=2)
    parser.add_argument("--epochs", type=int, help="The number of epochs.", default=2)
    parser.add_argument("--show_metrics", action="store_true", help="Show metrics.", default=True)
    parser.add_argument("--measure_time", action="store_true", help="Measure time.", default=False)
    parser.add_argument("--use_unix_socket", action="store_true", help="Use Unix socket.", default=False)
    parser.add_argument("--use_local_protocol", action="store_true", help="Use local protocol.", default=False)
    parser.add_argument("--token", type=str, help="The API token for the Web Logger.", default="")

    # check (cannot use the unix socket and the local protocol at the same time)
    args = parser.parse_args()

    if args.use_unix_socket and args.use_local_protocol:
        parser.error("Cannot use the unix socket and the local protocol at the same time.")

    return args


def mnist(n, r, e, show_metrics=True, measure_time=False, use_unix_socket=False, use_local_protocol=False):
    if measure_time:
        start_time = time.time()
    tf.config.experimental.list_physical_devices()
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Node Creation
    nodes = []
    for i in range(n):
        if use_local_protocol:
            address = f"node-{i}"
        else:
            address = f"unix:///tmp/p2pfl-{i}.sock" if use_unix_socket else "127.0.0.1"

        data_module = MnistFederatedDM(sub_id=i, number_sub=n)
        train_dataset, val_dataset, _ = data_module.get_data()

        # need a way to pass parameters to the learner directly?
        node = Node(
            MLP(),
            (train_dataset, val_dataset),
            learner=KerasLearner,
            protocol=InMemoryCommunicationProtocol if use_local_protocol else GrpcCommunicationProtocol,
            address=address,
        )

        node.start()
        nodes.append(node)

    # Node Connection
    for i in range(len(nodes) - 1):
        nodes[i + 1].connect(nodes[i].addr)
        time.sleep(0.1)
    wait_convergence(nodes, n - 1, only_direct=False)

    # Start Learning
    nodes[0].set_start_learning(rounds=r, epochs=e)

    # Wait and check
    wait_4_results(nodes)

    # Local Logs
    if show_metrics:
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

    if measure_time:
        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    # Parse args
    args = __parse_args()

    # Set logger
    if args.token != "":
        logger.connect_web("http://localhost:3000/api/v1", args.token)

    # Settings
    set_test_settings()

    # Launch experiment
    mnist(
        args.nodes,
        args.rounds,
        args.epochs,
        show_metrics=args.show_metrics,
        measure_time=args.measure_time,
        use_unix_socket=args.use_unix_socket,
        use_local_protocol=args.use_local_protocol,
    )
