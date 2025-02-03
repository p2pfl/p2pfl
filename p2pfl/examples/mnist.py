#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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

"""Example of a P2PFL MNIST experiment, using a MLP model and a MnistFederatedDM."""

# poetry run snakeviz _MainThread-0.pstat
# poetry run gprof2dot -f pstats Gossiper-10.pstat | dot -Tpng -o output.png && open output.png

import argparse
import time
import uuid

import matplotlib.pyplot as plt

from p2pfl.communication.protocols.grpc.grpc_communication_protocol import GrpcCommunicationProtocol
from p2pfl.communication.protocols.memory.memory_communication_protocol import InMemoryCommunicationProtocol
from p2pfl.learning.aggregators.scaffold import Scaffold
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import wait_convergence, wait_to_finish


def set_standalone_settings(disable_ray: bool = False) -> None:
    """
    Set settings for testing.

    Important:
        - HEARTBEAT_PERIOD: Too high values can cause late node discovery/fault detection. Too low values can cause high CPU usage.
        - GOSSIP_PERIOD: Too low values can cause high CPU usage.
        - TTL: Low TTLs can cause that some messages are not delivered.

    """
    Settings.GRPC_TIMEOUT = 0.5
    Settings.HEARTBEAT_PERIOD = 5
    Settings.HEARTBEAT_TIMEOUT = 40
    Settings.GOSSIP_PERIOD = 1
    Settings.TTL = 40
    Settings.GOSSIP_MESSAGES_PER_PERIOD = 9999999999
    Settings.AMOUNT_LAST_MESSAGES_SAVED = 10000
    Settings.GOSSIP_MODELS_PERIOD = 1
    Settings.GOSSIP_MODELS_PER_ROUND = 4
    Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 10
    Settings.TRAIN_SET_SIZE = 4
    Settings.VOTE_TIMEOUT = 60
    Settings.AGGREGATION_TIMEOUT = 60
    Settings.WAIT_HEARTBEATS_CONVERGENCE = 0.2 * Settings.HEARTBEAT_TIMEOUT
    Settings.LOG_LEVEL = "INFO"
    Settings.EXCLUDE_BEAT_LOGS = True
    Settings.DISABLE_RAY = disable_ray
    logger.set_level(Settings.LOG_LEVEL)  # Refresh (maybe already initialized)


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2PFL MNIST experiment using the Web Logger.")
    parser.add_argument("--nodes", type=int, help="The number of nodes.", default=2)
    parser.add_argument("--rounds", type=int, help="The number of rounds.", default=2)
    parser.add_argument("--epochs", type=int, help="The number of epochs.", default=1)
    parser.add_argument("--show_metrics", action="store_true", help="Show metrics.", default=True)
    parser.add_argument("--measure_time", action="store_true", help="Measure time.", default=False)
    parser.add_argument("--token", type=str, help="The API token for the Web Logger.", default="")
    parser.add_argument("--protocol", type=str, help="The protocol to use.", default="grpc", choices=["grpc", "unix", "memory"])
    parser.add_argument("--framework", type=str, help="The framework to use.", default="pytorch", choices=["pytorch", "tensorflow", "flax"])
    parser.add_argument("--aggregator", type=str, help="The aggregator to use.", default="fedavg", choices=["fedavg", "scaffold"])
    parser.add_argument("--profiling", action="store_true", help="Enable profiling.", default=False)
    parser.add_argument("--reduced_dataset", action="store_true", help="Use a reduced dataset just for testing.", default=False)
    parser.add_argument("--use_scaffold", action="store_true", help="Use the Scaffold aggregator.", default=False)
    parser.add_argument("--disable_ray", action="store_true", help="Disable Ray.", default=False)
    parser.add_argument(
        "--topology",
        type=str,
        choices=[t.value for t in TopologyType],
        default="line",
        help="The network topology (star, full, line, ring).",
    )
    args = parser.parse_args()
    # parse topology to TopologyType enum
    args.topology = TopologyType(args.topology)

    return args


def create_tensorflow_model() -> P2PFLModel:
    """Create a TensorFlow model."""
    import tensorflow as tf  # type: ignore

    from p2pfl.learning.frameworks.tensorflow.keras_model import MLP as MLP_KERAS
    from p2pfl.learning.frameworks.tensorflow.keras_model import KerasModel

    model = MLP_KERAS()  # type: ignore[no-untyped-call]
    model(tf.zeros((1, 28, 28, 1)))
    return KerasModel(model)


def create_pytorch_model() -> P2PFLModel:
    """Create a PyTorch model."""
    from p2pfl.learning.frameworks.pytorch.lightning_model import MLP, LightningModel

    return LightningModel(MLP())  # type: ignore[no-untyped-call]


def mnist(
    n: int,
    r: int,
    e: int,
    show_metrics: bool = True,
    measure_time: bool = False,
    protocol: str = "grpc",
    framework: str = "pytorch",
    aggregator: str = "fedavg",
    reduced_dataset: bool = False,
    topology: TopologyType = TopologyType.LINE,
) -> None:
    """
    P2PFL MNIST experiment.

    Args:
        n: The number of nodes.
        r: The number of rounds.
        e: The number of epochs.
        show_metrics: Show metrics.
        measure_time: Measure time.
        protocol: The protocol to use.
        framework: The framework to use.
        aggregator: The aggregator to use.
        reduced_dataset: Use a reduced dataset just for testing.
        topology: The network topology (star, full, line, ring).

    """
    if measure_time:
        start_time = time.time()

    # Check settings
    if n > Settings.TTL:
        raise ValueError(
            "For in-line topology TTL must be greater than the number of nodes." "Otherwise, some messages will not be delivered."
        )

    # Imports
    if framework == "tensorflow":
        from p2pfl.learning.frameworks.tensorflow.keras_learner import KerasLearner

        model_fn = create_tensorflow_model
        learner = KerasLearner
    elif framework == "pytorch":
        from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner

        model_fn = create_pytorch_model
        learner = LightningLearner  # type: ignore
    else:
        raise ValueError(f"Framework {args.framework} not added on this example.")

    # Data
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(
        n * 50 if reduced_dataset else n,
        RandomIIDPartitionStrategy,  # type: ignore
    )

    # Node Creation
    nodes = []
    for i in range(n):
        address = f"node-{i}" if protocol == "memory" else f"unix:///tmp/p2pfl-{i}.sock" if protocol == "unix" else "127.0.0.1"

        # Nodes
        node = Node(
            model_fn(),
            partitions[i],
            learner=learner,  # type: ignore
            protocol=InMemoryCommunicationProtocol if protocol == "memory" else GrpcCommunicationProtocol,  # type: ignore
            address=address,
            simulation=True,
            aggregator=Scaffold() if aggregator == "scaffold" else None,
        )
        node.start()
        nodes.append(node)

    try:
        adjacency_matrix = TopologyFactory.generate_matrix(topology, len(nodes))
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)

        wait_convergence(nodes, n - 1, only_direct=False, wait=60)  # type: ignore

        if r < 1:
            raise ValueError("Skipping training, amount of round is less than 1")

        # Start Learning
        nodes[0].set_start_learning(rounds=r, epochs=e)

        # Wait and check
        wait_to_finish(nodes, timeout=60 * 60)  # 1 hour

        # Local Logs
        if show_metrics:
            local_logs = logger.get_local_logs()
            if local_logs != {}:
                logs_l = list(local_logs.items())[0][1]
                #  Plot experiment metrics
                for round_num, round_metrics in logs_l.items():
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
                logs_g = list(global_logs.items())[0][1]  # Accessing the nested dictionary directly
                # Plot experiment metrics
                for node_name, node_metrics in logs_g.items():
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
    except Exception as e:
        raise e
    finally:
        # Stop Nodes
        for node in nodes:
            node.stop()

        if measure_time:
            print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    # Parse args
    args = __parse_args()

    set_standalone_settings(disable_ray=args.disable_ray)  # todo: not working on the logger because it is imported at the top of the file

    if args.profiling:
        import os  # noqa: I001
        import yappi  # type: ignore

        # Start profiler
        yappi.start()

    # Set logger
    if args.token != "":
        logger.connect_web("http://localhost:3000/api/v1", args.token)

    # Launch experiment
    try:
        mnist(
            args.nodes,
            args.rounds,
            args.epochs,
            show_metrics=args.show_metrics,
            measure_time=args.measure_time,
            protocol=args.protocol,
            framework=args.framework,
            aggregator=args.aggregator,
            reduced_dataset=args.reduced_dataset,
            topology=args.topology,
        )
    finally:
        if args.profiling:
            # Stop profiler
            yappi.stop()
            # Save stats
            profile_dir = os.path.join("profile", "mnist", str(uuid.uuid4()))
            os.makedirs(profile_dir, exist_ok=True)
            for thread in yappi.get_thread_stats():
                yappi.get_func_stats(ctx_id=thread.id).save(f"{profile_dir}/{thread.name}-{thread.id}.pstat", type="pstat")
