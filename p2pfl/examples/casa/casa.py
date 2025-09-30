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

"""Example IoT casa dataset for Human Daily Activity Recognition (HDAR)."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tqdm

from p2pfl.communication.protocols.protobuff.grpc import GrpcCommunicationProtocol
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.examples.casa.model.lstm_tensorflow import model_build_fn  # type: ignore
from p2pfl.examples.casa.transforms import get_casa_transforms  # Import the transforms
from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.aggregators.fedmedian import FedMedian
from p2pfl.learning.aggregators.fedopt.fedadagrad import FedAdagrad
from p2pfl.learning.aggregators.fedopt.fedadam import FedAdam
from p2pfl.learning.aggregators.fedopt.fedyogi import FedYogi
from p2pfl.learning.aggregators.fedprox import FedProx
from p2pfl.learning.aggregators.krum import Krum
from p2pfl.learning.aggregators.scaffold import Scaffold
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import set_standalone_settings, wait_convergence, wait_to_finish


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2PFL CASA experiment using the Web Logger.")
    parser.add_argument("--nodes", type=int, help="The number of nodes.", default=2)
    parser.add_argument("--rounds", type=int, help="The number of rounds.", default=2)
    parser.add_argument("--epochs", type=int, help="The number of epochs.", default=1)
    parser.add_argument("--show_metrics", action="store_true", help="Show metrics.", default=True)
    parser.add_argument("--measure_time", action="store_true", help="Measure time.", default=False)
    parser.add_argument("--protocol", type=str, help="The protocol to use.", default="grpc", choices=["grpc", "unix", "memory"])
    parser.add_argument("--seed", type=int, help="The seed to use.", default=666)
    parser.add_argument("--batch_size", type=int, help="The batch size for training.", default=128)
    parser.add_argument(
        "--topology",
        type=str,
        choices=[t.value for t in TopologyType],
        default="line",
        help="The network topology (star, full, line, , random3).",
    )
    parser.add_argument("--save_csv", action="store_true", help="Save results to CSV files.", default=True)
    parser.add_argument("--output_dir", type=str, help="Directory to save CSV results.", default="results/casa")
    parser.add_argument(
        "--aggregator",
        type=str,
        choices=["fedavg", "fedprox", "fedmedian", "krum", "scaffold", "fedadam", "fedadagrad", "fedyogi"],
        default="fedavg",
        help="The aggregation algorithm to use.",
    )
    args = parser.parse_args()
    # parse topology to TopologyType enum
    args.topology = TopologyType(args.topology)

    return args


def save_experiment_results(output_dir: Path, start_time: float | None = None) -> None:
    """
    Save experiment results to CSV files.

    Args:
        output_dir: Directory to save results
        start_time: Start time of the experiment for execution time calculation

    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save message logs
    all_msgs = logger.get_messages(direction="all")
    if all_msgs:
        try:
            pandas_msgs = pd.DataFrame(all_msgs)
            msg_csv_path = output_dir / "messages.csv"
            pandas_msgs.to_csv(msg_csv_path, index=False)
            print(f"Saved messages log to: {msg_csv_path}")
        except Exception as e:
            print(f"Error saving messages log: {e}")

    # Save global metrics
    global_metrics_data = logger.get_global_logs()
    if global_metrics_data:
        flattened_global_metrics = []
        try:
            for exp, nodes in global_metrics_data.items():
                for node, metrics in nodes.items():
                    for metric_name, values in metrics.items():
                        for round_num, value in values:
                            flattened_global_metrics.append(
                                {"experiment": exp, "node": node, "metric": metric_name, "round": round_num, "value": value}
                            )

            if flattened_global_metrics:
                pandas_global_metrics = pd.DataFrame(flattened_global_metrics)
                global_metrics_csv_path = output_dir / "global_metrics.csv"
                pandas_global_metrics.to_csv(global_metrics_csv_path, index=False)
                print(f"Saved global metrics log to: {global_metrics_csv_path}")
        except Exception as e:
            print(f"Error saving global metrics: {e}")

    # Save system metrics
    system_metrics_data = logger.get_system_metrics()
    if system_metrics_data:
        flattened_system_metrics = []
        try:
            for timestamp, sys_metrics in system_metrics_data.items():
                for sys_metric_name, sys_value in sys_metrics.items():
                    flattened_system_metrics.append(
                        {"timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"), "metric_name": sys_metric_name, "metric_value": sys_value}
                    )

            if flattened_system_metrics:
                pandas_system_metrics = pd.DataFrame(flattened_system_metrics)
                system_metrics_csv_path = output_dir / "system_resources.csv"
                pandas_system_metrics.to_csv(system_metrics_csv_path, index=False)
                print(f"Saved system resource metrics log to: {system_metrics_csv_path}")
        except Exception as e:
            print(f"Error saving system resource metrics: {e}")

    # Save execution time if start_time is provided
    if start_time is not None:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal execution time: {execution_time:.4f} seconds")

        time_csv_path = output_dir / "execution_time.csv"
        try:
            time_df = pd.DataFrame({"Execution Time (s)": [f"{execution_time:.4f}"]})
            time_df.to_csv(time_csv_path, index=False)
            print(f"Saved execution time to: {time_csv_path}")
        except Exception as e:
            print(f"Error saving execution time: {e}")


def casa(
    n: int,
    r: int,
    e: int,
    show_metrics: bool = True,
    measure_time: bool = False,
    protocol: str = "grpc",
    topology: TopologyType = TopologyType.LINE,
    batch_size: int = 128,
    save_csv: bool = False,
    output_dir: str = "results/casa",
    aggregator: str = "fedavg",
) -> None:
    """
    P2PFL CASA experiment.

    Args:
        n: The number of nodes.
        r: The number of rounds.
        e: The number of epochs.
        show_metrics: Show metrics.
        measure_time: Measure time.
        protocol: The protocol to use.
        topology: The network topology (star, full, line, ring).
        batch_size: The batch size for training.
        save_csv: Save results to CSV files.
        output_dir: Directory to save CSV results.
        aggregator: The aggregation algorithm to use.

    """
    if measure_time:
        start_time = time.time()

    # Check settings
    if n > Settings.gossip.TTL and topology == TopologyType.LINE:
        raise ValueError(
            "For in-line topology TTL must be greater than the number of nodes. Otherwise, some messages will not be delivered."
        )

    # Select aggregator
    aggregator_map = {
        "fedavg": FedAvg,
        "fedprox": FedProx,
        "fedmedian": FedMedian,
        "krum": Krum,
        "scaffold": Scaffold,
        "fedadam": FedAdam,
        "fedadagrad": FedAdagrad,
        "fedyogi": FedYogi,
    }

    aggregator_class = aggregator_map.get(aggregator.lower(), FedAvg)
    print(f"Using aggregator: {aggregator_class.__name__}")

    # Data creation
    print("Creating data for each node...")
    node_data = []
    for i in tqdm.tqdm(range(n)):
        # Data
        data_dir = None if n == 1 else f"casa{i + 1}"  # Use different data directories for each node if more than one node
        data = P2PFLDataset.from_huggingface("p2pfl/casa", data_dir=data_dir)
        data.set_batch_size(batch_size)
        data.set_transforms(get_casa_transforms())  # Apply the transforms to format the data
        node_data.append(data)

    # Node Creation
    nodes = []
    for i in range(n):
        address = f"node-{i}" if protocol == "memory" else f"unix:///tmp/p2pfl-{i}.sock" if protocol == "unix" else "127.0.0.1"

        # Nodes
        node = Node(
            model_build_fn(),
            node_data[i],
            aggregator=aggregator_class(),
            protocol=MemoryCommunicationProtocol() if protocol == "memory" else GrpcCommunicationProtocol(),
            addr=address,
        )
        node.start()
        nodes.append(node)

    try:
        adjacency_matrix = TopologyFactory.generate_matrix(topology, len(nodes))
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)

        wait_convergence(nodes, n - 1, only_direct=False, wait=160, debug=True)  # type: ignore

        if r < 1:
            raise ValueError("Skipping training, amount of round is less than 1")

        # Start Learning
        nodes[0].set_start_learning(rounds=r, epochs=e, trainset_size=n // 2)

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
                            x, y = zip(*values, strict=False)
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
                        x, y = zip(*values, strict=False)
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

        # Save CSV results if requested
        if save_csv:
            output_path = Path(output_dir)
            save_experiment_results(output_path, start_time if measure_time else None)


if __name__ == "__main__":
    # Parse args
    args = __parse_args()

    set_standalone_settings()
    Settings.training.RAY_ACTOR_POOL_SIZE = 4
    Settings.heartbeat.TIMEOUT = 120
    Settings.gossip.TTL = args.nodes  # ensure all messages arrive
    Settings.training.AGGREGATION_TIMEOUT = 300

    # Seed
    if args.seed is not None:
        Settings.general.SEED = args.seed

    # Launch experiment
    casa(
        args.nodes,
        args.rounds,
        args.epochs,
        show_metrics=args.show_metrics,
        measure_time=args.measure_time,
        protocol=args.protocol,
        topology=args.topology,
        batch_size=args.batch_size,
        save_csv=args.save_csv,
        output_dir=args.output_dir,
        aggregator=args.aggregator,
    )
