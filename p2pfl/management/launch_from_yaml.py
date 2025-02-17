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
"""Launch from YAMLs."""

import importlib
from typing import Any

import yaml

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory
from p2pfl.utils.utils import wait_convergence, wait_to_finish


def load_by_package_and_name(package_name, class_name) -> Any:
    """
    Load a class by package and name.

    Args:
        package_name: The package name.
        class_name: The class name.

    """
    module = importlib.import_module(package_name)
    return getattr(module, class_name)


def run_from_yaml(yaml_path: str):
    """
    Run a simulation from a YAML file.

    Args:
        yaml_path: The path to the YAML file.

    """
    # Parse YAML configuration
    with open(yaml_path) as file:
        config = yaml.safe_load(file)

    # Update settings
    custom_settings = config.get("settings", {})
    if custom_settings:
        Settings.set_from_dict(custom_settings)
        # Refresh (already initialized)
        logger.set_level(Settings.general.LOG_LEVEL)

    # Get Amount of Nodes
    network_config = config.get("network", {})
    if not network_config:
        raise ValueError("Missing 'network' configuration in YAML file.")
    n = network_config.get("nodes")
    if not n:
        raise ValueError("Missing 'n' under 'network' configuration in YAML file.")

    ###########
    # Dataset #
    ###########

    experiment_config = config.get("experiment", {})
    dataset_config = experiment_config.get("dataset", {})  # Get dataset config
    if not dataset_config:
        raise ValueError("Missing 'dataset' configuration in YAML file.")
    data_source = dataset_config.get("source")
    if not data_source:
        raise ValueError("Missing 'source' under 'dataset' configuration in YAML file.")
    dataset_name = dataset_config.get("name")
    if not dataset_name:
        raise ValueError("Dataset source is 'huggingface' but 'name' is missing in YAML.")

    # Load data
    dataset = None
    if data_source == "huggingface":
        dataset = P2PFLDataset.from_huggingface(dataset_name)
    elif data_source == "csv":
        dataset = P2PFLDataset.from_csv(dataset_name)
    elif data_source == "json":
        dataset = P2PFLDataset.from_json(dataset_name)
    elif data_source == "parquet":
        dataset = P2PFLDataset.from_parquet(dataset_name)
    elif data_source == "pandas":
        dataset = P2PFLDataset.from_pandas(dataset_name)
    elif data_source == "custom":
        # TODO: Just do the same as with the model
        # TODO: Just do the same as with the model
        # TODO: Just do the same as with the model
        # TODO: Just do the same as with the model
        raise NotImplementedError("Custom datasets are not supported to be loaded from YAML.")

    if not dataset:
        print("P2PFLDataset loading process completed without creating a dataset object (check for errors above).")
        return None

    # Partitioning
    partitioning_config = dataset_config.get("partitioning", {})
    if not partitioning_config:
        raise ValueError("Missing 'partitioning' configuration in YAML file.")
    partition_package = partitioning_config.get("package")
    partition_class_name = partitioning_config.get("strategy")
    if not partition_package or not partition_class_name:
        raise ValueError("Missing 'partition_strategy' configuration in YAML file.")
    partitions = dataset.generate_partitions(
        n,
        load_by_package_and_name(
            partition_package,
            partition_class_name,
        ),
        **partitioning_config.get("params", {}),
    )

    #########
    # Model #
    #########

    model_config = experiment_config.get("model", {})
    if not model_config:
        raise ValueError("Missing 'model' configuration in YAML file.")
    model_package = model_config.get("package")
    model_build_fn = model_config.get("model_build_fn")
    if not model_package or not model_build_fn:
        raise ValueError("Missing 'model' configuration in YAML file.")
    model_class = load_by_package_and_name(
        model_package,
        model_build_fn,
    )

    def model_fn() -> P2PFLModel:
        return model_class(**model_config.get("params", {}))

    ##############
    # Aggregator #
    ##############

    aggregator = experiment_config.get("aggregator")
    if not aggregator:
        raise ValueError("Missing 'aggregator' configuration in YAML file.")
    aggregator_package = aggregator.get("package")
    aggregator_class_name = aggregator.get("aggregator")
    if not aggregator_package or not aggregator_class_name:
        raise ValueError("Missing 'aggregator' configuration in YAML file.")
    aggregator_class = load_by_package_and_name(
        aggregator_package,
        aggregator_class_name,
    )

    def aggregator_fn() -> Aggregator:
        return aggregator_class(**aggregator.get("params", {}))

    ###########
    # Network #
    ###########

    # Create nodes
    nodes: list[Node] = []
    protocol_package = network_config.get("package")
    protocol_class_name = network_config.get("protocol")
    if not protocol_package or not protocol_class_name:
        raise ValueError("Missing 'protocol' configuration in YAML file.")
    protocol = load_by_package_and_name(
        protocol_package,
        protocol_class_name,
    )
    for i in range(n):
        node = Node(
            model_fn(),
            partitions[i],
            protocol=protocol,
            simulation=True,
            aggregator=aggregator_fn(),
        )
        node.start()
        nodes.append(node)

    try:
        # Connect nodes
        topology = network_config.get("topology")
        if not topology:
            raise ValueError("Missing 'topology' configuration in YAML file.")
        if n > Settings.gossip.TTL:
            raise ValueError("TTL less than the number of nodes. Some messages will not be delivered depending on the topology.")
        adjacency_matrix = TopologyFactory.generate_matrix(topology, len(nodes))
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)
        wait_convergence(nodes, n - 1, only_direct=False, wait=60, debug=False)  # type: ignore

        # Additional connections
        additional_connections = network_config.get("additional_connections")
        if additional_connections:
            for source, connect_to in additional_connections:
                nodes[source].connect(nodes[connect_to].addr)

        # Start Learning
        r = experiment_config.get("rounds")
        e = experiment_config.get("epochs")
        if r < 1:
            raise ValueError("Skipping training, amount of round is less than 1")

        # Start Learning
        nodes[0].set_start_learning(rounds=r, epochs=e)

        # Wait and check
        wait_to_finish(nodes, timeout=60 * 60)  # 1 hour | TODO: Make this configurable

    except Exception as e:
        raise e
    finally:
        # Stop Nodes
        for node in nodes:
            node.stop()
