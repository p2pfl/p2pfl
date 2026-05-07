#
# This file is part of the p2pfl distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
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

import logging
import os
import time
import uuid
from typing import Any

import yaml

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.launch_from_yaml.utils import export_experiment_data, load_by_package_and_name, resize_partitions
from p2pfl.management.logger import logger
from p2pfl.management.logger.logger import P2PFLogger
from p2pfl.management.p2pfl_web_services import P2pflWebServices
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory
from p2pfl.utils.utils import wait_convergence, wait_to_finish


def _find_web_services(lgr: P2PFLogger) -> P2pflWebServices | None:
    """Walk the logger decorator chain to find the P2pflWebServices instance."""
    current: Any = lgr
    while current is not None:
        if hasattr(current, "_p2pfl_web_services") and current._p2pfl_web_services is not None:
            return current._p2pfl_web_services
        current = getattr(current, "_p2pfl_logger", None)
    return None


async def run_from_yaml(yaml_path: str, debug: bool = False) -> None:
    """
    Run a simulation from a YAML file.

    Args:
        yaml_path: The path to the YAML file.
        debug: If True, enable debug mode.

    """
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Parse YAML configuration
    with open(yaml_path) as file:
        config = yaml.safe_load(file)

    # Update settings
    custom_settings = config.get("settings", {})
    if custom_settings:
        Settings.set_from_dict(custom_settings)
        logger.set_level(Settings.general.LOG_LEVEL)

    # Get Amount of Nodes
    network_config = config.get("network", {})
    if not network_config:
        raise ValueError("Missing 'network' configuration in YAML file.")
    n = network_config.get("nodes")
    if not n:
        raise ValueError("Missing 'n' under 'network' configuration in YAML file.")

    #############
    # Profiling #
    #############

    profiling = config.get("profiling", {})
    profiling_enabled = profiling.get("enabled", False)
    profiling_output_dir = profiling.get("output_dir", "profile")
    if profiling_enabled:
        import asyncio
        import threading
        from concurrent.futures import ThreadPoolExecutor

        import yappi  # type: ignore

        yappi.set_context_name_callback(lambda: threading.current_thread().name)
        yappi.start()
        asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor(thread_name_prefix="p2pfl-asyncio"))

    actual_exp_name: str | None = None
    start_time = None
    if profiling.get("measure_time", False):
        start_time = time.time()

    ###################
    # Remote Loggers  #
    ###################

    remote_loggers = config.get("remote_loggers", {})
    if remote_loggers:
        logger.connect(**remote_loggers)

    ###########
    # Dataset #
    ###########

    experiment_config = config.get("experiment", {})
    dataset_config = experiment_config.get("dataset", {})
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
        package = dataset_config.get("package")
        dataset_class = dataset_config.get("class")
        if not package or not dataset_class:
            raise ValueError("Missing package or class for custom dataset")
        dataset_class = load_by_package_and_name(package, dataset_class)
        dataset = dataset_class(**dataset_config.get("params", {}))

    if not dataset:
        print("P2PFLDataset loading process completed without creating a dataset object (check for errors above).")
        return None

    # Batch size
    dataset.set_batch_size(dataset_config.get("batch_size", 1))

    # Partitioning (do this BEFORE applying transforms)
    partitioning_config = dataset_config.get("partitioning", {})
    if not partitioning_config:
        raise ValueError("Missing 'partitioning' configuration in YAML file.")
    partition_package = partitioning_config.get("package")
    partition_class_name = partitioning_config.get("strategy")
    if not partition_package or not partition_class_name:
        raise ValueError("Missing 'partition_strategy' configuration in YAML file.")
    reduced_dataset = partitioning_config.get("reduced_dataset", False)
    reduction_factor = partitioning_config.get("reduction_factor", 1)
    partitions = dataset.generate_partitions(
        n * reduction_factor if reduced_dataset else n,
        load_by_package_and_name(partition_package, partition_class_name),
        **partitioning_config.get("params", {}),
    )

    # Resize partitions for scalability tests
    samples_per_node = partitioning_config.get("samples_per_node", None)
    if samples_per_node is not None:
        test_samples_per_node = partitioning_config.get("test_samples_per_node", None)
        partitions = resize_partitions(partitions, samples_per_node, test_samples_per_node)

    # Transforms (apply AFTER partitioning)
    transforms_config = dataset_config.get("transforms", None)
    if transforms_config:
        transforms_package = transforms_config.get("package")
        transform_function = transforms_config.get("function")
        if not transforms_package or not transform_function:
            raise ValueError("Missing 'transforms' configuration in YAML file.")
        transform_fn = load_by_package_and_name(transforms_package, transform_function)
        params = transforms_config.get("params", {})
        transform = transform_fn(**params) if params else transform_fn
        for partition in partitions:
            partition.set_transforms(transform)

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
    model_class = load_by_package_and_name(model_package, model_build_fn)

    def model_fn() -> P2PFLModel:
        params = model_config.get("params", {})
        params = {**params, "compression": model_config.get("compression", None)}
        return model_class(**params)

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
    aggregator_class = load_by_package_and_name(aggregator_package, aggregator_class_name)

    def aggregator_fn() -> Aggregator:
        return aggregator_class(**aggregator.get("params", {}))

    ############
    # Workflow #
    ############
    workflow = experiment_config.get("workflow")
    if not workflow:
        raise ValueError("Missing 'workflow' configuration in YAML file.")
    workflow_name: str = workflow

    ###########
    # Network #
    ###########

    experiment_failed = False

    # Create nodes
    nodes: list[Node] = []
    protocol_package = network_config.get("package")
    protocol_class_name = network_config.get("protocol")
    if not protocol_package or not protocol_class_name:
        raise ValueError("Missing 'protocol' configuration in YAML file.")
    protocol = load_by_package_and_name(protocol_package, protocol_class_name)
    for i in range(n):
        node = Node(
            model_fn(),
            partitions[i],
            protocol=protocol(),
            aggregator=aggregator_fn(),
        )
        await node.start()
        nodes.append(node)

    try:
        # Connect nodes
        topology = network_config.get("topology")
        if not topology:
            raise ValueError("Missing 'topology' configuration in YAML file.")
        if n > Settings.gossip.TTL:
            print(
                f""""TTL less than the number of nodes ({Settings.gossip.TTL} < {n}).
                Some messages will not be delivered depending on the topology."""
            )
        adjacency_matrix = TopologyFactory.generate_matrix(topology, len(nodes))
        await TopologyFactory.connect_nodes(adjacency_matrix, nodes)
        await wait_convergence(nodes, n - 1, only_direct=False, wait=60, debug=True)  # type: ignore

        # Additional connections
        additional_connections = network_config.get("additional_connections")
        if additional_connections:
            for source, connect_to in additional_connections:
                await nodes[source].connect(nodes[connect_to].address)

        # Start Learning
        r = experiment_config.get("rounds")
        e = experiment_config.get("epochs")
        trainset_size = experiment_config.get("trainset_size")
        if r < 1:
            raise ValueError("Skipping training, amount of round is less than 1")

        # Forward workflow-specific params (tau, dmax, top_k_neighbors, etc.)
        reserved_keys = {"rounds", "epochs", "trainset_size", "workflow", "dataset", "model", "aggregator", "wait_timeout", "name", "seed"}
        extra_params = {k: v for k, v in experiment_config.items() if k not in reserved_keys}

        # Start Learning
        actual_exp_name = await nodes[0].set_start_learning(
            rounds=r, epochs=e, trainset_size=trainset_size, workflow=workflow_name, **extra_params
        )

        # Wait and check
        wait_timeout = experiment_config.get("wait_timeout", 60)
        await wait_to_finish(nodes, timeout=wait_timeout * 60, debug=debug)

    except (Exception, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            print("\nInterrupted.")
        else:
            experiment_failed = True
            print(f"\nExperiment failed: {e}")
    finally:
        # Collect data BEFORE stopping nodes (stop clears workflow)
        export_results = config.get("export_results", False)
        all_timings: list[dict[str, Any]] = []
        messages: list[Any] = []
        if export_results:
            for node in nodes:
                if node.workflow and hasattr(node.workflow, "stage_timings"):
                    all_timings.extend(node.workflow.stage_timings)
            messages = logger.get_messages()

        # Stop Nodes
        for node in nodes:
            await node.stop()

        # Execution time
        if start_time:
            print(f"Execution time: {time.time() - start_time} seconds")

        # Yappi profiling (optional)
        if profiling_enabled:
            import yappi

            yappi.stop()
            profile_dir = os.path.join(profiling_output_dir, str(uuid.uuid4()))
            os.makedirs(profile_dir, exist_ok=True)
            for thread in yappi.get_thread_stats():
                yappi.get_func_stats(ctx_id=thread.id).save(f"{profile_dir}/{thread.name}-{thread.id}.pstat", type="pstat")
            print(f"Profile stats saved in {profile_dir}")

            # Upload to web services if connected
            web_services = _find_web_services(logger)
            if web_services is not None and actual_exp_name is not None:
                web_services.upload_profiling(actual_exp_name, profile_dir)

        # Export experiment data (opt-in)
        if export_results:
            exp_name = experiment_config.get("name", "experiment")
            output_dir = config.get("output_dir", "results")
            exp_dir = os.path.join(output_dir, exp_name)

            if experiment_failed:
                print(f"Experiment '{exp_name}' FAILED — results not saved.")
            else:
                export_experiment_data(exp_dir, all_timings, messages, logger.get_global_logs())
                print(f"Experiment data saved in {exp_dir}")
