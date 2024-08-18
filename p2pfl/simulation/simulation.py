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

from typing import Any, Dict, Optional

from p2pfl.communication.memory.memory_communication_protocol import InMemoryCommunicationProtocol
from p2pfl.learning.pytorch.lightning_learner import LightningLearner
from p2pfl.node import Node
from p2pfl.simulation.actor_pool import SuperActorPool
from p2pfl.simulation.virtual_learner import VirtualNodeLearner
import ray
from p2pfl.management.logger import Logger, logger
import matplotlib.pyplot as plt

from p2pfl.utils import (
    wait_convergence,
    set_test_settings,
    wait_4_results,
)

NodeToPartitionMapping = Dict[int, int]

"""
def _create_node_id_to_partition_mapping(
    num_clients: int,
) -> NodeToPartitionMapping:
    Generate a node_id:partition_id mapping.
    nodes_mapping: NodeToPartitionMapping = {}  # {node-id; partition-id}
    for i in range(num_clients):
        while True:
            node_id = generate_rand_int_from_bytes(NODE_ID_NUM_BYTES)
            if node_id not in nodes_mapping:
                break
        nodes_mapping[node_id] = i
    return nodes_mapping
"""

def start_simulation(model, data, num_nodes,
                     client_resources: Optional[Dict[str, float]] = None,
                     ray_init_args: Optional[Dict[str, Any]]=None):
    """
    ray_init_args : Optional[Dict[str, Any]] (default: None)
        Optional dictionary containing arguments for the call to `ray.init`.
        If ray_init_args is None (the default), Ray will be initialized with
        the following default args:

        { "ignore_reinit_error": True, "include_dashboard": False }

        An empty dictionary can be used (ray_init_args={}) to prevent any
        arguments from being passed to ray.init.
    """
    
    logger.info.remote("localhost", "Starting simulation...")

    # Default arguments for Ray initialization
    if not ray_init_args:
        ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": True,
        }

    if ray.is_initialized():
        ray.shutdown()

    # Initialize Ray
    context = ray.init(**ray_init_args)
    logger.info.remote("localhost", f"Ray dashboard url: {context.dashboard_url}")

    cluster_resources = ray.cluster_resources()
    logger.info.remote("localhost", f"Ray initialized with resources: {cluster_resources}")

    # Log the resources that a single client will be able to use
    if client_resources is None:
        logger.info.remote("localhost",
            "No `client_resources` specified. Using minimal resources for clients.",
        )
        client_resources = {"num_cpus": 1, "num_gpus": 0.0}

    # Each client needs at the very least one CPU
    if "num_cpus" not in client_resources:
        logger.debug.remote("localhost",
            f"No `num_cpus` specified in `client_resources`. "+
            "Using `num_cpus=1` for each client."
        )
        client_resources["num_cpus"] = 1

    logger.info.remote("localhost",
        f"Resources for each Virtual Client: {client_resources}"
    )

    # Instantiate ActorPool
    pool = SuperActorPool(
        resources=client_resources,
    )
    
    #f_stop = threading.Event()
    
    # List actors
    logger.info.remote("localhost",
        f"Creating {pool.__class__.__name__} with {pool.num_actors} actors"
    )

    # Create node-id to partition-id mapping
    #nodes_mapping = _create_node_id_to_partition_mapping(num_nodes)


    # Register one RayClientProxy object for each client with the ClientManager
    #for node_id, partition_id in nodes_mapping.items():
    nodes = []
    for node_id in range(num_nodes):
        node = Node(
            model,
            data,
            address=f"virtual_{node_id}",
            protocol=InMemoryCommunicationProtocol,
            learner=LightningLearner,
            simulation=True
        )
        node.start()
        nodes.append(node)

    # Start Learning
    try:
        # Start Learning
        nodes[0].set_start_learning(rounds=2, epochs=2)

        # Wait and check
        wait_4_results(nodes)

        # Local Logs
        local_logs = logger.get_local_logs.remote()
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
        global_logs = logger.get_global_logs.remote()
        if global_logs != {}:
            logs = list(global_logs.items())[0][
                1
            ]  # Accessing the nested dictionary directly
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

    except Exception as ex:
        logger.error.remote("localhost", str(ex))
        #logger.error("localhost", traceback.format_exc())
        logger.error.remote("localhost",
            "Your simulation crashed :(. This could be because of several reasons. "+
            "The most common are: "+
            "\n\t > Sometimes, issues in the simulation code itself can cause crashes. "+
            "It's always a good idea to double-check your code for any potential bugs "+
            "or inconsistencies that might be contributing to the problem. "+
            "For example: "+
            "\n\t\t - You might be using a class attribute in your clients that "+
            "hasn't been defined."+
            "\n\t\t - There could be an incorrect method call to a 3rd party library "+
            "(e.g., PyTorch)."+
            "\n\t\t - The return types of methods in your clients/strategies might be "+
            "incorrect."+
            "\n\t > Your system couldn't fit a single VirtualClient: try lowering "+
            "`client_resources`."+
            "\n\t > All the actors in your pool crashed. This could be because: "+
            "\n\t\t - You clients hit an out-of-memory (OOM) error and actors couldn't "+
            "recover from it. Try launching your simulation with more generous "+
            "`client_resources` setting (i.e. it seems %s is "+
            "not enough for your run). Use fewer concurrent actors. "+
            "\n\t\t - You were running a multi-node simulation and all worker nodes "+
            "disconnected. The head node might still be alive but cannot accommodate "+
            "any actor with resources: %s."+
            "\nTake a look at the simulation examples for guidance "
        )
        raise RuntimeError("Simulation crashed.") from ex

    finally:
        # Stop time monitoring resources in cluster
        #f_stop.set()
        #event(EventType.START_SIMULATION_LEAVE)
        ray.shutdown()

    return