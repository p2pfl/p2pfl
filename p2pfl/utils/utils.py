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
"""Utils."""

import time
from typing import List, Optional, Union

import numpy as np

from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings

"""
Module to define constants for the p2pfl system.
"""

###################
# Global Settings #
###################


def set_standalone_settings() -> None:
    """
    Set settings for testing.

    Important:
        - HEARTBEAT PERIOD: Too high values can cause late node discovery/fault detection. Too low values can cause high CPU usage.
        - GOSSIP PERIOD: Too low values can cause high CPU usage.
        - TTL: Low TTLs can cause that some messages are not delivered.

    """
    Settings.general.GRPC_TIMEOUT = 10
    Settings.heartbeat.PERIOD = 1
    Settings.heartbeat.TIMEOUT = 10
    Settings.heartbeat.WAIT_CONVERGENCE = 2
    Settings.heartbeat.EXCLUDE_BEAT_LOGS = True
    Settings.gossip.PERIOD = 0
    Settings.gossip.TTL = 10
    Settings.gossip.MESSAGES_PER_PERIOD = 9999999999
    Settings.gossip.AMOUNT_LAST_MESSAGES_SAVED = 10000
    Settings.gossip.MODELS_PERIOD = 2
    Settings.gossip.MODELS_PER_ROUND = 4
    Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS = 10
    Settings.training.VOTE_TIMEOUT = 60
    Settings.training.AGGREGATION_TIMEOUT = 60
    Settings.general.LOG_LEVEL = "INFO"
    logger.set_level(Settings.general.LOG_LEVEL)  # Refresh (maybe already initialized)


def wait_convergence(
    nodes: list[Node | CommunicationProtocol],
    n_neis: int,
    wait: Union[int, float] = 5,
    only_direct: bool = False,
    debug: bool = False,
) -> None:
    """
    Wait until all nodes have n_neis neighbors.

    Args:
        nodes: List of nodes.
        n_neis: Number of neighbors.
        wait: Time to wait.
        only_direct: Only direct neighbors.
        debug: Debug mode.

    Raises:
        AssertionError: If the condition is not met.

    """
    acum = 0.0
    while True:
        begin = time.time()
        if all(len(n.get_neighbors(only_direct=only_direct)) == n_neis for n in nodes):
            break
        if debug:
            logger.info(
                "Waiting for convergence",
                str([list(n.get_neighbors(only_direct=only_direct).keys()) for n in nodes]),
            )
        time.sleep(0.1)
        acum += time.time() - begin
        if acum > wait:
            raise AssertionError()


def full_connection(node: Node, nodes: List[Node]) -> None:
    """
    Connect node to all nodes.

    Args:
        node: Node to connect.
        nodes: List of nodes

    """
    for n in nodes:
        node.connect(n.addr)


def wait_to_finish(nodes: List[Node], timeout=3600, debug=False) -> None:
    """
    Wait until all nodes have finished the workflow.

    Args:
        nodes: List of nodes.
        timeout: Timeout in seconds (default: 1 hour = 3600 seconds).
        debug: Debug mode.

    Raises:
        TimeoutError: If the nodes don't finish within the timeout period.

    """
    # Wait until all nodes finish the workflow
    start = time.time()
    while True:
        if debug:
            logger.info(
                "Waiting for nodes to finish",
                str([n.learning_workflow.finished for n in nodes]),
            )
        if all(n.learning_workflow.finished for n in nodes):
            break
        time.sleep(1)
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(f"Timeout waiting for nodes to finish (elapsed: {int(elapsed//60)} minutes {int(elapsed%60)} seconds)")


def check_equal_models(nodes: List[Node]) -> None:
    """
    Check that all nodes have the same model.

    Args:
        nodes: List of nodes.

    Raises:
        AssertionError: If the condition is not met.

    """
    model_params: Optional[List[np.ndarray]] = None
    first = True
    for node in nodes:
        if first:
            model_params = node.learner.get_model().get_parameters()
            first = False
        else:
            # compare layers with a tolerance
            if model_params is None:
                raise ValueError("Model parameters are None")
            for i, layer in enumerate(model_params):
                assert np.allclose(
                    layer,
                    node.learner.get_model().get_parameters()[i],
                    atol=1e-1,
                )
