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


def set_test_settings() -> None:
    """Set settings for testing."""
    Settings.GRPC_TIMEOUT = 0.5
    Settings.HEARTBEAT_PERIOD = 0.5
    Settings.HEARTBEAT_TIMEOUT = 2
    Settings.GOSSIP_PERIOD = 0
    Settings.TTL = 10
    Settings.GOSSIP_MESSAGES_PER_PERIOD = 100
    Settings.AMOUNT_LAST_MESSAGES_SAVED = 100
    Settings.GOSSIP_MODELS_PERIOD = 1
    Settings.GOSSIP_MODELS_PER_ROUND = 4
    Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 4
    Settings.TRAIN_SET_SIZE = 4
    Settings.VOTE_TIMEOUT = 60
    Settings.AGGREGATION_TIMEOUT = 60
    Settings.WAIT_HEARTBEATS_CONVERGENCE = 0.2 * Settings.HEARTBEAT_TIMEOUT
    Settings.LOG_LEVEL = "DEBUG"
    logger.set_level(Settings.LOG_LEVEL)  # Refresh (maybe already initialized)


def wait_convergence(
    nodes: List[Union[Node, CommunicationProtocol]], n_neis: int, wait: Union[int, float] = 5, only_direct: bool = False
) -> None:
    """
    Wait until all nodes have n_neis neighbors.

    Args:
        nodes: List of nodes.
        n_neis: Number of neighbors.
        wait: Time to wait.
        only_direct: Only direct neighbors.

    Raises:
        AssertionError: If the condition is not met.

    """
    acum = 0.0
    while True:
        begin = time.time()
        if all(len(n.get_neighbors(only_direct=only_direct)) == n_neis for n in nodes):
            break
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


def wait_to_finish(nodes: List[Node], timeout=60):
    """
    Wait until all nodes have finished the workflow.

    Args:
        nodes: List of nodes.
        timeout: Timeout.

    """
    # Wait untill all nodes finised the workflow
    start = time.time()
    while True:
        if all(n.learning_workflow.finished for n in nodes):
            break
        time.sleep(1)
        if time.time() - start > timeout:
            raise TimeoutError("Timeout waiting for nodes to finish")


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
