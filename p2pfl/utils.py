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
"""Utils."""

import time
from typing import Any, List

import numpy as np

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


def wait_convergence(nodes: List[Node], n_neis: int, wait: int = 5, only_direct: bool = False) -> None:
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


def wait_4_results(nodes: List[Node]) -> None:
    """
    Wait until all nodes have finished the rounds.

    Args:
        nodes: List of nodes.

    """
    while True:
        time.sleep(1)
        finish = True
        for f in [node.state.round is None for node in nodes]:
            finish = finish and f

        if finish:
            break


def check_equal_models(nodes: List[Node]) -> None:
    """
    Check that all nodes have the same model.

    Args:
        nodes: List of nodes.

    Raises:
        AssertionError: If the condition is not met.

    """
    model: Any = None
    first = True
    for node in nodes:
        if node.state.learner is None:
            raise AssertionError()
        if first:
            model = node.state.learner.get_parameters()
            first = False
        else:
            # compare layers with a tolerance
            for layer in model:
                assert np.allclose(
                    model[layer],
                    node.state.learner.get_parameters()[layer],
                    atol=1e-1,
                )
