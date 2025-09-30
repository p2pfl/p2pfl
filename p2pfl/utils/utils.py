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
    Settings.training.RAY_ACTOR_POOL_SIZE = 1
    Settings.general.LOG_LEVEL = "INFO"
    logger.set_level(Settings.general.LOG_LEVEL)  # Refresh (maybe already initialized)


def wait_convergence(
    nodes: list[Node | CommunicationProtocol],
    n_neis: int,
    wait: int | float = 5,
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
            if debug:
                _print_connectivity_matrix(nodes, only_direct, final=True)
            break
        if debug:
            _print_connectivity_matrix(nodes, only_direct, final=False)
        time.sleep(1)
        acum += time.time() - begin
        if acum > wait:
            raise AssertionError()


def _print_connectivity_matrix(
    nodes: list[Node | CommunicationProtocol],
    only_direct: bool = False,
    final: bool = False,
) -> None:
    """
    Print a visual connectivity matrix showing node connections.

    Args:
        nodes: List of nodes.
        only_direct: Only direct neighbors.
        final: Whether this is the final converged state.

    """
    n = len(nodes)

    # Build connectivity matrix
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    for i, node in enumerate(nodes):
        neighbors = node.get_neighbors(only_direct=only_direct)
        for j, other_node in enumerate(nodes):
            if i != j and other_node.addr in neighbors:
                matrix[i][j] = 1

    # Build complete visualization as a single string
    output_lines = []

    # Print header
    if final:
        output_lines.append("=" * 50)
        output_lines.append("CONVERGENCE ACHIEVED - Final Network Topology")
        output_lines.append("=" * 50)
    else:
        output_lines.append("-" * 50)
        output_lines.append("Waiting for convergence - Current Network State")
        output_lines.append("-" * 50)

    # Print node addresses for reference
    output_lines.append("Node Addresses:")
    for i, node in enumerate(nodes):
        addr_display = node.addr if len(str(node.addr)) < 20 else f"...{str(node.addr)[-17:]}"
        output_lines.append(f"  Node {i}: {addr_display}")

    # Print connectivity matrix with visual formatting
    output_lines.append("\nConnectivity Matrix:")
    output_lines.append("    " + "".join(f" {i:2}" for i in range(n)))
    output_lines.append("   +" + "---" * n + "+")

    for i in range(n):
        row_str = f"{i:2} |"
        for j in range(n):
            if i == j:
                row_str += " · "  # Self-connection (diagonal)
            elif matrix[i][j] == 1:
                row_str += " ■ "  # Connected
            else:
                row_str += " □ "  # Not connected
        row_str += f"| ({sum(matrix[i])} connections)"
        output_lines.append(row_str)

    output_lines.append("   +" + "---" * n + "+")

    # Print summary statistics
    total_connections = sum(sum(row) for row in matrix)
    output_lines.append("\nSummary:")
    output_lines.append(f"  Total nodes: {n}")
    output_lines.append(f"  Total connections: {total_connections}")
    output_lines.append(f"  Average connections per node: {total_connections / n:.1f}")

    # Check convergence status
    connections_per_node = [sum(row) for row in matrix]
    if all(c == connections_per_node[0] for c in connections_per_node):
        output_lines.append(f"  Status: Uniform topology ({connections_per_node[0]} connections each)")
    else:
        output_lines.append(f"  Status: Non-uniform (range: {min(connections_per_node)}-{max(connections_per_node)})")

    output_lines.append("=" * 50 if final else "-" * 50)
    output_lines.append("")  # Add blank line for readability

    # Log the complete visualization with proper format
    matrix_display = "\n".join(output_lines)
    if final:
        logger.info("Network Convergence", matrix_display)
    else:
        logger.info("Waiting for convergence", matrix_display)


def full_connection(node: Node, nodes: list[Node]) -> None:
    """
    Connect node to all nodes.

    Args:
        node: Node to connect.
        nodes: List of nodes

    """
    for n in nodes:
        node.connect(n.addr)


def wait_to_finish(nodes: list[Node], timeout=3600, debug=False) -> None:
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
            raise TimeoutError(f"Timeout waiting for nodes to finish (elapsed: {int(elapsed // 60)} minutes {int(elapsed % 60)} seconds)")


def check_equal_models(nodes: list[Node]) -> None:
    """
    Check that all nodes have the same model.

    Args:
        nodes: List of nodes.

    Raises:
        AssertionError: If the condition is not met.

    """
    model_params: list[np.ndarray] | None = None
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
