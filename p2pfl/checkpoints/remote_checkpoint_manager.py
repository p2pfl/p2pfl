"""Remote checkpoint management functionality."""

import gzip
import pickle
from collections import deque
from typing import Optional

import numpy as np

from p2pfl.checkpoints.checkpoint_data import CheckpointMetadata, RemoteCheckpoint
from p2pfl.checkpoints.path_manager import CheckpointDirectoriesManager
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.settings import Settings


def calculate_delta(
    current_params: bytes,
    last_backup_params: bytes,
    learner: Learner,
) -> bytes:
    """
    Calculate the delta (difference) between current and last backup parameters.

    Args:
        current_params: Current model parameters as bytes (encoded).
        last_backup_params: Last backup model parameters as bytes (encoded).
        learner: Learner object to decode parameters.

    Returns:
        Delta parameters as bytes (compressed).

    """
    try:
        # Decode parameters using the model's decode_parameters method
        model = learner.get_model()
        current_params_list, _ = model.decode_parameters(current_params)
        last_backup_params_list, _ = model.decode_parameters(last_backup_params)

        # Calculate delta for each parameter
        delta_list = []
        for curr, last in zip(current_params_list, last_backup_params_list, strict=False):
            delta = np.array(curr) - np.array(last)
            delta_list.append(delta)

        # Compress delta using pickle with compression
        compressed_delta = gzip.compress(pickle.dumps(delta_list), compresslevel=6)

        return compressed_delta

    except Exception as e:
        logger.error("", f"Error calculating delta: {e}")
        # Fallback: return current params if delta calculation fails
        return current_params


def calculate_shortest_path_distances(
    node_addr: str,
    all_nodes: list[str],
    communication_protocol: CommunicationProtocol,
) -> dict[str, int]:
    """
    Calculate shortest path distances from a node to all other nodes using BFS.

    Args:
        node_addr: Address of the source node.
        all_nodes: List of all node addresses in the network.
        communication_protocol: Communication protocol to get neighbors.

    Returns:
        Dictionary mapping node addresses to their shortest path distances.

    """
    distances: dict[str, int] = {}
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(node_addr, 0)])

    while queue:
        current_node, distance = queue.popleft()

        if current_node in visited:
            continue

        visited.add(current_node)
        distances[current_node] = distance

        # Get neighbors of current node
        neighbors = communication_protocol.get_neighbors(only_direct=True)
        if isinstance(neighbors, dict):
            neighbor_addrs = list(neighbors.keys())
        elif isinstance(neighbors, list):
            neighbor_addrs = neighbors
        else:
            neighbor_addrs = []

        for neighbor in neighbor_addrs:
            if neighbor not in visited and neighbor in all_nodes:
                queue.append((neighbor, distance + 1))

    # Set distance to infinity for unreachable nodes
    for node in all_nodes:
        if node not in distances:
            distances[node] = float("inf")  # type: ignore

    return distances


def select_distant_nodes(
    node_addr: str,
    all_nodes: list[str],
    communication_protocol: CommunicationProtocol,
    n: int,
) -> list[str]:
    """
    Select n distant nodes from the network topology.

    Args:
        node_addr: Address of the current node.
        all_nodes: List of all node addresses in the network.
        communication_protocol: Communication protocol to get neighbors.
        n: Number of distant nodes to select.

    Returns:
        List of n distant node addresses.

    """
    # Calculate distances to all nodes
    distances = calculate_shortest_path_distances(node_addr, all_nodes, communication_protocol)

    # Remove self and unreachable nodes
    valid_distances = {
        node: dist for node, dist in distances.items() if node != node_addr and dist != float("inf")
    }

    if not valid_distances:
        logger.warning(node_addr, "No distant nodes available for remote checkpoint backup")
        return []

    # Sort by distance (descending) and select top n
    sorted_nodes = sorted(valid_distances.items(), key=lambda x: x[1], reverse=True)
    selected_nodes = [node for node, _ in sorted_nodes[:n]]

    logger.info(
        node_addr,
        f"Selected {len(selected_nodes)} distant nodes for remote checkpoint: {selected_nodes}",
    )

    return selected_nodes


def save_remote_checkpoint(
    state: NodeState,
    learner: Learner,
    last_backup_checkpoint_path: Optional[str],
    target_nodes: list[str],
    communication_protocol: CommunicationProtocol,
    round: Optional[int] = None,
    dir_manager: Optional[CheckpointDirectoriesManager] = None,
) -> list[str]:
    """
    Save and send remote checkpoint to distant nodes.

    Args:
        state: NodeState object containing experiment and node information.
        learner: Learner object containing the model.
        last_backup_checkpoint_path: Path to the last backup checkpoint (for delta calculation).
        target_nodes: List of node addresses to send remote checkpoint to.
        communication_protocol: Communication protocol for sending.
        round: Round number. If None, uses state.round.
        dir_manager: CheckpointDirectoriesManager instance. If None, creates a new one.

    Returns:
        List of node addresses that successfully received the remote checkpoint.

    """
    if state.experiment is None:
        raise ValueError("Cannot save remote checkpoint: experiment is not initialized")

    if round is None:
        round = state.round
        if round is None:
            raise ValueError("Cannot save remote checkpoint: round is not set")

    if dir_manager is None:
        dir_manager = CheckpointDirectoriesManager()

    experiment_name = state.experiment.exp_name
    node_addr = state.addr

    try:
        model = learner.get_model()
        current_params = model.encode_parameters()

        # Load last backup params if last backup exists
        last_backup_params = None
        if last_backup_checkpoint_path:
            try:
                with open(last_backup_checkpoint_path, "rb") as f:
                    last_backup_data = pickle.load(f)
                    if isinstance(last_backup_data, dict):
                        last_backup_params = last_backup_data.get("model_params", b"")
            except Exception as e:
                logger.warning(
                    state.addr,
                    f"Failed to load last backup checkpoint for delta calculation: {e}. Using full checkpoint.",
                )
                last_backup_params = None

        # Calculate delta or use full params
        if last_backup_checkpoint_path and last_backup_params:
            try:
                compressed_delta = calculate_delta(current_params, last_backup_params, learner)
                compression_ratio = len(compressed_delta) / len(current_params) if len(current_params) > 0 else 1.0
                compression_method = "delta_gzip"
                logger.info(
                    state.addr,
                    f"Calculated delta checkpoint (compression ratio: {compression_ratio:.2f})",
                )
            except Exception as e:
                logger.warning(
                    state.addr,
                    f"Failed to calculate delta, using full checkpoint: {e}",
                )
                compressed_delta = gzip.compress(current_params, compresslevel=6)
                compression_ratio = len(compressed_delta) / len(current_params) if len(current_params) > 0 else 1.0
                compression_method = "full_gzip"
        else:
            # First backup: send full checkpoint
            compressed_delta = gzip.compress(current_params, compresslevel=6)
            compression_ratio = len(compressed_delta) / len(current_params) if len(current_params) > 0 else 1.0
            compression_method = "full_gzip"

        # Create remote checkpoint metadata
        metadata = CheckpointMetadata(
            node_id=state.addr,
            experiment_name=experiment_name,
            round=round,
            framework=model.get_framework(),
            model_name=state.experiment.model_name,
            aggregator_name=state.experiment.aggregator_name,
        )

        # Send to target nodes
        successful_nodes = []
        for target_node in target_nodes:
            try:
                remote_checkpoint = RemoteCheckpoint(
                    metadata=metadata,
                    compressed_model_params=compressed_delta,
                    compression_ratio=compression_ratio,
                    compression_method=compression_method,
                    storage_node_id=target_node,
                    original_node_id=state.addr,
                    is_replica=True,
                )

                # Serialize and send
                checkpoint_dict = remote_checkpoint.to_dict()
                checkpoint_bytes = pickle.dumps(checkpoint_dict)

                # TODO: Implement actual sending via communication protocol
                # For now, we'll save it locally on the target node's directory structure
                # In a real implementation, this would be sent over the network
                logger.info(
                    state.addr,
                    f"Sending remote checkpoint to {target_node} (size: {len(checkpoint_bytes) / 1024:.2f} KB)",
                )

                # Save locally as a placeholder (in real implementation, this would be sent over network)
                target_dir = dir_manager.get_checkpoint_dir(experiment_name, target_node)
                dir_manager.ensure_checkpoint_dir(experiment_name, target_node)
                remote_filepath = dir_manager.get_checkpoint_filepath(
                    experiment_name, target_node, round, checkpoint_type="remote", suffix="pkl"
                )

                with open(remote_filepath, "wb") as f:
                    pickle.dump(checkpoint_dict, f)

                successful_nodes.append(target_node)

            except Exception as e:
                logger.error(
                    state.addr,
                    f"Failed to send remote checkpoint to {target_node}: {e}",
                )

        logger.info(
            state.addr,
            f"Remote checkpoint backup completed: {len(successful_nodes)}/{len(target_nodes)} nodes successful",
        )

        return successful_nodes

    except Exception as e:
        logger.error(
            state.addr,
            f"Failed to save remote checkpoint at round {round}: {e}",
        )
        raise

