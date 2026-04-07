"""Checkpoint loading functionality."""

import pickle
from typing import Optional

import numpy as np

from p2pfl.checkpoints.checkpoint_data import LocalCheckpoint, RemoteCheckpoint
from p2pfl.checkpoints.path_manager import CheckpointDirectoriesManager
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.settings import Settings


def load_checkpoint(filepath: str) -> LocalCheckpoint:
    """
    Load a checkpoint from a file.

    Args:
        filepath: Path to the checkpoint file.

    Returns:
        LocalCheckpoint object.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the checkpoint file is invalid.

    """
    try:
        with open(filepath, "rb") as f:
            checkpoint_dict = pickle.load(f)
            checkpoint = LocalCheckpoint.from_dict(checkpoint_dict)
            return checkpoint
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint from {filepath}: {e}")


def load_remote_checkpoint(filepath: str) -> RemoteCheckpoint:
    """
    Load a remote checkpoint from a file.

    Args:
        filepath: Path to the remote checkpoint file.

    Returns:
        RemoteCheckpoint object.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the checkpoint file is invalid.

    """
    try:
        with open(filepath, "rb") as f:
            checkpoint_dict = pickle.load(f)
            checkpoint = RemoteCheckpoint.from_dict(checkpoint_dict)
            return checkpoint
    except FileNotFoundError:
        raise FileNotFoundError(f"Remote checkpoint file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Failed to load remote checkpoint from {filepath}: {e}")


def restore_from_checkpoint(
    checkpoint: LocalCheckpoint | RemoteCheckpoint,
    learner: Learner,
    last_backup_checkpoint: Optional[LocalCheckpoint] = None,
) -> None:
    """
    Restore model from a checkpoint.

    Args:
        checkpoint: Checkpoint to restore from (LocalCheckpoint or RemoteCheckpoint).
        learner: Learner object to restore the model to.
        last_backup_checkpoint: Last backup checkpoint (for delta restoration).

    """
    try:
        import gzip

        if isinstance(checkpoint, RemoteCheckpoint):
            # Restore from remote checkpoint (delta)
            model = learner.get_model()
            if last_backup_checkpoint:
                # Apply delta to last backup
                # Decompress delta
                delta_params_list = pickle.loads(gzip.decompress(checkpoint.compressed_model_params))

                # Decode last backup params
                last_backup_params_list, _ = model.decode_parameters(last_backup_checkpoint.model_params)

                # Apply delta
                restored_params = []
                for delta, last_backup in zip(delta_params_list, last_backup_params_list, strict=False):
                    restored = np.array(last_backup) + np.array(delta)
                    restored_params.append(restored)

                # Set model parameters directly
                model.set_parameters(restored_params)
            else:
                # No last backup: decompress full checkpoint
                # The decompressed data should be in the same format as model_params
                decompressed_params = gzip.decompress(checkpoint.compressed_model_params)
                params_list, _ = model.decode_parameters(decompressed_params)
                model.set_parameters(params_list)

        else:
            # Restore from local checkpoint
            model = learner.get_model()
            params_list, _ = model.decode_parameters(checkpoint.model_params)
            model.set_parameters(params_list)

        logger.info("", f"Model restored from checkpoint (round {checkpoint.metadata.round})")

    except Exception as e:
        logger.error("", f"Failed to restore from checkpoint: {e}")
        raise


def rejoin(
    state: NodeState,
    learner: Learner,
    communication_protocol,
    experiment_name: Optional[str] = None,
    round: Optional[int] = None,
) -> bool:
    """
    Rejoin the training process by restoring from checkpoint.

    Priority:
    1. Try local checkpoint recovery
    2. Try remote checkpoint recovery from neighbors
    3. Initialize fresh (worst case)

    Args:
        state: NodeState object.
        learner: Learner object.
        communication_protocol: Communication protocol for fetching remote checkpoints.
        experiment_name: Experiment name. If None, uses state.experiment.exp_name.
        round: Round number. If None, uses state.round.

    Returns:
        True if recovery was successful, False otherwise.

    """
    if state.experiment is None:
        logger.warning(state.addr, "Cannot rejoin: experiment not initialized")
        return False

    if experiment_name is None:
        experiment_name = state.experiment.exp_name

    if round is None:
        round = state.round
        if round is None:
            logger.warning(state.addr, "Cannot rejoin: round not set")
            return False

    dir_manager = CheckpointDirectoriesManager()

    # Priority 1: Try local checkpoint recovery
    logger.info(state.addr, "Attempting local checkpoint recovery...")
    # Try different checkpoint types
    for checkpoint_type in ["local", "aggregated", "round_finished"]:
        local_filepath = dir_manager.get_checkpoint_filepath(
            experiment_name, state.addr, round, checkpoint_type=checkpoint_type
        )
        if dir_manager.check_checkpoint_exists(experiment_name, state.addr, round, checkpoint_type=checkpoint_type, suffix="pkl"):
            try:
                checkpoint = load_checkpoint(local_filepath)
                restore_from_checkpoint(checkpoint, learner)
                logger.info(
                    state.addr,
                    f"Successfully recovered from local checkpoint (type: {checkpoint_type})",
                )
                return True
            except Exception as e:
                logger.debug(state.addr, f"Local checkpoint recovery failed ({checkpoint_type}): {e}")
                continue

    # Priority 2: Try remote checkpoint recovery from neighbors
    logger.info(state.addr, "Attempting remote checkpoint recovery from neighbors...")
    neighbors = communication_protocol.get_neighbors(only_direct=False)
    if isinstance(neighbors, dict):
        neighbor_addrs = list(neighbors.keys())
    elif isinstance(neighbors, list):
        neighbor_addrs = neighbors
    else:
        neighbor_addrs = []

    for neighbor_addr in neighbor_addrs:
        try:
            # Check if neighbor has remote checkpoint for this node
            # Note: We need to check all rounds up to current round
            for check_round in range(round, max(0, round - Settings.general.REMOTE_CHECKPOINT_INTERVAL * 2), -1):
                remote_filepath = dir_manager.get_checkpoint_filepath(
                    experiment_name, neighbor_addr, check_round, checkpoint_type="remote", suffix="pkl"
                )

                if not dir_manager.check_checkpoint_exists(experiment_name, neighbor_addr, check_round, checkpoint_type="remote", suffix="pkl"):
                    continue
                # Try to load last backup for delta restoration
                last_backup = None
                for backup_round in range(check_round - 1, max(0, check_round - Settings.general.REMOTE_CHECKPOINT_INTERVAL * 2), -1):
                    last_backup_filepath = dir_manager.get_checkpoint_filepath(
                        experiment_name, state.addr, backup_round, checkpoint_type="local"
                    )
                    if dir_manager.check_checkpoint_exists(experiment_name, state.addr, backup_round, checkpoint_type="local", suffix="pkl"):
                        try:
                            last_backup = load_checkpoint(last_backup_filepath)
                            break
                        except Exception:
                            continue

                try:
                    remote_checkpoint = load_remote_checkpoint(remote_filepath)
                    if remote_checkpoint.original_node_id == state.addr:
                        restore_from_checkpoint(remote_checkpoint, learner, last_backup)
                        logger.info(
                            state.addr,
                            f"Successfully recovered from remote checkpoint on {neighbor_addr} (round {check_round})",
                        )
                        return True
                except Exception as e:
                    logger.debug(state.addr, f"Failed to load remote checkpoint from {neighbor_addr} (round {check_round}): {e}")
                    continue

        except Exception as e:
            logger.debug(state.addr, f"Error checking neighbor {neighbor_addr}: {e}")
            continue

    # Priority 3: Initialize fresh (worst case)
    logger.warning(state.addr, "No checkpoint found. Initializing fresh model.")
    return False

