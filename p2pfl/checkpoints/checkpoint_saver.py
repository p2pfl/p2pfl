"""Checkpoint saving functionality."""

import pickle
from typing import Optional

from p2pfl.checkpoints.checkpoint_data import CheckpointMetadata, LocalCheckpoint
from p2pfl.checkpoints.path_manager import CheckpointDirectoriesManager
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState


def save_checkpoint(
    state: NodeState,
    learner: Learner,
    round: Optional[int] = None,
    include_evaluation: bool = False,
    checkpoint_type: str = "local",
    dir_manager: Optional[CheckpointDirectoriesManager] = None,
) -> str:
    """
    Save a checkpoint to disk.

    This function collects all necessary information from the state and learner,
    creates a LocalCheckpoint object, and saves it to a file using pickle.

    Args:
        state: NodeState object containing experiment and node information.
        learner: Learner object containing the model.
        round: Round number. If None, uses state.round.
        include_evaluation: Whether to include evaluation metrics (may be slow).
        checkpoint_type: Type of checkpoint ("local", "aggregated", etc.).
        dir_manager: CheckpointDirectoriesManager instance. If None, creates a new one.

    Returns:
        The filepath where the checkpoint was saved.

    Raises:
        ValueError: If state.experiment is None or round is invalid.
        OSError: If the file cannot be written.

    """
    if state.experiment is None:
        raise ValueError("Cannot save checkpoint: experiment is not initialized")

    if round is None:
        round = state.round
        if round is None:
            raise ValueError("Cannot save checkpoint: round is not set")

    if dir_manager is None:
        dir_manager = CheckpointDirectoriesManager()

    experiment_name = state.experiment.exp_name
    node_addr = state.addr
    dir_manager.ensure_checkpoint_dir(experiment_name, node_addr)

    filepath = dir_manager.get_checkpoint_filepath(experiment_name, node_addr, round, checkpoint_type=checkpoint_type)

    try:
        model = learner.get_model()
        model_params = model.encode_parameters()
        framework = model.get_framework()
        try:
            contributors = model.get_contributors()
        except ValueError:
            contributors = [state.addr] 

        try:
            num_samples = model.get_num_samples()
        except ValueError:
            num_samples = 0 
        model_additional_info = model.get_info()
        compression_info = None
        if hasattr(model, "compression") and model.compression:
            compression_info = model.compression.copy()
            is_compressed = len(model.compression) > 0
        else:
            is_compressed = False
        experiment_metadata = state.experiment.to_dict()
        evaluation_metrics = None
        if include_evaluation:
            try:
                evaluation_metrics = learner.evaluate()
                logger.debug(
                    state.addr,
                    f"Evaluation metrics collected for checkpoint: {evaluation_metrics}",
                )
            except Exception as e:
                logger.warning(
                    state.addr,
                    f"Failed to evaluate model for checkpoint: {e}. Continuing without evaluation metrics.",
                )
        metadata = CheckpointMetadata(
            node_id=state.addr,
            experiment_name=experiment_name,
            round=round,
            framework=framework,
            model_name=state.experiment.model_name,
            aggregator_name=state.experiment.aggregator_name,
        )
        checkpoint = LocalCheckpoint(
            metadata=metadata,
            model_params=model_params,
            experiment_metadata=experiment_metadata,
            evaluation_metrics=evaluation_metrics,
            contributors=contributors,
            num_samples=num_samples,
            model_additional_info=model_additional_info,
            compression_info=compression_info,
            checkpoint_type=checkpoint_type,
            is_compressed=is_compressed,
        )
        with open(filepath, "wb") as f:
            checkpoint_dict = checkpoint.to_dict()
            pickle.dump(checkpoint_dict, f)
        checkpoint_size_mb = checkpoint.get_size_bytes() / (1024 * 1024)
        logger.info(
            state.addr,
            f"Checkpoint saved: round={round}, size={checkpoint_size_mb:.2f}MB, path={filepath}",
        )

        return filepath

    except Exception as e:
        logger.error(
            state.addr,
            f"Failed to save checkpoint at round {round}: {e}",
        )
        raise


def save_checkpoint_simple(
    state: NodeState,
    learner: Learner,
    checkpoint_dir: Optional[str] = None,
    round: Optional[int] = None,
) -> str:
    """
    Simplified checkpoint saving function with backward compatibility.

    This is a convenience wrapper around save_checkpoint() that matches
    the original function signature from the workflow design.

    Args:
        state: NodeState object containing experiment and node information.
        learner: Learner object containing the model.
        checkpoint_dir: Base directory for checkpoints (deprecated, use Settings).
        round: Round number. If None, uses state.round.

    Returns:
        The filepath where the checkpoint was saved.

    """
    dir_manager = CheckpointDirectoriesManager(checkpoint_dir=checkpoint_dir) if checkpoint_dir else None

    return save_checkpoint(
        state=state,
        learner=learner,
        round=round,
        include_evaluation=False,
        checkpoint_type="local",
        dir_manager=dir_manager,
    )

