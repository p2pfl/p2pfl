"""Checkpoint manager for managing checkpoint directories and file paths."""

import os
import re
from typing import Optional

from p2pfl.management.logger import logger
from p2pfl.settings import Settings


class CheckpointDirectoriesManager:
    """
    Manager for checkpoint directories and file paths.

    This class handles the creation and management of checkpoint directories
    organized by experiment name and node address. It provides methods to:
    - Create checkpoint directory structures
    - Generate checkpoint file paths
    - Check if directories exist and create them if needed
    - List existing checkpoints for a given experiment/node

    The directory structure is:
    ```
    {CHECKPOINT_DIR}/
      └── {experiment_name}/
          └── {node_addr}/
              ├── checkpoint_round_0.pkl
              ├── checkpoint_round_1.pkl
              └── ...
    ```

    Args:
        checkpoint_dir: Base directory for checkpoints. If None, uses Settings.general.CHECKPOINT_DIR.

    """

    def __init__(self, checkpoint_dir: Optional[str] = None) -> None:
        """Initialize the checkpoint manager."""
        self.base_dir = checkpoint_dir if checkpoint_dir is not None else Settings.general.CHECKPOINT_DIR

    def get_checkpoint_dir(self, experiment_name: str, node_addr: str) -> str:
        """
        Get the checkpoint directory path for a specific experiment and node.

        The directory structure is: {base_dir}/{experiment_name}/{node_addr}/

        Args:
            experiment_name: Name of the experiment.
            node_addr: Address of the node.

        Returns:
            The checkpoint directory path.

        """
        safe_exp_name = self._sanitize_filename(experiment_name)
        safe_node_addr = self._sanitize_filename(node_addr)

        checkpoint_dir = os.path.join(self.base_dir, safe_exp_name, safe_node_addr)
        return checkpoint_dir

    def ensure_checkpoint_dir(self, experiment_name: str, node_addr: str) -> str:
        """
        Ensure the checkpoint directory exists, creating it if necessary.

        Args:
            experiment_name: Name of the experiment.
            node_addr: Address of the node.

        Returns:
            The checkpoint directory path.

        Raises:
            OSError: If the directory cannot be created.

        """
        checkpoint_dir = self.get_checkpoint_dir(experiment_name, node_addr)

        if not os.path.exists(checkpoint_dir):
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                logger.debug(
                    node_addr,
                    f"Created checkpoint directory: {checkpoint_dir}",
                )
            except OSError as e:
                logger.error(
                    node_addr,
                    f"Failed to create checkpoint directory {checkpoint_dir}: {e}",
                )
                raise

        return checkpoint_dir

    def get_checkpoint_filepath(self, experiment_name: str, node_addr: str, round: int, checkpoint_type: str = "local", suffix: str = "pkl") -> str:
        """
        Generate the file path for a checkpoint at a specific round.

        Args:
            experiment_name: Name of the experiment.
            node_addr: Address of the node.
            round: Round number.
            checkpoint_type: Type of checkpoint ("local", "aggregated", "round_finished", etc.).
            suffix: File extension (default: "pkl").

        Returns:
            The full file path for the checkpoint.

        """
        checkpoint_dir = self.get_checkpoint_dir(experiment_name, node_addr)
        safe_type = self._sanitize_filename(checkpoint_type)
        filename = f"checkpoint_round_{round}_{safe_type}.{suffix}"
        filepath = os.path.join(checkpoint_dir, filename)
        return filepath

    def get_latest_checkpoint_filepath(self, experiment_name: str, node_addr: str, suffix: str = "pkl") -> Optional[str]:
        """
        Get the file path of the latest checkpoint for a given experiment and node.

        Args:
            experiment_name: Name of the experiment.
            node_addr: Address of the node.
            suffix: File extension (default: "pkl").

        Returns:
            The file path of the latest checkpoint, or None if no checkpoints exist.

        """
        checkpoint_dir = self.get_checkpoint_dir(experiment_name, node_addr)

        if not os.path.exists(checkpoint_dir):
            return None

        checkpoint_files = self.list_checkpoints(experiment_name, node_addr, suffix=suffix)

        if not checkpoint_files:
            return None

        checkpoint_files.sort(key=lambda x: self._extract_round_from_filename(x), reverse=True)
        return checkpoint_files[0]

    def list_checkpoints(self, experiment_name: str, node_addr: str, suffix: str = "pkl") -> list[str]:
        """
        List all checkpoint files for a given experiment and node.

        Args:
            experiment_name: Name of the experiment.
            node_addr: Address of the node.
            suffix: File extension (default: "pkl").

        Returns:
            List of full file paths to checkpoint files, sorted by round number.

        """
        checkpoint_dir = self.get_checkpoint_dir(experiment_name, node_addr)

        if not os.path.exists(checkpoint_dir):
            return []

        pattern = re.compile(r"checkpoint_round_(\d+)_[^.]+\.{}$".format(re.escape(suffix)))

        checkpoint_files = []
        for filename in os.listdir(checkpoint_dir):
            filepath = os.path.join(checkpoint_dir, filename)
            if os.path.isfile(filepath) and pattern.match(filename):
                checkpoint_files.append(filepath)

        checkpoint_files.sort(key=lambda x: self._extract_round_from_filename(x))

        return checkpoint_files

    def check_checkpoint_exists(
        self,
        experiment_name: str,
        node_addr: str,
        round: int,
        checkpoint_type: str = "local",
        suffix: str = "pkl",
    ) -> bool:
        """
        Check if a checkpoint file exists for a specific round.

        Args:
            experiment_name: Name of the experiment.
            node_addr: Address of the node.
            round: Round number.
            checkpoint_type: Type of checkpoint ("local", "aggregated", "remote", etc.).
            suffix: File extension (default: "pkl").

        Returns:
            True if the checkpoint exists, False otherwise.

        """
        filepath = self.get_checkpoint_filepath(experiment_name, node_addr, round, checkpoint_type=checkpoint_type, suffix=suffix)
        return os.path.exists(filepath) and os.path.isfile(filepath)

    def cleanup_old_checkpoints(
        self,
        experiment_name: str,
        node_addr: str,
        keep_last_n: int,
        suffix: str = "pkl",
    ) -> list[str]:
        """
        Clean up old checkpoint files, keeping only the most recent N checkpoints.

        Args:
            experiment_name: Name of the experiment.
            node_addr: Address of the node.
            keep_last_n: Number of recent checkpoints to keep.
            suffix: File extension (default: "pkl").

        Returns:
            List of file paths that were deleted.

        """
        checkpoint_files = self.list_checkpoints(experiment_name, node_addr, suffix=suffix)

        if len(checkpoint_files) <= keep_last_n:
            return []

        checkpoint_files.sort(key=lambda x: self._extract_round_from_filename(x))

        files_to_delete = checkpoint_files[:-keep_last_n] if keep_last_n > 0 else checkpoint_files

        deleted_files = []
        for filepath in files_to_delete:
            try:
                os.remove(filepath)
                deleted_files.append(filepath)
                logger.debug(
                    node_addr,
                    f"Deleted old checkpoint: {filepath}",
                )
            except OSError as e:
                logger.warning(
                    node_addr,
                    f"Failed to delete checkpoint {filepath}: {e}",
                )

        return deleted_files

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a string to be safe for use as a filename/directory name.

        Args:
            filename: The string to sanitize.

        Returns:
            A sanitized version of the string safe for filesystem use.

        """
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        sanitized = sanitized.strip(' .')
        sanitized = re.sub(r'_+', '_', sanitized)
        if not sanitized:
            sanitized = "unknown"

        return sanitized

    def _extract_round_from_filename(self, filepath: str) -> int:
        """
        Extract the round number from a checkpoint filename.

        Args:
            filepath: Full path to the checkpoint file.

        Returns:
            The round number, or -1 if extraction fails.

        """
        filename = os.path.basename(filepath)
        match = re.search(r"checkpoint_round_(\d+)", filename)
        if match:
            return int(match.group(1))
        return -1

