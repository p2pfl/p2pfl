"""Checkpoint data structures for local and remote checkpointing."""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

CHECKPOINT_VERSION = 1
"""Current checkpoint format version for compatibility checking."""


@dataclass
class CheckpointMetadata:
    """
    Metadata for a checkpoint.

    Contains information about the checkpoint that doesn't change between
    local and remote versions.

    Attributes:
        node_id: The address/ID of the node that created this checkpoint.
        experiment_name: Name of the experiment.
        round: Training round number when checkpoint was created.
        version: Checkpoint format version for compatibility.
        timestamp: Unix timestamp when checkpoint was created.
        framework: Framework name (e.g., "pytorch", "tensorflow", "flax").
        model_name: Name of the model architecture.
        aggregator_name: Name of the aggregator used.
    """

    node_id: str
    experiment_name: str
    round: int
    version: int = CHECKPOINT_VERSION
    timestamp: float = field(default_factory=time.time)
    framework: Optional[str] = None
    model_name: Optional[str] = None
    aggregator_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "node_id": self.node_id,
            "experiment_name": self.experiment_name,
            "round": self.round,
            "version": self.version,
            "timestamp": self.timestamp,
            "framework": self.framework,
            "model_name": self.model_name,
            "aggregator_name": self.aggregator_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMetadata":
        """Create metadata from dictionary."""
        return cls(
            node_id=data["node_id"],
            experiment_name=data["experiment_name"],
            round=data["round"],
            version=data.get("version", CHECKPOINT_VERSION),
            timestamp=data.get("timestamp", time.time()),
            framework=data.get("framework"),
            model_name=data.get("model_name"),
            aggregator_name=data.get("aggregator_name"),
        )


@dataclass
class LocalCheckpoint:
    """
    Local checkpoint data structure.

    Contains complete checkpoint information for local storage.
    This includes full model parameters, experiment metadata, and evaluation metrics.

    Attributes:
        metadata: Checkpoint metadata.
        model_params: Encoded model parameters (bytes, may be compressed).
        experiment_metadata: Full experiment configuration dictionary.
        evaluation_metrics: Evaluation metrics dictionary (optional).
        contributors: List of node addresses that contributed to this model.
        num_samples: Number of training samples used.
        model_additional_info: Additional model information (e.g., from callbacks).
        compression_info: Information about compression applied to model_params.
        checkpoint_type: Type of checkpoint ("local", "remote", "aggregated").
        is_compressed: Whether model_params is compressed.
    """

    metadata: CheckpointMetadata
    model_params: bytes
    experiment_metadata: dict[str, Any]
    evaluation_metrics: Optional[dict[str, float]] = None
    contributors: list[str] = field(default_factory=list)
    num_samples: int = 0
    model_additional_info: dict[str, Any] = field(default_factory=dict)
    compression_info: Optional[dict[str, Any]] = None
    checkpoint_type: str = "local"
    is_compressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "model_params": self.model_params,
            "experiment_metadata": self.experiment_metadata,
            "evaluation_metrics": self.evaluation_metrics,
            "contributors": self.contributors,
            "num_samples": self.num_samples,
            "model_additional_info": self.model_additional_info,
            "compression_info": self.compression_info,
            "checkpoint_type": self.checkpoint_type,
            "is_compressed": self.is_compressed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocalCheckpoint":
        """Create checkpoint from dictionary."""
        return cls(
            metadata=CheckpointMetadata.from_dict(data["metadata"]),
            model_params=data["model_params"],
            experiment_metadata=data["experiment_metadata"],
            evaluation_metrics=data.get("evaluation_metrics"),
            contributors=data.get("contributors", []),
            num_samples=data.get("num_samples", 0),
            model_additional_info=data.get("model_additional_info", {}),
            compression_info=data.get("compression_info"),
            checkpoint_type=data.get("checkpoint_type", "local"),
            is_compressed=data.get("is_compressed", False),
        )

    def get_size_bytes(self) -> int:
        """Get the size of the checkpoint in bytes."""
        return len(self.model_params)


@dataclass
class RemoteCheckpoint:
    """
    Remote checkpoint data structure.

    A compressed replica stored on neighbor nodes.
    Contains minimal information needed for recovery, with compressed model parameters.

    Attributes:
        metadata: Checkpoint metadata (same as local).
        compressed_model_params: Compressed model parameters (bytes).
        compression_ratio: Compression ratio achieved.
        compression_method: Method used for compression.
        storage_node_id: ID of the node storing this remote checkpoint.
        original_node_id: ID of the node that originally created this checkpoint.
        is_replica: Whether this is a replica of another node's checkpoint.
    """

    metadata: CheckpointMetadata
    compressed_model_params: bytes
    compression_ratio: Optional[float] = None
    compression_method: Optional[str] = None
    storage_node_id: str = ""
    original_node_id: str = ""
    is_replica: bool = True

    def __post_init__(self) -> None:
        """Set original_node_id from metadata if not provided."""
        if not self.original_node_id:
            self.original_node_id = self.metadata.node_id
        if not self.storage_node_id:
            self.storage_node_id = self.metadata.node_id

    def to_dict(self) -> dict[str, Any]:
        """Convert remote checkpoint to dictionary for serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "compressed_model_params": self.compressed_model_params,
            "compression_ratio": self.compression_ratio,
            "compression_method": self.compression_method,
            "storage_node_id": self.storage_node_id,
            "original_node_id": self.original_node_id,
            "is_replica": self.is_replica,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RemoteCheckpoint":
        """Create remote checkpoint from dictionary."""
        return cls(
            metadata=CheckpointMetadata.from_dict(data["metadata"]),
            compressed_model_params=data["compressed_model_params"],
            compression_ratio=data.get("compression_ratio"),
            compression_method=data.get("compression_method"),
            storage_node_id=data.get("storage_node_id", ""),
            original_node_id=data.get("original_node_id", ""),
            is_replica=data.get("is_replica", True),
        )

    def get_size_bytes(self) -> int:
        """Get the size of the remote checkpoint in bytes."""
        return len(self.compressed_model_params)

    def get_tag(self) -> tuple[str, int, float]:
        """
        Get the checkpoint tag for identification.

        Returns:
            Tuple of (node_id, version, timestamp) for tagging.
        """
        return (self.original_node_id, self.metadata.version, self.metadata.timestamp)


@dataclass
class CheckpointInfo:
    """
    Lightweight checkpoint information for listing and querying.

    Contains only metadata without the actual model parameters.
    Useful for discovering available checkpoints without loading full data.

    Attributes:
        metadata: Checkpoint metadata.
        size_bytes: Size of the checkpoint file in bytes.
        filepath: Path to the checkpoint file (if local).
        checkpoint_type: Type of checkpoint ("local", "remote").
        is_available: Whether the checkpoint is currently available.
    """

    metadata: CheckpointMetadata
    size_bytes: int = 0
    filepath: Optional[str] = None
    checkpoint_type: str = "local"
    is_available: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint info to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "size_bytes": self.size_bytes,
            "filepath": self.filepath,
            "checkpoint_type": self.checkpoint_type,
            "is_available": self.is_available,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointInfo":
        """Create checkpoint info from dictionary."""
        return cls(
            metadata=CheckpointMetadata.from_dict(data["metadata"]),
            size_bytes=data.get("size_bytes", 0),
            filepath=data.get("filepath"),
            checkpoint_type=data.get("checkpoint_type", "local"),
            is_available=data.get("is_available", True),
        )

    def get_tag(self) -> tuple[str, int, float]:
        """
        Get the checkpoint tag for identification.

        Returns:
            Tuple of (node_id, version, timestamp) for tagging.
        """
        return (self.metadata.node_id, self.metadata.version, self.metadata.timestamp)

