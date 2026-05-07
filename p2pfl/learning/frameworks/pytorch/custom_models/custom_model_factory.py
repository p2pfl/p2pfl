"""PyTorch Lightning custom model factory."""

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger


class LightningCustomModelFactory:
    """Factory for creating custom PyTorch Lightning models."""

    @classmethod
    def create_model(cls, type: str, model: P2PFLModel) -> P2PFLModel:
        """
        Create a custom model.

        Args:
            type: The type of model.
            model: The model.

        Returns:
            The custom model.

        """
        if type == "AsyDFL":
            from p2pfl.learning.frameworks.pytorch.custom_models.asydfl_model import AsyDFLLightningModel

            return model if isinstance(model, AsyDFLLightningModel) else AsyDFLLightningModel(model)
        else:
            logger.error("LightningCustomModelFactory", f"Unsupported type: {type}")
            raise ValueError(f"Unsupported type: {type}")
