#
# This file is part of the p2pfl distribution
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

"""Abstract aggregator - STATELESS design."""

from abc import abstractmethod
from collections.abc import Sequence

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel, P2PFLModelDecorator, TreeBasedModel, WeightBasedModel
from p2pfl.utils.node_component import NodeComponent, allow_no_addr_check


class NoModelsToAggregateError(Exception):
    """Exception raised when there are no models to aggregate."""

    pass


class IncompatibleModelError(Exception):
    """Exception raised when a model type is incompatible with the aggregator."""

    pass


class Aggregator(NodeComponent):
    """
    Abstract base class for all aggregators.

    Important:
        We do not recomend to inherit directly from this class. Instead, inherit from:
        - ``WeightAggregator``: For neural network aggregation (FedAvg, etc.)
        - ``TreeAggregator``: For tree ensemble aggregation (FedXgbBagging, etc.)

    Args:
        disable_partial_aggregation: Whether to disable partial aggregation.

    Attributes:
        SUPPORTS_PARTIAL_AGGREGATION: Whether partial aggregation is supported.

    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = False  # Default, subclasses should override

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """Initialize the aggregator."""
        # Initialize instance's partial_aggregation based on the class's support
        self.partial_aggregation: bool = self.__class__.SUPPORTS_PARTIAL_AGGREGATION

        # If the class supports it, allow disabling it for this instance
        if self.partial_aggregation and disable_partial_aggregation:
            self.partial_aggregation = False

        # (address) Super
        NodeComponent.__init__(self)

    @allow_no_addr_check
    @abstractmethod
    def _accepts_model(self, model: P2PFLModel) -> bool:
        """
        Check if this aggregator accepts the given model type.

        Args:
            model: The model to check.

        Returns:
            True if the model is compatible, False otherwise.

        """
        raise NotImplementedError

    @allow_no_addr_check
    def validate_models(self, models: list[P2PFLModel]) -> None:
        """
        Validate that all models are compatible with this aggregator.

        Args:
            models: List of models to validate.

        Raises:
            IncompatibleModelError: If any model is incompatible with this aggregator.

        """
        for model in models:
            if not self._accepts_model(model):
                raise IncompatibleModelError(f"{self.__class__.__name__} is not compatible with {model.__class__.__name__}")

    def aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Validate and aggregate the models.

        Automatically calls ``validate_models()`` before delegating to ``_aggregate()``.

        Args:
            models: List of models to aggregate.

        Returns:
            The aggregated model.

        """
        self.validate_models(models)
        return self._aggregate(models)

    @abstractmethod
    def _aggregate(self, models: list[P2PFLModel]) -> P2PFLModel:
        """
        Implement the actual aggregation logic.

        Override this method in subclasses to define the aggregation algorithm.

        Args:
            models: List of validated models to aggregate.

        Returns:
            The aggregated model.

        """
        raise NotImplementedError

    @staticmethod
    def collect_contributors(models: Sequence[P2PFLModel]) -> list[str]:
        """Collect deduplicated contributors from models, preserving order."""
        seen: set[str] = set()
        contributors: list[str] = []
        for m in models:
            for c in m.get_contributors():
                if c not in seen:
                    seen.add(c)
                    contributors.append(c)
        return contributors

    def get_required_callbacks(self) -> list[str]:
        """
        Get the required callbacks for the aggregation.

        Returns:
            List of required callbacks.

        """
        return []


class WeightAggregator(Aggregator):
    """
    Base class for aggregators that work with neural network models.

    Inherit from this class for aggregators that:
        - Average or combine weight tensors
        - Work with PyTorch, TensorFlow, Flax models
        - Expect ``list[np.ndarray]`` of float32/float64 parameter arrays

    The validation is automatic via the template pattern: ``aggregate()`` calls
    ``validate_models()`` before delegating to ``_aggregate()``.

    Example:
        >>> class MyAggregator(WeightAggregator):
        ...     def _aggregate(self, models):
        ...         # Validation already done by aggregate()
        ...         # ... your averaging logic
        ...         pass

    """

    @allow_no_addr_check
    def _accepts_model(self, model: P2PFLModel) -> bool:
        """Check if the model is a weight-based model (neural network)."""
        return isinstance(_unwrap_model(model), WeightBasedModel)


class TreeAggregator(Aggregator):
    """
    Base class for aggregators that work with tree ensemble models.

    Inherit from this class for aggregators that:
        - Combine trees via bagging, boosting, or cycling
        - Work with XGBoost models
        - Expect serialized tree structures

    The validation is automatic via the template pattern: ``aggregate()`` calls
    ``validate_models()`` before delegating to ``_aggregate()``.

    Example:
        >>> class MyTreeAggregator(TreeAggregator):
        ...     def _aggregate(self, models):
        ...         # Validation already done by aggregate()
        ...         # ... your tree combination logic
        ...         pass

    """

    @allow_no_addr_check
    def _accepts_model(self, model: P2PFLModel) -> bool:
        """Check if the model is a tree-based model (XGBoost)."""
        return isinstance(_unwrap_model(model), TreeBasedModel)


def _unwrap_model(model: P2PFLModel) -> P2PFLModel:
    """Unwrap P2PFLModelDecorator layers to get the underlying model."""
    while isinstance(model, P2PFLModelDecorator):
        model = model._wrapped_model
    return model
