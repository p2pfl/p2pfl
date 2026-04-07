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

"""Abstract aggregator."""

import threading
from abc import abstractmethod

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel, TreeBasedModel, WeightBasedModel
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
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
        self.__train_set: list[str] = []  # TODO: Remove the trainset from the state
        self.__models: list[P2PFLModel] = []

        # Initialize instance's partial_aggregation based on the class's support
        self.partial_aggregation: bool = self.__class__.SUPPORTS_PARTIAL_AGGREGATION

        # If the class supports it, allow disabling it for this instance
        if self.partial_aggregation and disable_partial_aggregation:
            self.partial_aggregation = False

        # (addr) Super
        NodeComponent.__init__(self)

        # Locks
        self.__agg_lock = threading.Lock()
        self._finish_aggregation_event = threading.Event()
        self._finish_aggregation_event.set()

        # Unhandled models
        self.__unhandled_models: list[P2PFLModel] = []

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

    def get_required_callbacks(self) -> list[str]:
        """
        Get the required callbacks for the aggregation.

        Returns:
            List of required callbacks.

        """
        return []

    def set_nodes_to_aggregate(self, nodes_to_aggregate: list[str]) -> None:
        """
        List with the name of nodes to aggregate. Be careful, by setting new nodes, the actual aggregation will be lost.

        Args:
            nodes_to_aggregate: List of nodes to aggregate. Empty for no aggregation.

        Raises:
            RuntimeError: If the aggregation is running.

        """
        if not self._finish_aggregation_event.is_set():
            raise RuntimeError("It is not possible to set nodes to aggregate when the aggregation is running.")

        # Start new aggregation
        self.__train_set = nodes_to_aggregate
        self._finish_aggregation_event.clear()
        for m in self.__unhandled_models:
            self.add_model(m)
            # NOTE: Don´t need to send message indicating this aggregations. Self aggregation will be sufficient to notify the network.
        self.__unhandled_models = []

    def clear(self) -> None:
        """Clear the aggregation (remove trainset and release locks)."""
        with self.__agg_lock:
            self.__train_set = []
            self.__models = []
            self.__unhandled_models = []
            self._finish_aggregation_event.set()

    def get_aggregated_models(self) -> list[str]:
        """
        Get the list of aggregated models.

        Returns:
            Name of nodes that colaborated to get the model.

        """
        models_added = []
        for n in self.__models:
            models_added += n.get_contributors()
        return models_added

    def add_model(self, model: P2PFLModel) -> list[str]:
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            model: Model to add.

        Returns:
            List of contributors.

        """
        # Verify that contributors are not empty
        if model.get_contributors() == []:
            logger.debug(self.addr, "Received a model without a list of contributors.")
            self.__agg_lock.release()
            return []

        # Lock
        self.__agg_lock.acquire()

        #
        # TODO: (optimiazacion) Si llega un modelo completamente agregado, se tiene que saltar todo esto
        # TODO: A veces se agregan repetidos
        #

        # Check if aggregation is needed
        if len(self.__train_set) > len(self.get_aggregated_models()):
            # Check if all nodes are in the train_set
            if all(n in self.__train_set for n in model.get_contributors()):
                # Check if any model was added
                any_model_added = any(n in self.get_aggregated_models() for n in model.get_contributors())
                if not any_model_added:
                    # Aggregate model
                    self.__models.append(model)
                    models_added = str(len(self.get_aggregated_models()))
                    logger.info(
                        self.addr,
                        f"Model added ({models_added}/{str(len(self.__train_set))}) from {str(model.get_contributors())}",
                    )
                    # logger.debug(self.addr, f"Models added: {self.get_aggregated_models()}")

                    # Check if all models were added
                    if len(self.get_aggregated_models()) >= len(self.__train_set):
                        self._finish_aggregation_event.set()

                    # Unlock and Return
                    self.__agg_lock.release()
                    return self.get_aggregated_models()
                else:
                    logger.debug(
                        self.addr,
                        f"Can't add a model from a node ({model.get_contributors()}) that is already aggregated.",
                    )
            else:
                logger.debug(
                    self.addr,
                    f"Can't add a model from a node ({model.get_contributors()}) that is not in the training set.",
                )
        else:
            logger.debug(self.addr, "Received a model when is not needed. Saving a iteration to affor bandwith.")
            self.__unhandled_models.append(model)

        # Release and return
        self.__agg_lock.release()
        return []

    def wait_and_get_aggregation(self, timeout: int = Settings.training.AGGREGATION_TIMEOUT, state=None) -> P2PFLModel:
        """
        Wait for aggregation to finish.

        Args:
            timeout: Timeout in seconds.
            state: Optional NodeState to sync train_set from if aggregation times out.

        Returns:
            Aggregated model.

        Raises:
            Exception: If waiting for an aggregated model and several models were received.

        """
        # CRITICAL: Before waiting, check if state.train_set differs and missing models are failed nodes
        # If so, sync immediately to avoid waiting for timeout
        if state is not None and hasattr(state, 'train_set') and state.train_set:
            if set(state.train_set) != set(self.__train_set):
                # Train set has been updated (e.g., nodes failed)
                missing_models_before = self.get_missing_models()
                # Check if all missing models are failed nodes (not in state.train_set)
                failed_nodes = set(missing_models_before) - set(state.train_set)
                if failed_nodes == set(missing_models_before):
                    # All missing models are from failed nodes, sync immediately
                    logger.info(
                        self.addr,
                        f"Immediately syncing aggregator train_set (failed nodes detected): {self.__train_set} -> {state.train_set}"
                    )
                    self.__train_set = state.train_set.copy()
                    # Recalculate missing models
                    missing_models_after = self.get_missing_models()
                    if not missing_models_after:
                        # All required models are present, trigger aggregation completion
                        logger.info(self.addr, "All models present after sync. Triggering aggregation completion.")
                        self._finish_aggregation_event.set()
        
        # Wait for aggregation to finish (then release the lock again)
        event_set = self._finish_aggregation_event.wait(timeout=timeout)
        # Check that the aggregation is finished
        missing_models = self.get_missing_models()
        # Check if aggregation has timed out or event has been set correctly
        if not event_set:
            logger.info(self.addr, f"Aggregation wait timed out. Missing models: {missing_models}")
            
            # CRITICAL: If timeout occurred and state is provided, sync train_set
            # This handles the case where nodes failed during aggregation and state.train_set
            # was updated but aggregator's train_set wasn't (because aggregation was running)
            if state is not None and hasattr(state, 'train_set') and state.train_set:
                if set(state.train_set) != set(self.__train_set):
                    logger.info(
                        self.addr,
                        f"Syncing aggregator train_set from state: {self.__train_set} -> {state.train_set}"
                    )
                    # Update train_set directly (don't call set_nodes_to_aggregate as it would reset aggregation)
                    self.__train_set = state.train_set.copy()
                    # Recalculate missing models with updated train_set
                    missing_models = self.get_missing_models()
                    if missing_models:
                        logger.info(self.addr, f"After sync, still missing models: {missing_models}")
        else:
            if len(missing_models) > 0:
                logger.info(
                    self.addr,
                    f"ERROR: Aggregation event set, but missing models:  {missing_models}",
                )
            else:
                logger.info(self.addr, "Aggregating models.")

        # Notify node
        return self.aggregate(self.__models)

    def get_missing_models(self) -> set:
        """
        Obtain missing models for the aggregation.

        Returns:
            A set of missing models.

        """
        agg_models = []
        for m in self.__models:
            agg_models += m.get_contributors()
        missing_models = set(self.__train_set) - set(agg_models)
        return missing_models

    def __get_partial_aggregation(self, except_nodes: list[str]) -> P2PFLModel:
        """
        Obtain a partial aggregation.

        Args:
            except_nodes: List of nodes to exclude from the aggregation.

        Return:
            Aggregated model, nodes aggregated and aggregation weight.

        """
        models_to_aggregate = []
        for m in self.__models.copy():
            if all(n not in except_nodes for n in m.get_contributors()):
                models_to_aggregate.append(m)

        return self.aggregate(models_to_aggregate)

    def __get_remaining_model(self, except_nodes) -> P2PFLModel:
        """
        Obtain a random model from the remaining nodes.

        Args:
            except_nodes: List of nodes to exclude from the aggregation.

        Return:
            Aggregated model, nodes aggregated and aggregation weight.

        """
        for m in self.__models.copy():
            contributors = m.get_contributors()
            if all(n not in except_nodes for n in contributors):
                return m
        raise NoModelsToAggregateError("No remaining models available for aggregation.")

    def get_model(self, except_nodes) -> P2PFLModel:
        """
        Get corresponding aggregation depending if aggregator supports partial aggregations.

        Args:
            except_nodes: List of nodes to exclude from the aggregation.

        """
        if self.partial_aggregation:
            return self.__get_partial_aggregation(except_nodes)
        else:
            return self.__get_remaining_model(except_nodes)


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
        return isinstance(model, WeightBasedModel)


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
        return isinstance(model, TreeBasedModel)
