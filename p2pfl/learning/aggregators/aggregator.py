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

"""Abstract aggregator."""

import threading
from typing import List

from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.settings import Settings


class NoModelsToAggregateError(Exception):
    """Exception raised when there are no models to aggregate."""

    pass


class Aggregator:
    """
    Class to manage the aggregation of models.

    Args:
        node_name: String with the name of the node.

    """

    def __init__(self, node_name: str = "unknown") -> None:
        """Initialize the aggregator."""
        #
        # TODO> REVISAR INICIALIZACIÓN DE ATRIBUTOS RELACIONADAS CON EL AGREGADOR (quiza mejor en state)
        #
        self.node_name = node_name
        self.__train_set: List[str] = []
        self.__models: List[P2PFLModel] = []

        # Locks
        self.__agg_lock = threading.Lock()
        self._finish_aggregation_event = threading.Event()
        self._finish_aggregation_event.set()

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models.

        Args:
            models: Dictionary with the models to aggregate.

        """
        raise NotImplementedError

    def set_nodes_to_aggregate(self, nodes_to_aggregate: List[str]) -> None:
        """
        List with the name of nodes to aggregate. Be careful, by setting new nodes, the actual aggregation will be lost.

        Args:
            nodes_to_aggregate: List of nodes to aggregate. Empty for no aggregation.

        Raises:
            Exception: If the aggregation is running.

        """
        if self._finish_aggregation_event.is_set():
            self.__train_set = nodes_to_aggregate
            self._finish_aggregation_event.clear()
        else:
            raise Exception("It is not possible to set nodes to aggregate when the aggregation is running.")

    def clear(self) -> None:
        """Clear the aggregation (remove trainset and release locks)."""
        with self.__agg_lock:
            self.__train_set = []
            self.__models = []
            self._finish_aggregation_event.set()

    def get_aggregated_models(self) -> List[str]:
        """
        Get the list of aggregated models.

        Returns:
            Name of nodes that colaborated to get the model.

        """
        models_added = []
        for n in self.__models:
            models_added += n.get_contributors()
        return models_added

    def add_model(self, model: P2PFLModel) -> List[str]:
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            model: Model to add.

        Returns:
            List of contributors.

        """
        # Verify that contributors are not empty
        if model.get_contributors() == []:
            logger.debug(self.node_name, "Received a model without a list of contributors.")
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
                        self.node_name,
                        f"🧩 Model added ({models_added}/{ str(len(self.__train_set))}) from {str(model.get_contributors())}",
                    )

                    # Check if all models were added
                    if len(self.get_aggregated_models()) >= len(self.__train_set):
                        self._finish_aggregation_event.set()

                    # Unlock and Return
                    self.__agg_lock.release()
                    return self.get_aggregated_models()
                else:
                    logger.debug(
                        self.node_name,
                        f"Can't add a model from a node ({model.get_contributors()}) that is already in the training set.",
                    )
            else:
                logger.debug(
                    self.node_name,
                    f"Can't add a model from a node ({model.get_contributors()}) that is not in the training test.",
                )
        else:
            logger.debug(self.node_name, "🚫 Received a model when is not needed (already aggregated).")

        # Release and return
        self.__agg_lock.release()
        return []

    def wait_and_get_aggregation(self, timeout: int = Settings.AGGREGATION_TIMEOUT) -> P2PFLModel:
        """
        Wait for aggregation to finish.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Aggregated model.

        Raises:
            Exception: If waiting for an aggregated model and several models were received.

        """
        # Wait for aggregation to finish (then release the lock again)
        event_set = self._finish_aggregation_event.wait(timeout=timeout)
        # Check that the aggregation is finished
        missing_models = self.get_missing_models()
        # Check if aggregation has timed out or event has been set correctly
        if not event_set:
            logger.info(self.node_name, f"⏳ Aggregation wait timed out. Missing models: {missing_models}")
        else:
            if len(missing_models) > 0:
                logger.info(
                    self.node_name,
                    f"❌ Aggregation event set, but missing models:  {missing_models}",
                )
            else:
                logger.info(self.node_name, "🧠 Aggregating models.")

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

    def get_partial_aggregation(self, except_nodes: List[str]) -> P2PFLModel:
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
