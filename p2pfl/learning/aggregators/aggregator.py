#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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

import contextlib
import threading
from typing import Dict, List, Optional, Tuple, Union

import torch

from p2pfl.management.logger import logger
from p2pfl.settings import Settings


class NoModelsToAggregateError(Exception):
    """Exception raised when there are no models to aggregate."""

    pass


class Aggregator:
    """
    Class to manage the aggregation of models. Aggregate not implemented, strategy pattern.

    Args:
        node_name: String with the name of the node.

    """

    def __init__(self, node_name: str = "unknown") -> None:
        """Initialize the aggregator."""
        self.node_name = node_name
        self.__train_set: List[str] = []
        self.__waiting_aggregated_model = False
        self.__models: Dict[str, Tuple[Dict[str, torch.Tensor], int]] = {}

        # Locks
        self.__agg_lock = threading.Lock()
        self.__finish_aggregation_lock = threading.Lock()

    def aggregate(self, models: Dict[str, Tuple[Dict[str, torch.Tensor], int]]):
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
        if not self.__finish_aggregation_lock.locked():
            self.__train_set = nodes_to_aggregate
            self.__finish_aggregation_lock.acquire(timeout=Settings.AGGREGATION_TIMEOUT)
        else:
            raise Exception("It is not possible to set nodes to aggregate when the aggregation is running.")

    def set_waiting_aggregated_model(self, nodes: List[str]) -> None:
        """
        Indicate that the node is waiting for a completed aggregation. It won't participate in aggregation process.

        Args:
            nodes: List of nodes to aggregate. Empty for no aggregation.

        """
        self.set_nodes_to_aggregate(nodes)
        self.__waiting_aggregated_model = True

    def clear(self) -> None:
        """Clear the aggregation (remove trainset and release locks)."""
        self.__agg_lock.acquire()
        self.__train_set = []
        self.__models = {}
        with contextlib.suppress(Exception):
            self.__finish_aggregation_lock.release()
        self.__agg_lock.release()

    def get_aggregated_models(self) -> List[str]:
        """
        Get the list of aggregated models.

        Returns
            Name of nodes that colaborated to get the model.

        """
        # Get a list of nodes added
        models_added = [n.split() for n in list(self.__models.keys())]
        # Flatten list
        return [element for sublist in models_added for element in sublist]

    def add_model(self, model: Dict[str, torch.Tensor], contributors: List[str], weight: int) -> List[str]:
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            model: Model to add.
            contributors: List of contributors.
            weight: Weight of the model.

        Returns:
            List of contributors.

        """
        nodes = list(contributors)

        # Verify that contributors are not empty
        if contributors == []:
            logger.debug(self.node_name, "Received a model without a list of contributors.")
            self.__agg_lock.release()
            return []

        # Diffusion / Aggregation
        if self.__waiting_aggregated_model and self.__models == {}:
            if set(contributors) == set(self.__train_set):
                logger.info(self.node_name, "Received an aggregated model.")
                self.__models = {}
                self.__models = {" ".join(nodes): (model, 1)}
                self.__waiting_aggregated_model = False
                self.__finish_aggregation_lock.release()
                return contributors

        else:
            self.__agg_lock.acquire()

            # Check if aggregation is needed
            if len(self.__train_set) > len(self.get_aggregated_models()):
                # Check if all nodes are in the train_set
                if all(n in self.__train_set for n in nodes):
                    # Check if the model is a full/partial aggregation
                    if len(nodes) == len(self.__train_set):
                        self.__models = {}
                        self.__models[" ".join(nodes)] = (model, weight)
                        models_added = str(len(self.get_aggregated_models()))
                        logger.info(
                            self.node_name,
                            f"Model added ({models_added}/{ str(len(self.__train_set))}) from {str(nodes)}",
                        )
                        # Finish agg
                        self.__finish_aggregation_lock.release()
                        # Unlock and Return
                        self.__agg_lock.release()
                        return self.get_aggregated_models()

                    elif all(n not in self.get_aggregated_models() for n in nodes):
                        # Aggregate model
                        self.__models[" ".join(nodes)] = (model, weight)
                        models_added = str(len(self.get_aggregated_models()))
                        logger.info(
                            self.node_name,
                            f"Model added ({models_added}/{ str(len(self.__train_set))}) from {str(nodes)}",
                        )

                        # Check if all models were added
                        if len(self.get_aggregated_models()) >= len(self.__train_set):
                            self.__finish_aggregation_lock.release()

                        # Unloock and Return
                        self.__agg_lock.release()
                        return self.get_aggregated_models()

                    else:
                        logger.debug(
                            self.node_name,
                            f"Can't add a model that has already been added {nodes} / {self.get_aggregated_models()}",
                        )
                else:
                    logger.debug(
                        self.node_name,
                        f"Can't add a model from a node ({nodes}) that is not in the training test.",
                    )
            else:
                logger.debug(self.node_name, "Received a model when is not needed.")
            self.__agg_lock.release()
        return []

    def wait_and_get_aggregation(self, timeout: Optional[int] = None) -> Union[dict, None]:
        """
        Wait for aggregation to finish.

        Args:
            timeout: Timeout in seconds.

        Returns:
            Aggregated model.

        Raises:
            Exception: If waiting for an aggregated model and several models were received.

        """
        if timeout is None:
            timeout = Settings.AGGREGATION_TIMEOUT
        # Wait for aggregation to finish (then release the lock again)
        self.__finish_aggregation_lock.acquire(timeout=timeout)
        with contextlib.suppress(Exception):
            self.__finish_aggregation_lock.release()

        # If awaiting for an aggregated model, return it
        if self.__waiting_aggregated_model:
            if len(self.__models) == 1:
                return list(self.__models.values())[0][0]
            elif len(self.__models) == 0:
                logger.info(
                    self.node_name,
                    "Timeout reached by waiting for an aggregated model. Continuing with the local model.",
                )
            raise Exception(f"Waiting for an an aggregated but several models were received: {self.__models.keys()}")
        # Start aggregation
        n_model_aggregated = sum([len(nodes.split()) for nodes in list(self.__models.keys())])

        # Timeout / All models
        if n_model_aggregated != len(self.__train_set):
            missing_models = set(self.__train_set) - set(self.__models.keys())
            logger.info(
                self.node_name,
                f"Aggregating models, timeout reached. Missing models: {missing_models}",
            )
        else:
            logger.info(self.node_name, "Aggregating models.")

        # Notify node
        return self.aggregate(self.__models)

    def get_partial_aggregation(
        self, except_nodes: List[str]
    ) -> Tuple[Union[dict, None], Union[List[str], None], Union[int, None]]:
        """
        Obtain a partial aggregation.

        Args:
            except_nodes: List of nodes to exclude from the aggregation.

        Returns:
            Aggregated model, nodes aggregated and aggregation weight.

        """
        dict_aux = {}
        nodes_aggregated = []
        aggregation_weight = 0
        models = self.__models.copy()
        for n, (m, s) in list(models.items()):
            splited_nodes = n.split()
            if all(n not in except_nodes for n in splited_nodes):
                dict_aux[n] = (m, s)
                nodes_aggregated += splited_nodes
                aggregation_weight += s

        # If there are no models to aggregate
        if len(dict_aux) == 0:
            return None, None, None

        return (
            self.aggregate(dict_aux),
            nodes_aggregated,
            aggregation_weight,
        )
