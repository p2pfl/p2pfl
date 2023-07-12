#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/federated_learning_p2p).
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

import threading
import logging
from p2pfl.settings import Settings
from p2pfl.utils.observer import Events, Observable


"""
ESTO DEJA DE SER UN THREAD Y DEJA DE NOTIFICAR
"""

"""
METER EL STOP PARA QUE NO SE QUEDEN COLGADOS LOS LOCKS
"""


class Aggregator:
    """
    Class to manage the aggregation of models.

    Args:
        node_name: (str): String with the name of the node.
    """

    def __init__(self, node_name="unknown"):
        self.node_name = node_name
        threading.Thread.__init__(self, name="aggregator-" + node_name)
        self.daemon = True
        Observable.__init__(self)
        self.__train_set = []
        self.__waiting_aggregated_model = False  # POCO ELEGANTE -> REVISARLO
        self.__models = {}
        self.__agg_lock = threading.Lock()

        # Wait lock
        self.__finish_aggregation_lock = threading.Lock()

    def aggregate(self, models):
        """
        Aggregate the models.
        """
        print("Not implemented")

    def set_nodes_to_aggregate(self, l):
        """
        List with the name of nodes to aggregate. Be careful, by setting new nodes, the actual aggregation will be lost.

        Args:
            l: List of nodes to aggregate. Empty for no aggregation.
        """
        if not self.__finish_aggregation_lock.locked():
            self.__train_set = l
            self.__models = {}
            self.__finish_aggregation_lock.acquire()
        else:
            raise Exception(
                "It is not possible to set nodes to aggregate when the aggregation is running."
            )

    def clear(self):
        self.__agg_lock.acquire()
        self.__train_set = []
        try:
            self.__finish_aggregation_lock.release()
        except:
            pass
        self.__agg_lock.release()

    """
    UNICAMENTE RECIBIRÁ UN MODELO Y LUEGO LO DIFUNDIRÁ
    """

    def set_waiting_aggregated_model(self):
        """
        Indicates that the node is waiting for an aggregation. It won't participate in aggregation process.
        """
        self.__waiting_aggregated_model = True
        self.__finish_aggregation_lock.acquire(
            timeout=Settings.AGGREGATION_TIMEOUT
        )  # REVISAR

    def get_agregated_models(self):
        """
        Get the list of aggregated models.

        Returns:
            Name of nodes that colaborated to get the model.
        """
        return list(self.__models.keys())

    def add_model(self, model, nodes, weight):
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            model: Model to add.
            nodes: Nodes that colaborated to get the model.
            weight: Number of samples used to get the model.
        """
        if self.__waiting_aggregated_model and self.__models == {}:
            logging.info(f"({self.node_name}) Received an aggregated model.")
            self.__models = {"_": (model, 1)}  # AMOUNT OF SAMPLES IS UNNECESSARY
            self.__finish_aggregation_lock.release()  # REVISAR ------
            return None

        else:
            nodes = list(nodes)
            if nodes is not []:
                self.__agg_lock.acquire()

                # Get a list of nodes added
                models_added = [n.split() for n in list(self.__models.keys())]
                models_added = [
                    element for sublist in models_added for element in sublist
                ]  # Flatten list

                # Check if aggregation is needed
                if len(self.__train_set) > len(models_added):
                    # Check if all nodes are in the train_set
                    if all([n in self.__train_set for n in nodes]):
                        # Check if all node models are not aggregated
                        if all([n not in models_added for n in nodes]):
                            # Aggregate model
                            self.__models[" ".join(nodes)] = (model, weight)
                            logging.info(
                                f"({self.node_name}) Model added ({str(len(models_added) + len(nodes))}/{ str(len(self.__train_set))}) from {str(nodes)}"
                            )

                            # Check if all models have been added and unlock waiters
                            models_added = [
                                nodes.split() for nodes in list(self.__models.keys())
                            ]
                            models_added = [
                                element
                                for sublist in models_added
                                for element in sublist
                            ]  # Flatten list
                            if len(models_added) >= len(self.__train_set):
                                self.__finish_aggregation_lock.release()

                            # Build response
                            response = models_added + nodes

                            # Unloock
                            self.__agg_lock.release()

                            return response

                        else:
                            logging.debug(
                                f"({self.node_name}) Can't add a model that has already been added {nodes}"
                            )
                    else:
                        logging.debug(
                            f"({self.node_name}) Can't add a model from a node ({nodes}) that is not in the training test."
                        )
                else:
                    logging.debug(
                        f"({self.node_name}) Received a model when is not needed."
                    )
                self.__agg_lock.release()
            else:
                logging.info(f"({self.node_name}) Model contributors cannot be empty.")

            return None

    def wait_and_get_aggregation(self, timeout=Settings.AGGREGATION_TIMEOUT):
        """
        Wait for aggregation to finish.
        """

        # Wait for aggregation to finish (then release the lock again)
        self.__finish_aggregation_lock.acquire(timeout=timeout)
        try:
            self.__finish_aggregation_lock.release()
        except:
            pass

        # If awaiting for an aggregated model, return it
        if self.__waiting_aggregated_model:
            if len(self.__models) == 1:
                return list(self.__models.values())[0][0]
            print(f"keys: {self.__models.keys()}")
            return None
        # Start aggregation
        n_model_aggregated = sum(
            [len(nodes.split()) for nodes in list(self.__models.keys())]
        )

        # Timeout / All models
        if n_model_aggregated != len(self.__train_set):
            logging.info(
                f"({self.node_name}) Aggregating models, timeout reached. Missing models: {set(self.__train_set) - set(self.__models.keys())}"
            )
        else:
            logging.info(f"({self.node_name}) Aggregating models.")

        # Notify node
        return self.aggregate(self.__models)

    def get_partial_aggregation(self, except_nodes):
        dict_aux = {}
        nodes_aggregated = []
        aggregation_weight = 0
        models = self.__models.copy()
        for n, (m, s) in list(models.items()):
            splited_nodes = n.split()
            if all([n not in except_nodes for n in splited_nodes]):
                dict_aux[n] = (m, s)
                nodes_aggregated += splited_nodes
                aggregation_weight += s

        # If there are no models to aggregate
        if len(dict_aux) == 0:
            return None, None, None

        return (self.aggregate(dict_aux), nodes_aggregated, aggregation_weight)
