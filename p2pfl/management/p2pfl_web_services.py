#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""
Communication with P2PFL Web Services (via REST API).

.. todo:: Implement batch sending.

.. todo:: Implement get_pending_actions.

.. todo:: Implement unregister_node.

.. todo:: Implement get_experiment_id.
"""

import datetime
from typing import Dict

import requests

##################################
#    P2PFL Web Services (API)    #
##################################


class P2pflWebServicesError(Exception):
    """
    P2PFL Web Services Error.

    Args:
        code: Error code.
        message: Error message.

    """

    def __init__(self, code: int, message: str) -> None:
        """Initialize the error."""
        self.code = code
        self.message = message
        super().__init__(f"Error {code}: {message}")


class P2pflWebServices:
    """
    Class that manages the communication with the p2pfl-web services.

    Args:
        url: The URL of the p2pfl-web services.
        key: The key to access the services.

    """

    def __init__(self, url: str, key: str) -> None:
        """Initialize the p2pfl web services."""
        self.__url = url
        # http warning
        if not url.startswith("https://"):
            print("P2pflWebServices Warning: Connection must be over https, traffic will not be encrypted")
        self.__key = key
        self.node_id: Dict[str, int] = {}

    def __build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers["x-api-key"] = self.__key
        return headers

    def register_node(self, node: str, is_simulated: bool) -> None:
        """
        Register a node.

        Args:
            node: The node address.
            is_simulated: If the node is simulated.

        """
        # Send request
        data = {
            "address": node,
            "is_simulated": is_simulated,
            "creation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            response = requests.post(self.__url + "/node", json=data, headers=self.__build_headers(), timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(node, f"Error registering node: {e}")
            raise e
        # Get node id
        self.node_id[node] = response.json()["node_id"]

    def unregister_node(self, node: str) -> None:
        """
        Unregister a node.

        Args:
            node: The node address.

        """
        print("NOT IMPLEMENTED YET")

    def send_log(self, time: datetime.datetime, node: str, level: int, message: str) -> None:
        """
        Send a log message.

        Args:
            time: The time of the message.
            node: The node address.
            level: The log level.
            message: The message.

        """
        # get node id
        if node not in self.node_id:
            raise ValueError(f"Node {node} not registered")
        node_id = self.node_id[node]

        # Send request
        data = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "node_id": node_id,
            "level": level,
            "message": message,
        }
        try:
            response = requests.post(
                self.__url + "/node-log",
                json=data,
                headers=self.__build_headers(),
                timeout=5,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(node, f"Error logging message: {message}")
            if hasattr(e, "response") and e.response is not None and e.response.status_code == 401:
                print("Please check the API key or the node registration in the p2pfl-web services.")
            raise e

    def send_local_metric(self, exp: str, round: int, metric: str, node: str, value: float, step: int) -> None:
        """
        Send a local metric.

        Args:
            exp: The experiment.
            round: The round.
            metric: The metric.
            node: The node address.
            value: The value.
            step: The step.

        """
        # get node id
        if node not in self.node_id:
            raise ValueError(f"Node {node} not registered")
        node_id = self.node_id[node]

        # get experiment id
        # ------- NOT IMPLEMENTED ----------

        # Send request
        data = {
            "node_id": node_id,
            "exp_id": exp,
            "metric_name": metric,
            "round": round,
            "step": step,
            "value": value,
        }
        try:
            response = requests.post(
                self.__url + "/node-metric/local",
                json=data,
                headers=self.__build_headers(),
                timeout=5,
            )
            response.raise_for_status()
        except Exception as e:
            raise P2pflWebServicesError(response.status_code, response.text) from e

    def send_global_metric(self, exp: str, round: int, metric: str, node: str, value: float):
        """
        Send a local metric.

        Args:
            exp: The experiment.
            round: The round.
            metric: The metric.
            node: The node address.
            value: The value.

        """
        # get node id
        if node not in self.node_id:
            raise ValueError(f"Node {node} not registered")
        node_id = self.node_id[node]

        # get experiment id
        # ------- NOT IMPLEMENTED ----------

        # Send request
        data = {
            "node_id": node_id,
            "exp_id": exp,
            "metric_name": metric,
            "round": round,
            "value": value,
        }
        try:
            response = requests.post(
                self.__url + "/node-metric/global",
                json=data,
                headers=self.__build_headers(),
                timeout=5,
            )
            response.raise_for_status()
        except Exception as e:
            raise P2pflWebServicesError(response.status_code, response.text) from e

    def send_system_metric(self, node: str, metric: str, value: float, time: datetime.datetime):
        """
        Send a metric.

        Args:
            node: The node address.
            metric: The metric.
            value: The value.
            time: The time.

        """
        # get node id
        if node not in self.node_id:
            raise ValueError(f"Node {node} not registered")
        node_id = self.node_id[node]

        # Send request
        data = {
            "node_id": node_id,
            "metric_name": metric,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "value": value,
        }
        try:
            response = requests.post(
                self.__url + "/node-metric/system",
                json=data,
                headers=self.__build_headers(),
                timeout=5,
            )
            response.raise_for_status()
        except Exception as e:
            raise P2pflWebServicesError(response.status_code, response.text) from e

    def get_pending_actions(self):
        """Get pending actions from the p2pfl-web services."""
        raise NotImplementedError
