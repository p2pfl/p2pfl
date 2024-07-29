#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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

"""Client."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

# TODO: Encapsular msg


class Client(ABC):
    """
    Client interface.

    It is used as a interface to help to decoulple communication protocols based on clien-server architecutes.
    """

    @abstractmethod
    def build_message(self, cmd: str, args: Optional[List[str]] = None, round: Optional[int] = None) -> Any:
        """
        Build a message to send to the neighbors.

        Args:
        ----
            cmd (string): Command of the message.
            args (list): Arguments of the message.
            round (int): Round of the message.

        Returns:
        -------
            any: Message to send.

        """
        pass

    @abstractmethod
    def build_weights(
        self,
        cmd: str,
        round: int,
        serialized_model: bytes,
        contributors: Optional[List[str]] = None,
        weight: int = 1,
    ) -> Any:
        """
        Build a weight message to send to the neighbors.

        Args:
        ----
            cmd (string): Command of the message.
            round (int): Round of the message.
            serialized_model (bytes): Serialized model to send.
            contributors (list): List of contributors.
            weight (int): Weight of the message.

        """
        pass

    @abstractmethod
    def send(
        self,
        nei: str,
        msg: Any,
        create_connection: bool = False,
    ) -> None:
        """Send a message to a neighbor."""
        pass

    @abstractmethod
    def broadcast(
        self,
        msg: Any,
        node_list: Optional[List[str]] = None,
    ) -> None:
        """Broadcast a message."""
        pass
