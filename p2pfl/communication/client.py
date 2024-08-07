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

"""Protocol agnostic client."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class Client(ABC):
    """
    Client interface.

    It is used as a interface to help to decoulple communication protocols based on clien-server architecutes.

    .. todo:: Encapsulate msg.
    """

    @abstractmethod
    def build_message(self, cmd: str, args: Optional[List[str]] = None, round: Optional[int] = None) -> Any:
        """
        Build a message to send to the neighbors.

        Args:
            cmd: Command of the message.
            args: Arguments of the message.
            round: Round of the message.

        Returns:
            Message to send.

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
            cmd: Command of the message.
            round: Round of the message.
            serialized_model: Serialized model to send.
            contributors: List of contributors.
            weight: Weight of the message.

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
