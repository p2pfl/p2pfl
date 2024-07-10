from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union


class Client(ABC):
    @abstractmethod
    def build_message(
        self, cmd: str, args: Optional[List[str]] = None, round: Optional[int] = None
    ) -> any:
        """
        Build a message to send to the neighbors.

        Args:
            cmd (string): Command of the message.
            args (list): Arguments of the message.
            round (int): Round of the message.

        Returns:
            any: Message to send.
        """
        pass

    @abstractmethod
    def build_weights(
        self,
        cmd: str,
        round: int,
        serialized_model: bytes,
        contributors: Optional[List[str]] = [],
        weight: int = 1,
    ) -> any:
        """
        Build a weight message to send to the neighbors.

        Args:
            round (int): Round of the model.
            serialized_model (bytes): Serialized model.
            contributors (list): List of contributors of the model.
            weight (float): Weight of the model.
        """
        pass

    @abstractmethod
    def send(
        self,
        nei: str,
        msg: Union[any, any],
        create_connection: bool = False,
    ) -> None:
        """
        Send a message to a neighbor.
        """
        pass

    @abstractmethod
    def broadcast(
        self, msg: Dict[str, Union[str, int, List[str]]], node_list: Optional[List[str]] = None
    ) -> None:
        """
        Broadcast a message.
        """
        pass
