from abc import ABC, abstractmethod
from typing import List, Optional, Union
from p2pfl.commands.command import Command


class CommunicationProtocol(ABC):

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def add_command(self, cmds: Union[Command, List[Command]]) -> None:
        pass

    @abstractmethod
    def build_msg(self, msg: str, args: List[str], round: Optional[int] = None) -> str:
        pass

    @abstractmethod
    def build_weights(self, weights: any, PORHACER) -> any:
        pass

    @abstractmethod
    def send(
        self, nei: str, message: any, node_list: Optional[List[str]] = None
    ) -> None:
        pass

    @abstractmethod
    def broadcast(self, message: any) -> None:
        pass

    @abstractmethod
    def connect(self, addr: str, non_direct: bool = False) -> None:
        pass

    @abstractmethod
    def disconnect(self, nei: str) -> None:
        pass

    @abstractmethod
    def get_neighbors(self) -> None:
        pass

    @abstractmethod
    def get_address(self) -> None:
        pass

    @abstractmethod
    def wait_for_termination(self) -> None:
        pass
