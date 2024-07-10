from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

from p2pfl.commands.command import Command

# Define type aliases for clarity
CandidateCondition = Callable[[str], bool]
StatusFunction = Callable[[str], Any]
ModelFunction = Callable[[str], Tuple[Any, List[str], int]]


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
    def send(self, nei: str, message: any, node_list: Optional[List[str]] = None) -> None:
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

    @abstractmethod
    def gossip_weights(
        self,
        early_stopping_fn: Callable[[], bool],
        get_candidates_fn,
        status_fn: StatusFunction,
        model_fn: ModelFunction,
        period: float,
        create_connection: bool = False,
    ) -> None:
        pass
