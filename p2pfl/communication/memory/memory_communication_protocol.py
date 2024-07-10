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

from typing import Dict, List, Optional, Union, Callable, Any, Tuple

from p2pfl.communication.gossiper import Gossiper
from p2pfl.communication.heartbeater import Heartbeater
from p2pfl.communication.memory.memory_client import InMemoryClient
from p2pfl.communication.memory.memory_server import InMemoryServer
from p2pfl.communication.memory.memory_neighbors import InMemoryNeighbors
from p2pfl.communication.communication_protocol import CommunicationProtocol
from p2pfl.commands.command import Command
from p2pfl.commands.heartbeat_command import HeartbeatCommand
from p2pfl.settings import Settings

# Define type aliases for clarity
CandidateCondition = Callable[[str], bool]
StatusFunction = Callable[[str], Any]
ModelFunction = Callable[[str], Tuple[Any, List[str], int]]


class InMemoryCommunicationProtocol(CommunicationProtocol):
    def __init__(self, addr: str = "address", commands: List[Command] = []) -> None:
        # Address
        self.addr = addr

        # Neighbors
        self._neighbors = InMemoryNeighbors(self.addr)
        # Client
        self._client = InMemoryClient(self.addr, self._neighbors)
        # Gossip
        self._gossiper = Gossiper(self.addr, self._client)
        # Server
        self._server = InMemoryServer(self.addr, self._gossiper, self._neighbors, commands)
        # Hearbeat
        self._heartbeater = Heartbeater(self.addr, self._neighbors, self._client)
        # Commands
        self._server.add_command(HeartbeatCommand(self._heartbeater))
        self._server.add_command(commands)

    def get_address(self) -> str:
        return self.addr

    def start(self) -> None:
        self._server.start()
        self._heartbeater.start()
        self._gossiper.start()

    def stop(self) -> None:
        self._server.stop()
        self._heartbeater.stop()
        self._gossiper.stop()
        self._neighbors.clear_neighbors()

    def add_command(self, cmds: Union[Command, List[Command]]) -> None:
        self._server.add_command(cmds)

    def connect(self, addr: str, non_direct: bool = False) -> None:
        return self._neighbors.add(addr, non_direct=non_direct)

    def disconnect(self, nei: str, disconnect_msg: bool = True) -> None:
        self._neighbors.remove(nei, disconnect_msg=disconnect_msg)

    def build_msg(self, cmd: str, args: List[str] = [], round: Optional[int] = None) -> any:
        return self._client.build_message(cmd, args, round)

    def build_weights(
        self,
        cmd: str,
        round: int,
        serialized_model: bytes,
        contributors: Optional[List[str]] = [],
        weight: int = 1,
    ) -> any:
        return self._client.build_weights(cmd, round, serialized_model, contributors, weight)

    def send(
        self,
        nei: str,
        msg: Union[
            Dict[str, Union[str, int, List[str], bytes]],
            Dict[str, Union[str, int, bytes, List[str]]],
        ],
    ) -> None:
        self._client.send(nei, msg)

    def broadcast(
        self, msg: Dict[str, Union[str, int, List[str]]], node_list: Optional[List[str]] = None
    ) -> None:
        self._client.broadcast(msg, node_list)

    def get_neighbors(self, only_direct: bool = False) -> List[str]:
        return self._neighbors.get_all(only_direct)

    def wait_for_termination(self) -> None:
        self._server.wait_for_termination()

    def gossip_weights(
        self,
        early_stopping_fn: Callable[[], bool],
        get_candidates_fn,
        status_fn: StatusFunction,
        model_fn: ModelFunction,
        period: float = Settings.GOSSIP_MODELS_PERIOD,
        create_connection: bool = False,
    ) -> None:
        self._gossiper.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
            period,
            create_connection,
        )
