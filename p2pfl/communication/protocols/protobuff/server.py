#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
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

"""Protobuff server."""

import asyncio
import traceback
from abc import ABC, abstractmethod
from typing import Any

import google.protobuf.empty_pb2
import grpc

from p2pfl.communication.commands.command import Command
from p2pfl.communication.protocols.protobuff.gossiper import Gossiper
from p2pfl.communication.protocols.protobuff.neighbors import Neighbors
from p2pfl.communication.protocols.protobuff.proto import node_pb2, node_pb2_grpc
from p2pfl.management.logger import logger
from p2pfl.utils.node_component import NodeComponent, allow_no_addr_check

_MSG_BUFFER_MAX_SIZE = 200


class ProtobuffServer(ABC, node_pb2_grpc.NodeServicesServicer, NodeComponent):
    """
    Implementation of the server side logic of PROTOBUFF communication protocol.

    Args:
        addr: Address of the server.
        gossiper: Gossiper instance.
        neighbors: Neighbors instance.
        commands: List of commands to be executed by the server.

    """

    def __init__(
        self,
        gossiper: Gossiper,
        neighbors: Neighbors,
        commands: list[Command] | None = None,
    ) -> None:
        """Initialize the GRPC server."""
        # Message handlers
        if commands is None:
            commands = []
        self.__commands = {c.get_name(): c for c in commands}

        # (addr) Super
        NodeComponent.__init__(self)

        # Gossiper
        self._gossiper = gossiper

        # Neighbors
        self._neighbors = neighbors

        # Background tasks
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Buffer for messages that arrive before their command is registered
        self._pending_msgs_buffer: list[node_pb2.RootMessage] = []

    ####
    # Management
    ####

    @abstractmethod
    async def start(self, wait: bool = False) -> None:
        """
        Start the server.

        Args:
            wait: If True, wait for termination.

        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the server."""
        pass

    async def cleanup_tasks(self) -> None:
        """Cancel all background tasks and clear the pending message buffer."""
        for task in self._background_tasks:
            task.cancel()
        self._background_tasks.clear()
        self._pending_msgs_buffer.clear()

    @abstractmethod
    async def wait_for_termination(self) -> None:
        """Wait for termination."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the server is running.

        Returns:
            True if the server is running, False otherwise.

        """
        pass

    ####
    # Service Implementation (server logic on protobuff)
    ####

    async def handshake(self, request: node_pb2.HandShakeRequest, _: grpc.aio.ServicerContext) -> node_pb2.ResponseMessage:
        """
        Service. It is called when a node connects to another.

        Args:
            request: Request message.
            _: Context.

        """
        if await self._neighbors.add(request.addr, non_direct=False, handshake=False):
            return node_pb2.ResponseMessage()
        else:
            return node_pb2.ResponseMessage(error="Cannot add the node (duplicated or wrong direction)")

    async def disconnect(self, request: node_pb2.HandShakeRequest, _: grpc.aio.ServicerContext) -> google.protobuf.empty_pb2.Empty:
        """
        Service. It is called when a node disconnects from another.

        Args:
            request: Request message.
            _: Context.

        """
        await self._neighbors.remove(request.addr, disconnect_msg=False)
        return google.protobuf.empty_pb2.Empty()

    async def send(self, request: node_pb2.RootMessage, _: grpc.aio.ServicerContext) -> node_pb2.ResponseMessage:
        """
        Service. Handles both regular messages and model weights.

        Args:
            request: The RootMessage containing either a Message or Weights payload.
            _: Context.

        """
        # If message already processed, return
        if request.HasField("gossip_message") and not await self._gossiper.check_and_set_processed(request):
            return node_pb2.ResponseMessage()

        # Log
        package_type = "message" if request.HasField("gossip_message") else "weights"
        # Estimate size without full serialization to avoid multi-MB temporary allocations
        package_size = request.ByteSize()
        # Pass None for negative rounds, the logger will handle it
        round_num = request.round if request.round >= 0 else None
        logger.log_communication(
            self.address,
            "received",
            request.cmd,
            request.source,
            package_type,
            package_size,
            round_num,
        )

        # Process message/model
        cmd_out: str | None = None
        if request.cmd in self.__commands:
            try:
                if request.HasField("gossip_message"):
                    # Gossip messages are fire-and-forget (no response needed)
                    task = asyncio.create_task(
                        self.__commands[request.cmd].execute(request.source, request.round, *request.gossip_message.args)
                    )
                    self._track_background_task(task, request.cmd)
                elif request.HasField("direct_message"):
                    # Direct messages may expect responses - await them
                    result = await self.__commands[request.cmd].execute(request.source, request.round, *request.direct_message.args)
                    if result is not None:
                        cmd_out = str(result)
                elif request.HasField("weights"):
                    # Weights are fire-and-forget (no response needed)
                    task = asyncio.create_task(
                        self.__commands[request.cmd].execute(
                            request.source,
                            request.round,
                            weights=request.weights.weights,
                            contributors=request.weights.contributors,
                            num_samples=request.weights.num_samples,
                        )
                    )
                    self._track_background_task(task, request.cmd)
                else:
                    error_text = f"Error while processing command: {request.cmd}: No message or weights."
                    logger.error(self.address, error_text)
                    return node_pb2.ResponseMessage(error=error_text)
            except Exception as e:
                error_text = f"Error while processing command: {request.cmd}. {type(e).__name__}: {e}"
                logger.error(self.address, error_text + f"\n{traceback.format_exc()}")
                return node_pb2.ResponseMessage(error=error_text)
        else:
            # Buffer unknown messages for deferred replay when the command is registered
            if len(self._pending_msgs_buffer) < _MSG_BUFFER_MAX_SIZE:
                self._pending_msgs_buffer.append(request)
                logger.debug(self.address, f"📦 Buffered unknown command: {request.cmd} from {request.source}")
            else:
                logger.warning(self.address, f"Message buffer full, dropping unknown command: {request.cmd} from {request.source}")
            return node_pb2.ResponseMessage()

        # If message gossip
        if request.HasField("gossip_message") and request.gossip_message.ttl > 0:
            # Update ttl and gossip
            request.gossip_message.ttl -= 1
            await self._gossiper.add_message(request)

        return node_pb2.ResponseMessage(response=cmd_out)

    def _track_background_task(self, task: asyncio.Task[Any], cmd_name: str) -> None:
        """Track a background task and log any exceptions it raises."""
        self._background_tasks.add(task)

        def _on_task_done(t: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(t)
            if not t.cancelled() and t.exception() is not None:
                logger.error(self.address, f"Background command '{cmd_name}' failed: {t.exception()}")

        task.add_done_callback(_on_task_done)

    ####
    # Commands
    ####

    @allow_no_addr_check
    def add_command(self, cmds: Command | list[Command]) -> None:
        """
        Add a command.

        Args:
            cmds: Command or list of commands to be added.

        """
        if isinstance(cmds, list):
            new_names: set[str] = set()
            for cmd in cmds:
                self.__commands[cmd.get_name()] = cmd
                new_names.add(cmd.get_name())
        elif isinstance(cmds, Command):
            self.__commands[cmds.get_name()] = cmds
            new_names = {cmds.get_name()}
        else:
            raise Exception("Command not valid")

        # Replay any buffered messages that match the newly registered commands
        if self._pending_msgs_buffer:
            self._replay_buffered_messages(new_names)

    def _replay_buffered_messages(self, new_names: set[str]) -> None:
        """
        Replay buffered messages that match newly registered command names.

        Args:
            new_names: Set of command names that were just registered.

        """
        remaining: list[node_pb2.RootMessage] = []
        for msg in self._pending_msgs_buffer:
            if msg.cmd not in new_names:
                remaining.append(msg)
                continue

            logger.debug(self.address, f"📦 Replaying buffered command: {msg.cmd} from {msg.source}")

            if msg.HasField("gossip_message"):
                # Execute the command
                task = asyncio.create_task(self.__commands[msg.cmd].execute(msg.source, msg.round, *msg.gossip_message.args))
                self._track_background_task(task, msg.cmd)
                # Forward via gossip if TTL allows
                if msg.gossip_message.ttl > 0:
                    msg.gossip_message.ttl -= 1
                    task = asyncio.create_task(self._gossiper.add_message(msg))
                    self._track_background_task(task, f"{msg.cmd}_gossip_forward")
            elif msg.HasField("weights"):
                # Weights are fire-and-forget
                task = asyncio.create_task(
                    self.__commands[msg.cmd].execute(
                        msg.source,
                        msg.round,
                        weights=msg.weights.weights,
                        contributors=msg.weights.contributors,
                        num_samples=msg.weights.num_samples,
                    )
                )
                self._track_background_task(task, msg.cmd)
            elif msg.HasField("direct_message"):
                # Direct messages: response is lost (caller already got empty response)
                task = asyncio.create_task(self.__commands[msg.cmd].execute(msg.source, msg.round, *msg.direct_message.args))
                self._track_background_task(task, msg.cmd)

        self._pending_msgs_buffer = remaining

    @allow_no_addr_check
    def remove_command(self, cmds: str | Command | list[str | Command]) -> None:
        """
        Remove a command.

        Args:
            cmds: Command name, Command instance, or list of either.

        """
        if isinstance(cmds, list):
            for cmd in cmds:
                name = cmd if isinstance(cmd, str) else cmd.get_name()
                self.__commands.pop(name, None)
        elif isinstance(cmds, str):
            self.__commands.pop(cmds, None)
        elif isinstance(cmds, Command):
            self.__commands.pop(cmds.get_name(), None)
        else:
            raise Exception("Command not valid")
