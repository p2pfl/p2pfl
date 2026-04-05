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

"""Protocol agnostic gossiper."""

import asyncio
import contextlib
import random
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from p2pfl.communication.protocols.protobuff.client import ProtobuffClient
from p2pfl.communication.protocols.protobuff.neighbors import Neighbors
from p2pfl.communication.protocols.protobuff.proto import node_pb2
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.utils.node_component import NodeComponent


_GOSSIP_PENDING_MAX = 500  # Drop oldest gossip messages when queue exceeds this size


class Gossiper(NodeComponent):
    """Async-compatible Gossiper that spreads messages and models to neighbors."""

    def __init__(
        self,
        neighbors: Neighbors,
        build_msg: Callable[..., node_pb2.RootMessage],
        period: float | None = None,
        messages_per_period: int | None = None,
    ) -> None:
        """Initialize the gossiper."""
        super().__init__()
        self._neighbors = neighbors
        self.period = period or Settings.gossip.PERIOD
        self.messages_per_period = messages_per_period or Settings.gossip.MESSAGES_PER_PERIOD

        # State — OrderedDict for O(1) lookup and FIFO eviction
        self._processed_messages: OrderedDict[int, None] = OrderedDict()
        self._pending_msgs: list[tuple[node_pb2.RootMessage, list[ProtobuffClient]]] = []

        # Concurrency
        self._processed_messages_lock = asyncio.Lock()
        self._pending_msgs_lock = asyncio.Lock()
        self._terminate_event = asyncio.Event()

        self._task: asyncio.Task | None = None
        self.name = "gossiper-async"

        # Build msgs
        self.build_msg_fn = build_msg

    def set_address(self, address: str) -> str:
        """Assign address and update gossiper thread name."""
        address = super().set_address(address)
        self.name = f"gossiper-async-{address}"
        return address

    async def start(self) -> None:
        """Launch the gossiping task."""
        logger.info(self.address, "🏁 Starting gossiper...")
        self._terminate_event.clear()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Signal the gossiper to stop and wait for completion."""
        logger.info(self.address, "🛑 Stopping gossiper...")
        self._terminate_event.set()
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        # Free queued messages to release protobuf memory
        async with self._pending_msgs_lock:
            self._pending_msgs.clear()

    async def add_message(self, msg: node_pb2.RootMessage) -> None:
        """Queue a message to be gossiped to all direct neighbors."""
        async with self._pending_msgs_lock:
            # Evict oldest messages when queue is full
            if len(self._pending_msgs) >= _GOSSIP_PENDING_MAX:
                self._pending_msgs = self._pending_msgs[-(_GOSSIP_PENDING_MAX // 2) :]
            neighbors = [
                v[0] for addr, v in self._neighbors.get_all(only_direct=True).items() if addr != self.address and addr != msg.source
            ]
            self._pending_msgs.append((msg, neighbors))

    async def check_and_set_processed(self, msg: node_pb2.RootMessage) -> bool:
        """Mark a message as processed if new. Return True if new, False otherwise."""
        if msg.source == self.address:
            return False

        # Check if message was already processed
        async with self._processed_messages_lock:
            if msg.gossip_message.hash in self._processed_messages:
                return False

            # Evict oldest if at capacity
            if len(self._processed_messages) >= Settings.gossip.AMOUNT_LAST_MESSAGES_SAVED:
                self._processed_messages.popitem(last=False)

            # Add message
            self._processed_messages[msg.gossip_message.hash] = None
            return True

    async def _run(self) -> None:
        """Run main loop that periodically sends pending messages to neighbors."""
        while not self._terminate_event.is_set():
            start_time = asyncio.get_event_loop().time()
            messages_to_send = []
            remaining = self.messages_per_period

            async with self._pending_msgs_lock:
                while remaining > 0 and self._pending_msgs:
                    msg, targets = self._pending_msgs[0]
                    if len(targets) <= remaining:
                        messages_to_send.append((msg, targets))
                        self._pending_msgs.pop(0)
                        remaining -= len(targets)
                    else:
                        messages_to_send.append((msg, targets[:remaining]))
                        self._pending_msgs[0] = (msg, targets[remaining:])
                        remaining = 0

            # Send messages
            for msg, clients in messages_to_send:
                for client in clients:
                    try:
                        await client.send(msg, disconnect_on_error=False)
                    except Exception as e:
                        logger.warning(self.address, f"Failed to gossip to {client.nei_addr}: {e}")

            elapsed = asyncio.get_event_loop().time() - start_time
            await asyncio.sleep(max(0, self.period - elapsed))

    async def gossip_weights(
        self,
        early_stopping_fn: Callable[[], bool],
        get_candidates_fn: Callable[[], list[str]],
        status_fn: Callable[[], Any],
        model_fn: Callable[[str], tuple[Any, str, int, list[str]]],  # TODO: this can be simplified
        period: float,
        temporal_connection: bool,
    ) -> None:
        """
        Gossips model weights synchronously to a random subset of candidate nodes.

        Continues until early stopping is triggered or neighbor states stabilize.

        Args:
            early_stopping_fn: Whether to stop early.
            get_candidates_fn: Returns addresses of neighbor nodes.
            status_fn: Returns the current node state.
            model_fn: Generates the model to send to each node.
            period: Delay between gossip rounds.
            temporal_connection: Whether to use temporary connections.

        """
        last_status: list[Any] = []
        j = 0

        while True:
            start_time = asyncio.get_event_loop().time()

            if early_stopping_fn():
                logger.info(self.address, "Stopping model gossip process.")
                return

            # Get nodes which need models
            candidates = get_candidates_fn()

            # Determine end of gossip
            if not candidates:
                logger.info(self.address, "🤫 Gossip finished.")
                return

            logger.debug(self.address, f"👥 Remaining gossip targets: {candidates}")

            current_status = status_fn()
            if len(last_status) < Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS:
                last_status.append(current_status)
            else:
                last_status[j] = str(current_status)
                j = (j + 1) % Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS

                if all(status == last_status[0] for status in last_status):
                    logger.info(self.address, f"⏹️  Gossip stopped: {Settings.gossip.EXIT_ON_X_EQUAL_ROUNDS} identical rounds.")
                    logger.debug(self.address, f"Final status: {last_status[-1]}")
                    return

            # Randomly select a subset of candidate nodes
            sample_size = min(Settings.gossip.MODELS_PER_ROUND, len(candidates))
            sampled = random.sample(candidates, sample_size)

            # Get clients for sampled nodes
            clients = [v[0] for k, v in self._neighbors.get_all(only_direct=False).items() if k in sampled]

            # Generate and send model partial aggregations
            for client in clients:
                # Get Model
                model, command_name, round_num, model_hashes = model_fn(client.nei_addr)
                if model is None:
                    continue

                # Send model weights
                logger.debug(self.address, f"🗣️ Gossiping model to {client.nei_addr}.")
                await client.send(model, temporal_connection=temporal_connection)

            elapsed = asyncio.get_event_loop().time() - start_time
            await asyncio.sleep(max(0, period - elapsed))
