#
# This file is part of the p2pfl (see https://github.com/pguijas/p2pfl).
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
"""Setup and synchronization stage for AsyncDFL."""

from __future__ import annotations

import asyncio
import contextlib

from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.async_dfl.context import AsyncDFLContext, AsyncPeerState
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage


class SetupStage(Stage[AsyncDFLContext]):
    """Setup and initial synchronization stage for async DFL."""

    def __init__(self) -> None:
        """Initialize the setup stage."""
        super().__init__()
        self._nodes_ready = asyncio.Event()

    async def run(self) -> str | None:
        """Broadcast initialization, wait for all nodes, then proceed."""
        ctx = self.ctx
        self._nodes_ready.clear()

        ctx.learner.set_epochs(ctx.experiment.epochs_per_round)
        logger.info(ctx.address, "⏳ Starting async training.")
        logger.info(ctx.address, "⏳ Waiting initialization.")

        await self._create_peer(ctx, source=ctx.address)

        # Always broadcast at least once so peers that already received our
        # handler-created peer still learn we exist (avoids a race where we
        # skip the loop because _nodes_ready was set by an incoming handler
        # before we ever broadcast).
        try:
            await ctx.cp.broadcast_gossip(ctx.cp.build_msg("node_initialized"))
        except Exception as e:
            logger.debug(ctx.address, f"Error broadcasting node initialization command: {e}")

        # Re-broadcast periodically until all peers respond or timeout.
        rebroadcast_interval = 2.0
        timeout = Settings.training.SYNCHRONIZATION_TIMEOUT
        elapsed = 0.0
        while not self._nodes_ready.is_set() and elapsed < timeout:
            wait_time = min(rebroadcast_interval, timeout - elapsed)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._nodes_ready.wait(), timeout=wait_time)
            elapsed += wait_time

            if not self._nodes_ready.is_set():
                try:
                    await ctx.cp.broadcast_gossip(ctx.cp.build_msg("node_initialized"))
                except Exception as e:
                    logger.debug(ctx.address, f"Error broadcasting node initialization command: {e}")

        if not self._nodes_ready.is_set():
            logger.warning(ctx.address, "Timeout waiting for all nodes to initialize. Proceeding with available peers.")

        # Column-stochastic mixing weights (Alg. 1: p = 1/|N^out|, includes self)
        neighbors = list(ctx.cp.get_neighbors(only_direct=True))
        weight = 1.0 / (len(neighbors) + 1)
        for neighbor in neighbors:
            if neighbor in ctx.peers:
                ctx.peers[neighbor].mixing_weight = weight
        if ctx.address in ctx.peers:
            ctx.peers[ctx.address].mixing_weight = weight

        participants = neighbors + [ctx.address]
        logger.info(ctx.address, f"Peers in network: {participants}")
        logger.debug(ctx.address, "✅ All nodes synchronized.")
        return "training_round"

    def _all_nodes_started(self, ctx: AsyncDFLContext) -> bool:
        return len(ctx.peers) == (len(ctx.cp.get_neighbors(only_direct=True)) + 1)

    async def _create_peer(self, ctx: AsyncDFLContext, source: str = "") -> None:
        if source != ctx.address and source not in ctx.cp.get_neighbors(only_direct=True):
            return
        if source in ctx.peers:
            return
        ctx.peers[source] = AsyncPeerState()
        logger.debug(ctx.address, f"📡 {source} peer created")

        if self._all_nodes_started(ctx):
            self._nodes_ready.set()

    @on_message("node_initialized")
    async def handle_node_initialized(self, source: str, round: int, *args) -> None:
        """Handle a node_initialized message by creating a peer."""
        await self._create_peer(self.ctx, source)
