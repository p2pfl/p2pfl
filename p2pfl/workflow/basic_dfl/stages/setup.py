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
"""Setup stage for BasicDFL: peer synchronization and initial model distribution."""

from __future__ import annotations

import asyncio

from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.basic_dfl.context import BasicDFLContext, BasicPeerState
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.shared.gossiping import ModelGate
from p2pfl.workflow.shared.utils import wait_with_timeout


class SetupStage(Stage[BasicDFLContext]):
    """Synchronize peers, distribute the initial model via gossip, then proceed."""

    def __init__(self) -> None:
        """Initialize setup stage events."""
        super().__init__()
        self._nodes_ready = asyncio.Event()
        self._model_received = asyncio.Event()

    async def run(self) -> str | None:
        """Sync peers, gossip the initial model, then proceed to round_init."""
        ctx = self.ctx
        self._nodes_ready.clear()
        self._model_received.clear()

        ctx.learner.set_epochs(ctx.experiment.epochs_per_round)

        # Sync: gossip node_initialized and wait for all peers
        self._register_peer(ctx, ctx.address)
        try:
            await ctx.cp.broadcast_gossip(ctx.cp.build_msg("node_initialized"))
        except Exception as e:
            logger.debug(ctx.address, f"Error broadcasting node initialization: {e}")

        await wait_with_timeout(
            self._nodes_ready,
            Settings.training.SYNCHRONIZATION_TIMEOUT,
            ctx.address,
            "Timeout waiting for peers. Proceeding anyway.",
        )

        # Initiator starts the model gossip (no gate — no one has the model yet)
        if ctx.experiment.is_initiator:
            model = await ctx.learner.aget_model()
            encoded_model = model.encode_parameters()
            await ctx.cp.broadcast(ctx.cp.build_weights("initial_model", 0, encoded_model))
            self._model_received.set()

        # All nodes wait for the model to propagate
        await wait_with_timeout(
            self._model_received,
            Settings.training.SYNCHRONIZATION_TIMEOUT,
            ctx.address,
            "Timeout waiting for initial model.",
        )

        return "round_init"

    ###
    #    Callbacks
    ###

    def _register_peer(self, ctx: BasicDFLContext, source: str) -> None:
        if source in ctx.peers:
            return
        ctx.peers[source] = BasicPeerState()
        logger.debug(ctx.address, f"📡 {source} peer created")

        if len(ctx.peers) == len(ctx.cp.get_neighbors(only_direct=False)) + 1:
            self._nodes_ready.set()

    ###
    #    Message handlers
    ###

    @on_message("node_initialized")
    async def handle_node_initialized(self, source: str, round: int, *args) -> None:
        """Handle a node_initialized message by registering the peer."""
        self._register_peer(self.ctx, source)

    @on_message("pre_send_initial_model", during={"setup", "round_init", "learning", "voting"})
    async def handle_pre_send_initial_model(self, source: str, round: int, *args) -> str:
        """Accept the initial model only if not already received."""
        return "false" if self._model_received.is_set() else "true"

    @on_message("initial_model", weights=True)
    async def handle_initial_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Receive the initial model and gossip it to neighbors via gate."""
        if self._model_received.is_set():
            return
        ctx = self.ctx
        logger.info(ctx.address, f"📥 Initial model received from {source}.")
        await ctx.learner.aset_model(weights)

        # Re-gossip using gate (peers are synced, so gate queries will get proper responses)
        payload = ctx.cp.build_weights("initial_model", 0, weights)
        gate = ModelGate(ctx.cp, ctx.address, pre_send_command="pre_send_initial_model")
        for neighbor in ctx.cp.get_neighbors(only_direct=False):
            await gate.send_if_accepted(
                neighbor=neighbor,
                weight_command="initial_model",
                contributors=[source],
                round_num=0,
                payload=payload,
            )
        self._model_received.set()

    @on_message("node_ready")
    async def handle_node_ready(self, source: str, round: int, *args) -> None:
        """Handle a node_ready ACK from a peer."""
        self._register_peer(self.ctx, source)
