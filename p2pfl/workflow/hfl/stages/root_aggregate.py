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
"""Root aggregate-edges stage for HFL."""

from __future__ import annotations

import asyncio

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.hfl.context import HFLContext, HFLPeerState
from p2pfl.workflow.shared.utils import wait_with_timeout


class HFLRootAggregateStage(Stage[HFLContext]):
    """Root receives models from all child edges and aggregates them."""

    name = "root_aggregate"

    def __init__(self) -> None:
        """Initialize the aggregation stage."""
        super().__init__()
        self._edges_complete = asyncio.Event()

    def _all_edge_models_received(self, ctx: HFLContext) -> bool:
        expected = set(ctx.child_edge_addrs)
        received = {addr for addr, p in ctx.peers.items() if p.model is not None}
        return expected.issubset(received)

    @on_message("edge_model", weights=True, during={"root_aggregate", "round_finished"})
    async def handle_edge_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Receive a model from a child edge."""
        ctx = self.ctx
        if round != ctx.experiment.round:
            logger.warning(ctx.address, f"Ignoring edge model for round {round} (current: {ctx.experiment.round})")
            return
        if contributors is None or num_samples is None:
            raise ValueError("Contributors and num_samples are required")
        try:
            model = ctx.learner.get_model().build_copy(
                params=weights,
                num_samples=num_samples,
                contributors=list(contributors),
            )
            if source not in ctx.peers:
                ctx.peers[source] = HFLPeerState()
            ctx.peers[source].model = model
            logger.debug(ctx.address, f"Edge model received from {source}.")

            if self._all_edge_models_received(ctx):
                self._edges_complete.set()
        except DecodingParamsError:
            logger.error(ctx.address, "Error decoding edge model parameters.")
        except ModelNotMatchingError:
            logger.error(ctx.address, "Edge model does not match local model structure.")
        except Exception as e:
            logger.error(ctx.address, f"Error processing edge model: {e}")

    async def run(self) -> str | None:
        """Wait for all edge models and aggregate."""
        ctx = self.ctx
        self._edges_complete.clear()

        if not self._all_edge_models_received(ctx):
            await wait_with_timeout(
                self._edges_complete,
                Settings.training.AGGREGATION_TIMEOUT,
                ctx.address,
                "Timeout waiting for edge models, proceeding with available.",
            )

        models = [p.model for p in ctx.peers.values() if p.model is not None]
        if models:
            global_model = ctx.aggregator.aggregate(models)
            ctx.learner.set_model(global_model)

        logger.info(ctx.address, f"Root aggregation done ({len(models)} edge models).")
        return "root_distribute"
