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
"""Edge aggregate-workers stage for HFL."""

from __future__ import annotations

import asyncio

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.hfl.context import HFLContext, HFLPeerState
from p2pfl.workflow.shared.utils import wait_with_timeout


class HFLEdgeAggregateWorkersStage(Stage[HFLContext]):
    """Edge waits for worker models and aggregates them with its own."""

    name = "edge_aggregate_workers"

    def __init__(self) -> None:
        """Initialize the aggregation stage."""
        super().__init__()
        self._workers_complete = asyncio.Event()

    def _all_worker_models_received(self, ctx: HFLContext) -> bool:
        expected = set(ctx.worker_addrs)
        if ctx.edge_trains:
            expected |= {ctx.address}
        received = {addr for addr, p in ctx.peers.items() if p.model is not None}
        return expected.issubset(received)

    # Accept worker models during edge training too (workers may finish faster)
    @on_message("worker_model", weights=True, during={"edge_local_train", "edge_aggregate_workers", "round_finished"})
    async def handle_worker_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Receive a trained model from a worker."""
        ctx = self.ctx
        if round != ctx.experiment.round:
            logger.warning(ctx.address, f"Ignoring worker model for round {round} (current: {ctx.experiment.round})")
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
            logger.debug(ctx.address, f"Worker model received from {source}.")

            if self._all_worker_models_received(ctx):
                self._workers_complete.set()
        except DecodingParamsError:
            logger.error(ctx.address, "Error decoding worker model parameters.")
        except ModelNotMatchingError:
            logger.error(ctx.address, "Worker model does not match local model structure.")
        except Exception as e:
            logger.error(ctx.address, f"Error processing worker model: {e}")

    async def run(self) -> str | None:
        """Wait for all worker models, aggregate, and flatten contributors."""
        ctx = self.ctx
        self._workers_complete.clear()

        if not self._all_worker_models_received(ctx):
            await wait_with_timeout(
                self._workers_complete,
                Settings.training.AGGREGATION_TIMEOUT,
                ctx.address,
                "Timeout waiting for worker models, proceeding with available.",
            )

        # Aggregate all available models (stateless call)
        models = [p.model for p in ctx.peers.values() if p.model is not None]
        if models:
            agg_model = ctx.aggregator.aggregate(models)
            ctx.learner.set_model(agg_model)

        # Flatten contributors for the root aggregation phase:
        # The aggregated model has contributors=[edge, w1, w2, ...].
        # The root tracks by edge address only, so we flatten to
        # [ctx.address] to ensure correct matching.
        model = ctx.learner.get_model()
        model.set_contribution([ctx.address], model.get_num_samples())

        logger.info(ctx.address, "Worker aggregation done.")
        return "edge_sync_root"
