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
"""Edge sync-with-root stage for HFL."""

from __future__ import annotations

import asyncio

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.hfl.context import HFLContext
from p2pfl.workflow.shared.utils import wait_with_timeout


class HFLEdgeSyncRootStage(Stage[HFLContext]):
    """Send aggregated model to root and wait for the global model back.

    Replaces inter-edge gossip with a centralized root aggregation.
    """

    name = "edge_sync_root"

    def __init__(self) -> None:
        """Initialize the sync stage."""
        super().__init__()
        self._global_received = asyncio.Event()

    @on_message(
        "root_global_model",
        weights=True,
        during={"edge_sync_root", "edge_aggregate_workers"},
    )
    async def handle_root_global_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Receive the global model from the root node."""
        ctx = self.ctx
        if round != ctx.experiment.round:
            logger.warning(ctx.address, f"Ignoring root global model for round {round} (current: {ctx.experiment.round})")
            return
        if contributors is None or num_samples is None:
            raise ValueError("Contributors and num_samples are required")
        try:
            model = ctx.learner.get_model().build_copy(
                params=weights,
                num_samples=num_samples,
                contributors=list(contributors),
            )
            ctx.learner.set_model(model)
            self._global_received.set()
            logger.debug(ctx.address, f"Global model received from root {source}.")
        except DecodingParamsError:
            logger.error(ctx.address, "Error decoding root global model parameters.")
        except ModelNotMatchingError:
            logger.error(ctx.address, "Root global model does not match local model structure.")
        except Exception as e:
            logger.error(ctx.address, f"Error processing root global model: {e}")

    async def run(self) -> str | None:
        """Send model to root, wait for global model back."""
        ctx = self.ctx
        self._global_received.clear()

        # Send aggregated model to root
        model = ctx.learner.get_model()
        payload = ctx.cp.build_weights(
            "edge_model",
            ctx.experiment.round,
            model.encode_parameters(),
            model.get_contributors(),
            model.get_num_samples(),
        )
        await ctx.cp.send(ctx.root_addr, payload)
        logger.info(ctx.address, f"Aggregated model sent to root {ctx.root_addr}.")

        # Wait for global model from root
        if not self._global_received.is_set():
            await wait_with_timeout(
                self._global_received,
                Settings.training.AGGREGATION_TIMEOUT,
                ctx.address,
                "Timeout waiting for global model from root.",
            )

        return "edge_distribute"
