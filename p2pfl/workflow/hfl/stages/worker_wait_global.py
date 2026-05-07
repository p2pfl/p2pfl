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
"""Worker wait-for-global-model stage for HFL."""

from __future__ import annotations

import asyncio

from p2pfl.learning.frameworks.exceptions import DecodingParamsError, ModelNotMatchingError
from p2pfl.management.logger import logger
from p2pfl.settings import Settings
from p2pfl.workflow.engine.message import on_message
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.hfl.context import HFLContext
from p2pfl.workflow.shared.utils import wait_with_timeout


class HFLWorkerWaitGlobalStage(Stage[HFLContext]):
    """Worker waits for the global model from its edge node."""

    name = "worker_wait_global"

    def __init__(self) -> None:
        """Initialize the wait stage."""
        super().__init__()
        self._global_model_ready = asyncio.Event()

    @on_message("global_model", weights=True)
    async def handle_global_model(
        self,
        source: str,
        round: int,
        weights: bytes,
        contributors: list[str] | None,
        num_samples: int | None,
    ) -> None:
        """Receive the global model from the edge node."""
        ctx = self.ctx
        if round != ctx.experiment.round:
            logger.warning(ctx.address, f"Ignoring global model for round {round} (current: {ctx.experiment.round})")
            return
        try:
            model = ctx.learner.get_model().build_copy(
                params=weights,
                num_samples=num_samples or 0,
                contributors=list(contributors or []),
            )
            ctx.learner.set_model(model)
            self._global_model_ready.set()
        except DecodingParamsError:
            logger.error(ctx.address, "Error decoding global model parameters.")
        except ModelNotMatchingError:
            logger.error(ctx.address, "Global model does not match local model structure.")
        except Exception as e:
            logger.error(ctx.address, f"Error setting global model: {e}")

    async def run(self) -> str | None:
        """Wait for the global model, then proceed to round_finished."""
        ctx = self.ctx
        self._global_model_ready.clear()

        if not self._global_model_ready.is_set():
            await wait_with_timeout(
                self._global_model_ready,
                Settings.training.AGGREGATION_TIMEOUT,
                ctx.address,
                "Timeout waiting for global model from edge.",
            )

        return "round_finished"
