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
"""Worker training stage for HFL."""

from __future__ import annotations

from p2pfl.management.logger import logger
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.hfl.context import HFLContext
from p2pfl.workflow.shared.evaluate import evaluate_and_broadcast


class HFLWorkerTrainStage(Stage[HFLContext]):
    """Worker trains locally and sends model to its edge node."""

    name = "worker_train"

    async def run(self) -> str | None:
        """Train, then send model to edge."""
        ctx = self.ctx
        address = ctx.address

        await evaluate_and_broadcast(ctx)

        logger.info(address, "Worker training...")
        await ctx.learner.fit()
        logger.info(address, "Worker training done.")

        # Send trained model to edge
        model = ctx.learner.get_model()
        payload = ctx.cp.build_weights(
            "worker_model",
            ctx.experiment.round,
            model.encode_parameters(),
            model.get_contributors(),
            model.get_num_samples(),
        )
        await ctx.cp.send(ctx.edge_addr, payload)

        return "worker_wait_global"
