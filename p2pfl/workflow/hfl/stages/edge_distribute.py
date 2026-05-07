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
"""Edge distribute-global-model stage for HFL."""

from __future__ import annotations

from p2pfl.management.logger import logger
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.hfl.context import HFLContext


class HFLEdgeDistributeStage(Stage[HFLContext]):
    """Edge distributes the global model to all its workers."""

    name = "edge_distribute"

    async def run(self) -> str | None:
        """Send global model to every worker."""
        ctx = self.ctx
        model = ctx.learner.get_model()
        payload = ctx.cp.build_weights(
            "global_model",
            ctx.experiment.round,
            model.encode_parameters(),
            model.get_contributors(),
            model.get_num_samples(),
        )
        for worker_addr in ctx.worker_addrs:
            await ctx.cp.send(worker_addr, payload)

        logger.info(ctx.address, f"Global model distributed to {len(ctx.worker_addrs)} workers.")
        return "round_finished"
