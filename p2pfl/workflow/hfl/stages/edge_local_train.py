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
"""Edge local training stage for HFL."""

from __future__ import annotations

from p2pfl.management.logger import logger
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.hfl.context import HFLContext, HFLPeerState
from p2pfl.workflow.shared.evaluate import evaluate_and_broadcast


class HFLEdgeLocalTrainStage(Stage[HFLContext]):
    """Edge trains on its own local data."""

    name = "edge_local_train"

    async def run(self) -> str | None:
        """Train locally and store own model for aggregation."""
        ctx = self.ctx
        address = ctx.address

        await evaluate_and_broadcast(ctx)

        logger.info(address, "Edge training...")
        await ctx.learner.fit()
        logger.info(address, "Edge training done.")

        # Store own model in peers for upcoming worker aggregation
        if address not in ctx.peers:
            ctx.peers[address] = HFLPeerState()
        ctx.peers[address].model = ctx.learner.get_model()

        return "edge_aggregate_workers"
