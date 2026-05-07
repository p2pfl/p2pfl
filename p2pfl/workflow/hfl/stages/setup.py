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
"""Setup and branching stage for HFL."""

from __future__ import annotations

from p2pfl.management.logger import logger
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.hfl.context import HFLContext


class HFLSetupStage(Stage[HFLContext]):
    """Initialize the HFL workflow and branch by role."""

    name = "setup"

    async def run(self) -> str | None:
        """Set up learner, initialize round, and branch by role."""
        ctx = self.ctx

        ctx.learner.set_epochs(ctx.experiment.epochs_per_round)

        logger.info(ctx.address, f"Starting HFL as {ctx.role}.")

        if ctx.role == "worker":
            return "worker_train"
        if ctx.role == "root":
            return "root_aggregate"
        if ctx.edge_trains:
            return "edge_local_train"
        return "edge_aggregate_workers"
