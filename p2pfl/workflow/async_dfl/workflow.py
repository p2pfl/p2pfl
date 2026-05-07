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
"""AsyncDFL asynchronous decentralized federated learning workflow."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from p2pfl.management.logger import logger
from p2pfl.workflow.async_dfl.context import AsyncDFLContext
from p2pfl.workflow.async_dfl.stages import (
    FinishStage,
    SetupStage,
    TrainingRoundStage,
)
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.engine.workflow import Workflow

if TYPE_CHECKING:
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
    from p2pfl.learning.aggregators.aggregator import Aggregator
    from p2pfl.learning.frameworks.learner import Learner


class AsyncDFL(Workflow[AsyncDFLContext]):
    """
    AsyncDFL: asynchronous decentralized federated learning.

    Uses push-sum consensus for de-biased gradient updates, priority-based
    neighbor selection, and tau-periodic network updates.

    Flow: setup -> training_round -> (loop) -> finish
    """

    context_class = AsyncDFLContext

    def get_stages(self) -> list[Stage[AsyncDFLContext]]:
        """Return the stages for AsyncDFL."""
        return [SetupStage(), TrainingRoundStage(), FinishStage()]

    def create_context(
        self,
        address: str,
        learner: Learner,
        aggregator: Aggregator,
        cp: CommunicationProtocol,
        generator: random.Random,
        experiment: Experiment,
    ) -> AsyncDFLContext:
        """Build the typed context for AsyncDFL with push-sum model wrapping."""
        ctx = super().create_context(
            address=address, learner=learner, aggregator=aggregator, cp=cp, generator=generator, experiment=experiment
        )
        self._wrap_model_if_supported(ctx)
        return ctx

    def validate_experiment(self, ctx: AsyncDFLContext) -> None:
        """Resolve defaults and validate AsyncDFL-specific hyperparameters."""
        exp = ctx.experiment
        if not hasattr(exp, "tau"):
            exp.tau = 2
        if not hasattr(exp, "dmax"):
            exp.dmax = 5
        if not hasattr(exp, "top_k_neighbors"):
            exp.top_k_neighbors = 3
        if not hasattr(exp, "eval_every"):
            exp.eval_every = 0
        if exp.tau < 1:
            raise ValueError("tau must be >= 1.")
        if exp.dmax < 1:
            raise ValueError("dmax must be >= 1.")
        if exp.top_k_neighbors < 1:
            raise ValueError("top_k_neighbors must be >= 1.")
        if exp.eval_every < 0:
            raise ValueError("eval_every must be >= 0.")

    @staticmethod
    def _wrap_model_if_supported(ctx: AsyncDFLContext) -> None:
        """Wrap the learner's model for async DFL push-sum debiasing if the framework supports it."""
        try:
            from p2pfl.learning.frameworks.custom_model_factory import CustomModelFactory

            model = ctx.learner.get_model()
            wrapped = CustomModelFactory.create_model("AsyDFL", model)
            ctx.learner.set_model(wrapped)
            logger.info(ctx.address, "Model wrapped for push-sum debiasing.")
        except (ValueError, ImportError):
            logger.warning(
                ctx.address,
                "Push-sum model wrapping not available for this framework. Continuing without debiasing.",
            )
