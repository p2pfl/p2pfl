#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
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

"""P2PFL Node."""

import math
import os
import random
from typing import Tuple, Type

from p2pfl.communication.protocols.edge.client import EdgeClientConnection
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.learning.pytorch.lightning_learner import LightningLearner
from p2pfl.management.logger import logger
from p2pfl.settings import Settings

# Disbalbe grpc log (pytorch causes warnings)
if logger.get_level_name(logger.get_level()) != "DEBUG":
    os.environ["GRPC_VERBOSITY"] = "NONE"

class EdgeNode:
    def __init__(
        self,
        connect_to,
        model: P2PFLModel,
        data: P2PFLDataset,
        learner: Type[NodeLearner] = LightningLearner,
        **kwargs,
    ) -> None:
        """Initialize a node."""
        # Define callbacks (tmp - hardcoded)
        callbacks = {
            "vote": self._vote_train_set,
            "train": self._train,
            "validate": self._validate,
        }

        # Start communication protocol
        self.client = EdgeClientConnection(connect_to, callbacks)

        # Learning
        self.learner = learner(model, data, "EdgeNode")

    #####################
    #  Node Management  #
    #####################

    async def start(self):
        # TODO: P2PFL Web Services
        logger.info("EdgeNode", "Connecting...")
        await self.client.start()

    async def stop(self):
        logger.info("EdgeNode", "Stopping node...")
        raise NotImplementedError

    ##############
    #  Commands  # -> HARD CODED!
    ##############

    def _vote_train_set(self, candidates: list[str] = None, _: bytes = None) -> Tuple[list[str], None]:
        # Check if candidates are available
        if candidates is None:
            return [], None

        # Gen vote
        samples = min(Settings.TRAIN_SET_SIZE, len(candidates))
        nodes_voted = random.sample(candidates, samples)
        weights = [math.floor(random.randint(0, 1000) / (i + 1)) for i in range(samples)]
        votes = list(zip(nodes_voted, weights))

        # Build response
        return list(map(str, list(sum(votes, ())))), None

    def _train(self, args: list[str] = None, weights: bytes = None) -> Tuple[None, bytes]:
        # Set weights
        self.learner.set_model(weights)

        # Set epochs
        print(f"setting epochs to {args[0]}")
        self.learner.set_epochs(int(args[0]))

        # Train
        logger.info("EdgeNode", "🏋️‍♀️ Training...")
        self.learner.fit()

        # Build response
        return None, self.learner.get_model().encode_parameters()


    def _validate(self, args: list[str] = None, weights: bytes = None) -> Tuple[list[str], None]:
        # Check input
        ...

        # Set weights
        self.learner.set_model(weights)

        # Validate
        logger.info("EdgeNode", "🔬 Evaluating...")
        eval_results = self.learner.evaluate()
        logger.info("EdgeNode", f"📈 Evaluated. Results: {eval_results}")

        # Build response
        flattened_metrics = [str(item) for pair in eval_results.items() for item in pair]

        # Send metrics
        return flattened_metrics, None
