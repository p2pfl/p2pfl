#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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
"""Start learning stage."""

import time
from typing import Any

from p2pfl.communication.commands.message.model_initialized_command import ModelInitializedCommand
from p2pfl.communication.commands.weights.init_model_command import InitModelCommand
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.settings import Settings
from p2pfl.stages.stage import Stage
from p2pfl.stages.stage_factory import StageFactory


class StartLearningStage(Stage):
    """Start learning stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "StartLearningStage"

    @staticmethod
    def execute(
        rounds: int | None = None,
        epochs: int | None = None,
        experiment_name: str | None = None,
        state: NodeState | None = None,
        learner: Learner | None = None,
        communication_protocol: CommunicationProtocol | None = None,
        aggregator: Aggregator | None = None,
        **kwargs,
    ) -> type["Stage"] | None:
        """Execute the stage."""
        if (
            rounds is None
            or epochs is None
            or experiment_name is None
            or state is None
            or learner is None
            or communication_protocol is None
            or aggregator is None
        ):
            raise Exception("Invalid parameters on StartLearningStage.")

        # Init
        with state.start_thread_lock:
            state.set_experiment(
                experiment_name,
                rounds,
                dataset_name=learner.get_data().dataset_name,
                model_name=learner.get_model().__class__.__name__,
                aggregator_name=aggregator.__class__.__name__,
                framework_name=learner.get_model().get_framework(),
                learning_rate=getattr(learner.get_model().get_model(), "lr_rate", None),
                batch_size=learner.get_data().batch_size,
                epochs_per_round=epochs,
            )
            learner.set_epochs(epochs)
        begin = time.time()

        # Wait and gossip model inicialization
        logger.info(state.addr, "⏳ Waiting initialization.")
        state.model_initialized_lock.acquire()
        # Communicate Initialization
        communication_protocol.broadcast(communication_protocol.build_msg(ModelInitializedCommand.get_name()))
        logger.info(state.addr, "🗣️ Gossiping model initialization.")
        time.sleep(1.0)
        StartLearningStage.__gossip_model(state, communication_protocol, learner)

        # Wait to guarantee new connection heartbeats convergence
        wait_time = Settings.heartbeat.WAIT_CONVERGENCE - (time.time() - begin)
        if wait_time > 0:
            time.sleep(wait_time)

        # Vote
        return StageFactory.get_stage("VoteTrainSetStage")

    @staticmethod
    def __gossip_model(
        state: NodeState,
        communication_protocol: CommunicationProtocol,
        learner: Learner,
    ) -> None:
        def early_stopping_fn():
            return state.round is None

        # Wait a model (init or aggregated)
        def candidate_condition(node: str) -> bool:
            return node not in state.nei_status

        def get_candidates_fn() -> list[str]:
            return [n for n in communication_protocol.get_neighbors(only_direct=True) if candidate_condition(n)]

        def status_fn() -> Any:
            return get_candidates_fn()

        def model_fn(_: str) -> tuple[Any, str, int, list[str]]:
            if state.round is None:
                raise Exception("Round not initialized.")
            encoded_model = learner.get_model().encode_parameters()
            return (
                communication_protocol.build_weights(InitModelCommand.get_name(), state.round, encoded_model),
                InitModelCommand.get_name(),
                state.round,
                [str(state.round)],
            )

        # Gossip
        communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
        )
