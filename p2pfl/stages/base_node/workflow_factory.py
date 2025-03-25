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

"""Stage factory."""

from p2pfl.communication.commands.command import Command
from p2pfl.node import Node
from p2pfl.stages.workflow_factory import WorkflowFactory
from p2pfl.stages.workflows import StageWorkflow


class BasicDFLFactory(WorkflowFactory):
    """Factory class to create workflows. Main goal: Avoid cyclic imports."""

    @staticmethod
    def create_workflow() -> StageWorkflow:
        """Create a workflow."""
        from p2pfl.stages.base_node.workflow import BasicDFLWorkflow

        return BasicDFLWorkflow()

    @staticmethod
    def create_commands(node: Node) -> list[Command]:
        """Create a list of commands."""
        from p2pfl.communication.commands.message.metrics_command import MetricsCommand
        from p2pfl.communication.commands.message.model_initialized_command import ModelInitializedCommand
        from p2pfl.communication.commands.message.models_agregated_command import ModelsAggregatedCommand
        from p2pfl.communication.commands.message.models_ready_command import ModelsReadyCommand
        from p2pfl.communication.commands.message.start_learning_command import StartLearningCommand
        from p2pfl.communication.commands.message.stop_learning_command import StopLearningCommand
        from p2pfl.communication.commands.message.vote_train_set_command import VoteTrainSetCommand
        from p2pfl.communication.commands.weights.full_model_command import FullModelCommand
        from p2pfl.communication.commands.weights.init_model_command import InitModelCommand
        from p2pfl.communication.commands.weights.partial_model_command import PartialModelCommand

        return [
            StartLearningCommand(node),
            StopLearningCommand(node),
            ModelInitializedCommand(node),
            VoteTrainSetCommand(node),
            ModelsAggregatedCommand(node),
            ModelsReadyCommand(node),
            MetricsCommand(node),
            InitModelCommand(node),
            PartialModelCommand(node),
            FullModelCommand(node),
        ]
