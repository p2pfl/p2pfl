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

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from p2pfl.communication.commands.command import Command
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.stages.workflow_type import WorkflowType
from p2pfl.stages.workflows import StageWorkflow

if TYPE_CHECKING:  # Only imports the below statements during type checking
    from p2pfl.node import Node


class WorkflowFactory(abc.ABC):
    """Command interface."""

    @abc.abstractmethod
    def create_workflow() -> StageWorkflow:
        """Create a workflow."""
        raise NotImplementedError

    @abc.abstractmethod
    def create_commands(node: Node) -> list[Command]:
        """Create commands list."""
        raise NotImplementedError

    def create_model(model: P2PFLModel) -> P2PFLModel:
        """Create model."""
        return model



# Abstract factory pattern
class WorkflowFactoryProducer:
    """Factory class to create workflows. Main goal: Avoid cyclic imports."""

    @staticmethod
    def get_factory(workflow_name: WorkflowType) -> type[WorkflowFactory]:
        """Return the stage class."""
        if workflow_name == WorkflowType.BASIC:
            from p2pfl.stages.base_node import BasicDFLFactory

            return BasicDFLFactory
        elif workflow_name == WorkflowType.ASYNC:
            from p2pfl.stages.asyDFL import AsyDFLFactory

            return AsyDFLFactory
        else:
            raise Exception("Invalid workflow name.")
