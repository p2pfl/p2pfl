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
"""Wait aggregated models stage."""

import math
from typing import Optional, Type, Union

from p2pfl.communication.commands.message.asyDFL.ctx_info_updating_command import (
    IndexInformationUpdatingCommand,
    ModelInformationUpdatingCommand,
    PushSumWeightInformationUpdatingCommand,
)
from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks.learner import Learner
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.settings import Settings
from p2pfl.stages.asyDFL.stage_factory import AsyDFLStageFactory
from p2pfl.stages.stage import Stage, check_early_stop


class NeighborSelectionStage(Stage):
    """Neighbor selection stage."""

    @staticmethod
    def name():
        """Return the name of the stage."""
        return "NeighborSelectionStage"

    @staticmethod
    def execute(
        state: Optional[NodeState] = None, communication_protocol: Optional[CommunicationProtocol] = None,
        learner: Optional[Learner] = None,
        aggregator: Optional[Aggregator] = None,
        **kwargs
    ) -> Union[Type["Stage"], None]:
        """
        Execute the stage. Perform neighbor selection and update models.

        Parameters:
            state: The node state.
            communication_protocol: The communication protocol.
            learner: The learner.
            num_iterations: The total number of iterations T.
            mixing_weight_init: The initial mixing weight.
            tau: The interval for updating neighbors.
            epsilon: The convergence threshold.

        Returns:
            Union[Type["Stage"], None]: The next stage to
            execute or None if the process is finished.

        """
        if state.round % state.tau == 0:
            # Initialize a list to store neighbors with their computed priority
            neighbor_priorities = []

            # Calculate priority p(b_i,j) (Equation 40)
            for neighbor in list(communication_protocol.get_neighbors(only_direct=True)):
                neighbor_loss, neighbor_idx_local_iteration = state.losses.get(neighbor,(0,0))
                priority = NeighborSelectionStage.compute_priority(state.round,
                                state.push_times.get(neighbor,0),
                                neighbor_idx_local_iteration,
                                state.reception_times.get(state.addr,0),
                                state.losses[state.addr][0],
                                neighbor_loss,
                                20) # TODO: dmax parameter
                # Append the neighbor and their computed priority as a tuple
                neighbor_priorities.append((neighbor, priority))

            logger.info(state.addr, f"👥 Neighbor priorities: {neighbor_priorities}")
            # Sort the neighbors by their priority in descending order (highest priority first)
            neighbor_priorities.sort(key=lambda x: x[1], reverse=True)

            # Construct selected neighbor nodes set N*_i,t
            selected_neighbors = []
            # Greedily select neighbors based on highest priority
            for neighbor, _ in neighbor_priorities:
                # Apply a greedy selection strategy

                # TODO: adjust the condition
                if len(selected_neighbors) < 3:
                    selected_neighbors.append(neighbor)
                else:
                    break  # Stop once you reach the limit

            logger.info(state.addr, f"🧾 Selected neighbors: {selected_neighbors}")
            # Push model to selected neighbors and update push-sum weights
            for neighbor in selected_neighbors:
                NeighborSelectionStage.__send_model(state, communication_protocol, neighbor, learner.get_model())
                NeighborSelectionStage.__send_push_sum_weight(state, communication_protocol, neighbor, state.push_sum_weights[state.addr])
                state.push_times[neighbor] = state.round

            # Perform P2P updates with received models ω_j by Equations (5) and (6)
            # Set aggregated model
            if len(aggregator.get_aggregated_models()) > 0:
                learner.set_model(aggregator.wait_and_get_aggregation())
                for neighbor in aggregator.get_aggregated_models():
                    # Update push-sum weights
                    state.push_sum_weights[state.addr] += state.mixing_weights[neighbor] * state.push_sum_weights[neighbor]

                    # Send local iteration index to neighbors
                    NeighborSelectionStage.__send_local_iteration_index(state, communication_protocol, neighbor, state.round)

            # Clear the list of received models (W_i)
            #state.models_aggregated = {}
            aggregator.clear()

        state.increase_round()
        return AsyDFLStageFactory.get_stage("RepeatStage")

    @staticmethod
    def select_neighbors(priority: list) -> list:
        """
        Select the neighbors based on the priority p(b_i,j) of node i selecting neighbor j.

        Parameters:
            priority: The priority of selecting neighbors.

        Returns:
            list: The selected neighbors.

        """
        pass

    @staticmethod
    def compute_priority(
        ti: int,
        tp_ij: int,
        tj: int,
        tl_ji: int,
        f_ti: float,
        f_tj: float,
        dmax: int
        ):
        """
        Compute the priority p(b_ij) based on the given formula.

        Parameters:
            ti: Index of the local iteration on node i.
            tp_ij: Index of the local iteration when node i pushes model to node j.
            tj: Index of the local iteration on node j.
            tl_ji: Index of the local iteration when node j updates with model from node i.
            f_ti: Training loss or function value at node i.
            f_tj: Training loss or function value at node j.
            dmax: Maximum communication frequency bound.

        Returns:
            float: The computed priority p(b_ij).

        """
        dij = abs((ti - tp_ij) - (tj - tl_ji)) / dmax
        loss_term = math.exp(abs(f_ti - f_tj)) / math.exp(1)
        priority = dij + (1 - dij) * loss_term

        return min(priority, 1.0)  # Ensure priority does not exceed 1

    @staticmethod
    def __send_local_iteration_index(state: NodeState, communication_protocol: CommunicationProtocol, neighbor: str, index: int) -> None:
        """
        Send the local iteration index to the neighbors.

        Parameters:
            state: The node state.
            learner: The learner.
            communication_protocol: The communication protocol.

        """
        communication_protocol.send(
            nei=neighbor,
            msg=communication_protocol.build_msg(
                IndexInformationUpdatingCommand.get_name(),
                [str(index)],
                round=state.round,
            )
        )

    @staticmethod
    def __send_model(state: NodeState, communication_protocol: CommunicationProtocol, neighbor: str, model: P2PFLModel) -> None:
        """
        Send the local iteration index to the neighbors.

        Parameters:
            state: The node state.
            learner: The learner.
            communication_protocol: The communication protocol.

        """
        encoded_model = model.encode_parameters()
        communication_protocol.send(
            nei=neighbor,
            msg=communication_protocol.build_weights(
                ModelInformationUpdatingCommand.get_name(),
                state.round,
                encoded_model,
                model.get_contributors(),
                model.get_num_samples(),
            ),
        )

    @staticmethod
    def __send_push_sum_weight(state: NodeState, communication_protocol: CommunicationProtocol, neighbor: str, push_sum_weight: float) -> None:
        """
        Send the local iteration index to the neighbors.

        Parameters:
            state: The node state.
            learner: The learner.
            communication_protocol: The communication protocol.

        """
        communication_protocol.send(
            nei=neighbor,
            msg=communication_protocol.build_msg(
                PushSumWeightInformationUpdatingCommand.get_name(),
                [str(push_sum_weight)],
                round=state.round,
            ),
        )
