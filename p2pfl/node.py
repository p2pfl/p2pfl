#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/federated_learning_p2p).
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

import contextlib
import os
import threading
from typing import Any, Dict, Type

from p2pfl.commands.add_model_command import AddModelCommand
from p2pfl.commands.init_model_command import InitModelCommand
from p2pfl.commands.metrics_command import MetricsCommand
from p2pfl.commands.model_initialized_command import ModelInitializedCommand
from p2pfl.commands.models_agregated_command import ModelsAggregatedCommand
from p2pfl.commands.models_ready_command import ModelsReadyCommand
from p2pfl.commands.start_learning_command import StartLearningCommand
from p2pfl.commands.stop_learning_command import StopLearningCommand
from p2pfl.commands.vote_train_set_command import VoteTrainSetCommand
from p2pfl.communication.communication_protocol import CommunicationProtocol
from p2pfl.communication.grpc.grpc_communication_protocol import (
    GrpcCommunicationProtocol,
)
from p2pfl.exceptions import LearnerNotSetException, NodeRunningException, ZeroRoundsException
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.pytorch.lightning_learner import LightningLearner
from p2pfl.management.logger import logger
from p2pfl.node_state import NodeState
from p2pfl.stages.workflows import LearningWorkflow

# Disbalbe grpc log (pytorch causes warnings)
if logger.get_level_name(logger.get_level()) != "DEBUG":
    os.environ["GRPC_VERBOSITY"] = "NONE"


class Node:
    """
    Represents a learning node in the federated learning network.

    The following example shows how to create a node with a MLP model and a MnistFederatedDM dataset. Then, the node is
    started, connected to another node, and the learning process is started.

    >>> node = Node(
    ...     MLP(),
    ...     MnistFederatedDM(),
    ... )
    >>> node.start()
    >>> node.connect("127.0.0.1:666")
    >>> node.set_start_learning(rounds=2, epochs=1)

    Args:
        model: Model to be used in the learning process.
        data: Dataset to be used in the learning process.
        address: The address of the node.
        learner: The learner class to be used.
        aggregator: The aggregator class to be used.
        protocol: The communication protocol to be used.
        **kwargs: Additional arguments.

    .. todo::
        Instanciate the aggregator dynamically.

    .. todo::
        Connect nodes dynamically (while learning).

    """

    def __init__(
        self,
        model,
        data,
        address: str = "127.0.0.1",
        learner: Type[NodeLearner] = LightningLearner,
        aggregator: Type[Aggregator] = FedAvg,
        protocol: Type[CommunicationProtocol] = GrpcCommunicationProtocol,
        **kwargs,
    ) -> None:
        """Initialize a node."""
        # Communication protol
        self._communication_protocol = protocol(address)
        self.addr = self._communication_protocol.get_address()

        # Learning
        self.data = data
        self.model = model
        self.learner_class = learner
        self.aggregator = aggregator(
            node_name=self.addr
        )  # Ponerlo como learner (que se vaya instanciando dinamicamente)

        # State
        self.__running = False
        self.state = NodeState(self.addr)

        # Workflow
        self.learning_workflow = LearningWorkflow()

        # Commands
        commands = [
            StartLearningCommand(self.__start_learning_thread),
            StopLearningCommand(self.state, self.aggregator),
            ModelInitializedCommand(self.state),
            VoteTrainSetCommand(self.state),
            ModelsAggregatedCommand(self.state),
            ModelsReadyCommand(self.state),
            MetricsCommand(self.state),
            InitModelCommand(
                self.state,
                self.stop,
                self.aggregator,
                self._communication_protocol,
            ),
            AddModelCommand(
                self.state,
                self.stop,
                self.aggregator,
                self._communication_protocol,
            ),
        ]
        self._communication_protocol.add_command(commands)  # no esta en la interfaz

    #############################
    #  Neighborhood management  #
    #############################

    def connect(self, addr: str) -> bool:
        """
        Connect a node to another.

        Warning:
            Adding nodes while learning is running is not fully supported.

        Args:
            addr: The address of the node to connect to.

        Returns:
            True if the node was connected, False otherwise.

        """
        # Check running
        self.assert_running(True)
        # Connect
        logger.info(self.addr, f"Connecting to {addr}...")
        return self._communication_protocol.connect(addr)

    def get_neighbors(self, only_direct: bool = False) -> Dict[str, Any]:
        """
        Return the neighbors of the node.

        Args:
            only_direct: If True, only the direct neighbors will be returned.

        Returns:
            The list of neighbors.

        """
        return self._communication_protocol.get_neighbors(only_direct)

    def disconnect(self, addr: str) -> None:
        """
        Disconnects a node from another.

        Args:
            addr: The address of the node to disconnect from.

        """
        # Check running
        self.assert_running(True)
        # Disconnect
        logger.info(self.addr, f"Removing {addr}...")
        self._communication_protocol.disconnect(addr, disconnect_msg=True)

    #######################################
    #   Node Management (servicer loop)   #
    #######################################

    def assert_running(self, running: bool) -> None:
        """
        Assert that the node is running or not running.

        Args:
            running: True if the node must be running, False otherwise.

        Raises:
            NodeRunningException: If the node is not running and running is True, or if the node is running and running
            is False.

        """
        running_state = self.__running
        if running_state != running:
            raise NodeRunningException(f"Node is {'not ' if running_state else ''}running.")

    def start(self, wait: bool = False) -> None:
        """
        Start the node: server and neighbors(gossip and heartbeat).

        Args:
            wait: If True, the function will wait until the server is terminated.

        Raises:
            NodeRunningException: If the node is already running.

        """
        # Check not running
        self.assert_running(False)
        # Set running
        self.__running = True
        # P2PFL Web Services
        logger.register_node(self.addr, self.state, self.state.simulation)
        # Communication Protocol
        self._communication_protocol.start()
        if wait:
            self._communication_protocol.wait_for_termination()
            logger.info(self.addr, "gRPC terminated.")

    def stop(self) -> None:
        """
        Stop the node: server and neighbors(gossip and heartbeat).

        Raises:
            NodeRunningException: If the node is not running.

        """
        logger.info(self.addr, "Stopping node...")
        try:
            # Stop server
            self._communication_protocol.stop()
            # Set not running
            self.__running = False
            # State
            self.state.clear()
            # Unregister node
            logger.unregister_node(self.addr)
        except Exception:
            pass

    ##########################
    #    Learning Setters    #
    ##########################

    def set_data(self, data) -> None:
        """
        Set the data to be used in the learning process (by the learner).

        Args:
            data: Dataset to be used in the learning process.

        Raises:
            LearnerNotSetException: If the learner is already set.

        """
        self.data = data
        # If learner is already set (raise)
        if self.state.learner is not None:
            raise LearnerNotSetException("Data cannot be set after learner is set.")

    def set_model(self, model) -> None:
        """
        Set the model to be used in the learning process (by the learner).

        Args:
            model: Model to be used in the learning process.

        Raises:
            LearnerNotSetException: If the learner is already set.

        """
        self.model = model
        # If learner is already set (raise)
        if self.state.learner is not None:
            raise LearnerNotSetException("Model cannot be set after learner is set.")

    ###############################################
    #         Network Learning Management         #
    ###############################################

    def __start_learning_thread(self, rounds: int, epochs: int) -> None:
        learning_thread = threading.Thread(
            target=self.__start_learning,
            args=(rounds, epochs),
            name="learning_thread-" + self.addr,
        )
        learning_thread.daemon = True
        learning_thread.start()

    def set_start_learning(self, rounds: int = 1, epochs: int = 1) -> None:
        """
        Start the learning process in the entire network.

        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.

        Raises:
            ZeroRoundsException: If rounds is less than 1.

        """
        self.assert_running(True)

        if rounds < 1:
            raise ZeroRoundsException("Rounds must be greater than 0.")

        if self.state.round is None:
            # Broadcast start Learning
            logger.info(self.addr, "Broadcasting start learning...")
            self._communication_protocol.broadcast(
                self._communication_protocol.build_msg(StartLearningCommand.get_name(), [str(rounds), str(epochs)])
            )
            # Set model initialized
            self.state.model_initialized_lock.release()
            # Broadcast initialize model
            self._communication_protocol.broadcast(
                self._communication_protocol.build_msg(ModelInitializedCommand.get_name())
            )
            # Learning Thread
            self.__start_learning_thread(rounds, epochs)
        else:
            logger.info(self.addr, "Learning already started")

    def set_stop_learning(self) -> None:
        """Stop the learning process in the entire network."""
        if self.state.round is not None:
            # send stop msg
            self._communication_protocol.broadcast(
                self._communication_protocol.build_msg(StopLearningCommand.get_name())
            )
            # stop learning
            self.__stop_learning()
        else:
            logger.info(self.addr, "Learning already stopped")

    ##################################
    #         Local Learning         #
    ##################################

    def __start_learning(self, rounds: int, epochs: int) -> None:
        try:
            self.learning_workflow.run(
                rounds=rounds,
                epochs=epochs,
                state=self.state,
                model=self.model,
                data=self.data,
                communication_protocol=self._communication_protocol,
                early_stopping_fn=lambda: self.state.round is None,
                aggregator=self.aggregator,
                learner_class=self.learner_class,
            )
        except Exception as e:
            if logger.get_level_name(logger.get_level()) == "DEBUG":
                raise e
            logger.error(self.addr, f"Error: {e}")
            self.stop()

    def __stop_learning(self) -> None:
        logger.info(self.addr, "Stopping learning")
        # Leraner
        if self.state.learner is not None:
            self.state.learner.interrupt_fit()
        # Aggregator
        self.aggregator.clear()
        # State
        self.state.clear()
        logger.experiment_finished(self.addr)
        # Try to free wait locks
        with contextlib.suppress(Exception):
            self.state.wait_votes_ready_lock.release()
