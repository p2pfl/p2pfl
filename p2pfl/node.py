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

import time
import random
import math
import threading
from typing import Dict, List, Any, Type

from p2pfl.commands.init_model_command import InitModelCommand
from p2pfl.settings import Settings
from p2pfl.commands.metrics_command import MetricsCommand
from p2pfl.commands.model_initialized_command import ModelInitializedCommand
from p2pfl.commands.models_agregated_command import ModelsAggregatedCommand
from p2pfl.commands.models_ready_command import ModelsReadyCommand
from p2pfl.commands.start_learning_command import StartLearningCommand
from p2pfl.commands.stop_learning_command import StopLearningCommand
from p2pfl.commands.vote_train_set_command import VoteTrainSetCommand
from p2pfl.commands.add_model_command import AddModelCommand
from p2pfl.node_state import NodeState
from p2pfl.management.logger import logger
from p2pfl.learning.learner import NodeLearner
from p2pfl.learning.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.communication.communication_protocol import CommunicationProtocol
from p2pfl.communication.grpc.communication_protocol import GrpcCommunicationProtocol


"""
- revisar agregación de nodos en caliente
- revisar logging en general
- tiene sentido que lo del aprendizaje esté en este nodo!
- patrón estado al nodo
    - al final es algo secuencial: inicialización, votado, entrenamiento, agregación, ...
- model gossip provisional (hard-coded, se necesita mover el model gossiper)
"""

class Node:

    #####################
    #     Node Init     #
    #####################

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

        # Communication protol
        self._communication_protocol = protocol(address)
        self.addr = self._communication_protocol.get_address()

        # Learning
        self.data = data
        self.model = model
        self.learner_class = learner
        self.aggregator = aggregator(node_name=self.addr)  # Ponerlo como learner (que se vaya instanciando dinamicamente)

        # State
        self.__running = False
        self.state = NodeState(self.addr)

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
        Connects a node to another.

        > Careful: Adding nodes while learning is running is not fully supported.

        Args:
            addr (str): The address of the node to connect to.

        Returns:
            bool: True if the node was connected, False otherwise.
        """
        # Check running
        self.assert_running(True)
        # Connect
        logger.info(self.addr, f"Connecting to {addr}...")
        return self._communication_protocol.connect(addr)

    def get_neighbors(self, only_direct: bool = False) -> List[str]:
        """
        Returns the neighbors of the node.

        Args:
            only_direct (bool): If True, only the direct neighbors will be returned.

        Returns:
            list: The list of neighbors.
        """
        return self._communication_protocol.get_neighbors(only_direct)

    def disconnect(self, addr: str) -> None:
        """
        Disconnects a node from another.

        Args:
            addr (str): The address of the node to disconnect from.
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
        Asserts that the node is running or not running.

        Args:
            running (bool): True if the node must be running, False otherwise.

        Raises:
            Exception: If the node is not running and running is True, or if the node is running and running is False.
        """
        running_state = self.__running
        if running_state != running:
            raise Exception(f"Node is {'not ' if running_state else ''}running.")

    def start(self, wait: bool = False) -> None:
        """
        Starts the node: server and neighbors(gossip and heartbeat).

        Args:
            wait (bool): If True, the function will wait until the server is terminated.

        Raises:
            Exception: If the node is already running.
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
        Stops the node: server and neighbors(gossip and heartbeat).

        Raises:
            Exception: If the node is not running.
        """
        logger.info(self.addr, "Stopping node...")
        # Check running
        self.assert_running(True)
        # Stop server
        self._communication_protocol.stop()
        # Set not running
        self.__running = False
        # State
        self.state.clear()
        # Unregister node
        logger.unregister_node(self.addr)

    ##########################
    #    Learning Setters    #
    ##########################

    def set_data(self, data) -> None:
        """
        Set the data to be used in the learning process (by the learner).

        Args:
            data: Dataset to be used in the learning process.
        """
        self.data = data
        self.state.learner.set_data(data)

    def set_model(self, model) -> None:
        """
        Set the model to be used in the learning process (by the learner).

        Args:
            model: Model to be used in the learning process.
        """
        self.model = model
        self.state.learner.set_model(model)

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
        """
        self.assert_running(True)

        if rounds < 1:
            raise Exception("Rounds and epochs must be greater than 0.")

        if self.state.round is None:
            # Broadcast start Learning
            logger.info(self.addr, "Broadcasting start learning...")
            self._communication_protocol.broadcast(
                self._communication_protocol.build_msg(
                    StartLearningCommand.get_name(), [str(rounds), str(epochs)]
                )
            )
            # Set model initialized
            self.state.model_initialized_lock.release()
            # Broadcast initialize model
            self._communication_protocol.broadcast(
                self._communication_protocol.build_msg(
                    ModelInitializedCommand.get_name()
                )
            )
            # Learning Thread
            self.__start_learning_thread(rounds, epochs)
        else:
            logger.info(self.addr, "Learning already started")

    def set_stop_learning(self) -> None:
        """
        Stop the learning process in the entire network.
        """
        if self.state.round is not None:
            # send stop msg
            self._communication_protocol.broadcast(
                self._communication_protocol.build_msg(
                    StopLearningCommand.get_name()
                )
            )
            # stop learning
            self.__stop_learning()
        else:
            logger.info(self.addr, "Learning already stopped")

    ##################################
    #         Local Learning         #
    ##################################

    """
    revisar a partir de aqui
    """

    def __start_learning(self, rounds: int, epochs: int) -> None:
        self.state.start_thread_lock.acquire()  # Used to avoid create duplicated training threads
        if self.state.round is None:
            # Init
            self.state.set_experiment("experiment", rounds)
            logger.experiment_started(self.addr)
            self.state.learner = self.learner_class(self.model, self.data, self.addr, epochs)
            self.state.start_thread_lock.release()
            begin = time.time()

            # Wait and gossip model inicialization
            logger.info(self.addr, "Waiting initialization.")
            self.state.model_initialized_lock.acquire()
            logger.info(self.addr, "Gossiping model initialization.")
            self.__gossip_model_difusion(initialization=True)

            # Wait to guarantee new connection heartbeats convergence
            wait_time = Settings.WAIT_HEARTBEATS_CONVERGENCE - (time.time() - begin)
            if wait_time > 0:
                time.sleep(wait_time)

            # Train
            self.__train_step()
        else:
            self.state.start_thread_lock.release()

    def __stop_learning(self) -> None:
        logger.info(self.addr, "Stopping learning")
        # Leraner
        self.state.learner.interrupt_fit()
        # Aggregator
        self.aggregator.clear()
        # State
        self.state.clear()
        logger.experiment_finished(self.addr)
        # Try to free wait locks
        try:
            self.state.wait_votes_ready_lock.release()
        except Exception:
            pass

    ########################
    #    Training Steps    #
    ########################

    def __train_step(self) -> None:
        # Set train set
        if self.state.round is not None:
            self.state.train_set = self.__vote_train_set()
            self.state.train_set = self.__validate_train_set(self.state.train_set)
            logger.info(
                self.addr,
                f"Train set of {len(self.state.train_set)} nodes: {self.state.train_set}",
            )

        # Determine if node is in the train set
        if self.addr in self.state.train_set:
            # Set nodes to agg
            if self.state.round is not None:
                # Set Models To Aggregate
                self.aggregator.set_nodes_to_aggregate(self.state.train_set)

            # Evaluate and send metrics
            if self.state.round is not None:
                self.__evaluate()

            # Train
            if self.state.round is not None:
                self.__train()

            # Aggregate Model
            if self.state.round is not None:
                models_added = self.aggregator.add_model(
                    self.state.learner.get_parameters(),
                    [self.addr],
                    self.state.learner.get_num_samples()[0],
                )
                # send model added msg ---->> redundant (a node always owns its model)
                self._communication_protocol.broadcast(
                    self._communication_protocol.build_msg(
                        ModelsAggregatedCommand.get_name(), models_added, round=self.state.round
                    )
                )
                self.__gossip_model_aggregation()

        else:
            # Set Models To Aggregate
            logger.info(self.addr, "Waiting aregation.")
            self.aggregator.set_waiting_aggregated_model(self.state.train_set)

        # Gossip aggregated model (also syncrhonizes nodes)
        if self.state.round is not None:
            self.__wait_aggregated_model()
            self.__gossip_model_difusion()

        # Finish round
        if self.state.round is not None:
            self.__on_round_finished()

    def __wait_aggregated_model(self) -> None:
        params = self.aggregator.wait_and_get_aggregation()

        # Set parameters and communate it to the training process
        if params is not None:
            self.state.learner.set_parameters(params)
            logger.debug(
                self.addr, f"Broadcast aggregation done for round {self.state.round}"
            )
            # Share that aggregation is done
            self._communication_protocol.broadcast(
                self._communication_protocol.build_msg(
                    ModelsReadyCommand.get_name(), [], round=self.state.round
                )
            )
        else:
            logger.error(self.addr, "Aggregation finished with no parameters")
            self.stop()

    ################
    #    Voting    #
    ################

    def __vote_train_set(self) -> List[str]:
        # Vote (at least itself)
        candidates = list(self.get_neighbors(only_direct=False))
        if self.addr not in candidates:
            candidates.append(self.addr)
        logger.debug(self.addr, f"{len(candidates)} candidates to train set")

        # Send vote
        samples = min(Settings.TRAIN_SET_SIZE, len(candidates))
        nodes_voted = random.sample(candidates, samples)
        weights = [
            math.floor(random.randint(0, 1000) / (i + 1)) for i in range(samples)
        ]
        votes = list(zip(nodes_voted, weights))

        # Adding votes
        self.state.train_set_votes_lock.acquire()
        self.state.train_set_votes[self.addr] = dict(votes)
        self.state.train_set_votes_lock.release()

        # Send and wait for votes
        logger.info(self.addr, "Sending train set vote.")
        logger.debug(self.addr, f"Self Vote: {votes}")
        self._communication_protocol.broadcast(
            self._communication_protocol.build_msg(
                VoteTrainSetCommand.get_name(),
                list(map(str, list(sum(votes, tuple())))),
                round=self.state.round,
            )
        )
        logger.debug(self.addr, "Waiting other node votes.")

        # Get time
        count = 0.0
        begin = time.time()

        while True:
            # If the trainning has been interrupted, stop waiting
            if self.state.round is None:
                logger.info(self.addr, "Stopping on_round_finished process.")
                return []

            # Update time counters (timeout)
            count = count + (time.time() - begin)
            begin = time.time()
            timeout = count > Settings.VOTE_TIMEOUT

            # Clear non candidate votes
            self.state.train_set_votes_lock.acquire()
            nc_votes = {
                k: v for k, v in self.state.train_set_votes.items() if k in self.get_neighbors(only_direct=False)
            }
            self.state.train_set_votes_lock.release()

            # Determine if all votes are received
            votes_ready = set(self.get_neighbors(only_direct=False)) == set(nc_votes.keys())
            if votes_ready or timeout:
                if timeout and not votes_ready:
                    logger.info(
                        self.addr,
                        f"Timeout for vote aggregation. Missing votes from {set(self.get_neighbors(only_direct=False)) - set(nc_votes.keys())}",
                    )

                results: Dict[str, int] = {}
                for node_vote in list(nc_votes.values()):
                    for i in range(len(node_vote)):
                        k = list(node_vote.keys())[i]
                        v = list(node_vote.values())[i]
                        if k in results:
                            results[k] += v
                        else:
                            results[k] = v

                # Order by votes and get TOP X
                results_ordered = sorted(
                    results.items(), key=lambda x: x[0], reverse=True
                )  # to equal solve of draw (node name alphabetical order)
                results_ordered = sorted(
                    results_ordered, key=lambda x: x[1], reverse=True
                )
                top = min(len(results_ordered), Settings.TRAIN_SET_SIZE)
                results_ordered = results_ordered[0:top]

                # Clear votes
                self.state.train_set_votes = {}
                logger.info(self.addr, f"Computed {len(nc_votes)} votes.")
                return [i[0] for i in results_ordered]

            # Wait for votes or refresh every 2 seconds
            self.state.wait_votes_ready_lock.acquire(timeout=2)

    def __validate_train_set(self, train_set: List[str]) -> List[str]:
        # Verify if node set is valid (can happend that a node was down when the votes were being processed)
        for tsn in train_set:
            if tsn not in self.get_neighbors(only_direct=False):
                if tsn != self.addr:
                    train_set.remove(tsn)
        return train_set

    ############################
    #    Train and Evaluate    #
    ############################

    def __train(self) -> None:
        logger.info(self.addr, "Training...")
        self.state.learner.fit()

    def __evaluate(self) -> None:
        logger.info(self.addr, "Evaluating...")
        results = self.state.learner.evaluate()
        logger.info(self.addr, f"Evaluated. Results: {results}")
        # Send metrics
        if len(results) > 0:
            logger.info(self.addr, "Broadcasting metrics.")
            flattened_metrics = [item for pair in results.items() for item in pair]
            self._communication_protocol.broadcast(
                self._communication_protocol.build_msg(
                    MetricsCommand.get_name(),
                    flattened_metrics,
                    round=self.state.round,
                )
            )

    ######################
    #    Round finish    #
    ######################

    def __on_round_finished(self) -> None:
        # Check if learning is running
        if self.state.round is None:
            raise Exception("Round finished when learning is not running")

        # Set Next Round
        self.aggregator.clear()
        self.state.increase_round()
        logger.round_finished(self.addr)

        # Clear node aggregation
        self.state.models_aggregated = {}

        # Next Step or Finish
        logger.info(
            self.addr,
            f"Round {self.state.round} of {self.state.total_rounds} finished.",
        )
        if self.state.round < self.state.total_rounds:
            self.__train_step()
        else:
            # At end, all nodes compute metrics
            self.__evaluate()
            # Finish
            self.state.clear()
            self.state.model_initialized_lock.acquire()
            logger.info(self.addr, "Training finished!!.")

    #########################
    #    Model Gossiping    #
    #########################

    def get_aggregated_models(self, node: str) -> List[str]:
        """
        Get the models that have been aggregated by a given node in the actual round.

        Args:
            node (str): Node to get the aggregated models from.
        """
        try:
            return self.state.models_aggregated[node]
        except KeyError:
            return []

    def __gossip_model_aggregation(self) -> None:
        """
        CAREFULL: full connected trainset to increase aggregation speed. On real scenarios, this won't be possible, private networks and firewalls.
        Needed because the trainset can split the networks (and neighbors that are not in the trainset won't receive the aggregation).
        """
        # Anonymous functions
        def early_stopping_fn():
            return self.state.round is None

        def get_candidates_fn() -> List[str]:
            return [
                n for n in self.get_neighbors(only_direct=False) if (n not in self.aggregator.get_aggregated_models()) and (n in self.state.train_set)
            ]

        def status_fn() -> Any:
            return [
                (n, self.get_aggregated_models(n)) for n in self.get_neighbors(only_direct=False) if (n in self.state.train_set)
            ]

        def model_fn(node: str) -> Any:
            model, contributors, weight = self.aggregator.get_partial_aggregation(
                self.get_aggregated_models(node)
            )
            if model is None:
                return None
            encoded_model = self.state.learner.encode_parameters(params=model)
            return self._communication_protocol.build_weights(
                AddModelCommand.get_name(),
                self.state.round,
                encoded_model,
                contributors,
                weight,
            )

        # Gossip
        self._communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
            create_connection=True
        )

    def __gossip_model_difusion(self, initialization: bool = False) -> None:
        def early_stopping_fn():
            return self.state.round is None

        # Wait a model (init or aggregated)
        if initialization:
            def candidate_condition(node: str) -> bool:
                return node not in self.state.nei_status.keys()

        else:
            logger.info(self.addr, "Gossiping aggregated model.")
            fixed_round = self.state.round

            def candidate_condition(node: str) -> bool:
                return self.state.nei_status[node] < fixed_round

        def get_candidates_fn() -> List[str]:
            return [
                n for n in self.get_neighbors(only_direct=True) if candidate_condition(n)
            ]

        def status_fn() -> Any:
            return get_candidates_fn()

        def model_fn(node: str) -> Any:
            model = self.state.learner.get_parameters()
            contributors = self.aggregator.get_aggregated_models()
            weight = 1
            encoded_model = self.state.learner.encode_parameters(params=model)
            return self._communication_protocol.build_weights(
                InitModelCommand.get_name() if initialization else AddModelCommand.get_name(),
                self.state.round,
                encoded_model,
                contributors,
                weight,
            )

        # Gossip
        self._communication_protocol.gossip_weights(
            early_stopping_fn,
            get_candidates_fn,
            status_fn,
            model_fn,
        )
