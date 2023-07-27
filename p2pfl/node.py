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

import math
import random
import threading
import logging
import time
from p2pfl.base_node import BaseNode
from p2pfl.messages import LearningNodeMessages
from p2pfl.settings import Settings
from p2pfl.learning.pytorch.lightninglearner import LightningLearner
from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.exceptions import DecodingParamsError, ModelNotMatchingError
import p2pfl.proto.node_pb2 as node_pb2


class Node(BaseNode):
    """
    Class based on a BaseNode that allows **p2p Federated Learning**.

    Args:
        model: Model to be learned. Careful, model should be compatible with data and the learner.
        data: Dataset to be used in the learning process. Careful, model should be compatible with data and the learner.
        host (str): Host where the node will be listening.
        port (int): Port where the node will be listening.
        learner (NodeLearner): Learner to be used in the learning process. Default: LightningLearner.
        aggregator (Aggregator): Aggregator to be used in the learning process. Default: FedAvg.
        simulation (bool): If True, the node will be created in simulation mode. Default: False.
    Attributes:
        round (int): Round of the learning process.
        totalrounds (int): Total number of rounds of the learning process.
        learner (NodeLearner): Learner to be used in the learning process.
        aggregator (Aggregator): Aggregator to be used in the learning process.
    """

    #####################
    #     Node Init     #
    #####################

    def __init__(
        self,
        model,
        data,
        host="127.0.0.1",
        port=None,
        learner=LightningLearner,
        aggregator=FedAvg,
        simulation=False,
    ):
        # Super init
        BaseNode.__init__(self, host, port, simulation)

        # Add message handlers
        self.add_message_handler(
            LearningNodeMessages.START_LEARNING, self.__start_learning_callback
        )
        self.add_message_handler(
            LearningNodeMessages.STOP_LEARNING, self.__stop_learning_callback
        )
        self.add_message_handler(
            LearningNodeMessages.MODEL_INITIALIZED, self.__model_initialized_callback
        )
        self.add_message_handler(
            LearningNodeMessages.VOTE_TRAIN_SET, self.__vote_train_set_callback
        )
        self.add_message_handler(
            LearningNodeMessages.MODELS_AGGREGATED, self.__models_agregated_callback
        )
        self.add_message_handler(
            LearningNodeMessages.MODELS_READY, self.__models_ready_callback
        )
        self.add_message_handler(LearningNodeMessages.METRICS, self.__metrics_callback)

        # Learning
        self.round = None
        self.totalrounds = None
        self.__train_set = []
        self.__models_agregated = {}
        self.__nei_status = {}
        self.learner = learner(model, data, self.addr)
        self.aggregator = aggregator(node_name=self.addr)

        # Train Set Votes
        self.__train_set_votes = {}
        self.__train_set_votes_lock = threading.Lock()

        # Locks
        self.__start_thread_lock = threading.Lock()
        self.__wait_votes_ready_lock = threading.Lock()
        self.__model_initialized_lock = threading.Lock()
        self.__model_initialized_lock.acquire()

    ######################
    #    Msg Handlers    #
    ######################

    def __start_learning_callback(self, msg):
        self.__start_learning_thread(int(msg.args[0]), int(msg.args[1]))

    def __stop_learning_callback(self, _):
        self.__stop_learning()

    def __model_initialized_callback(self, msg):
        self.__nei_status[msg.source] = -1

    def __vote_train_set_callback(self, msg):
        if msg.round == self.round:
            # build vote dict
            votes = msg.args
            tmp_votes = {}
            for i in range(0, len(votes), 2):
                tmp_votes[votes[i]] = int(votes[i + 1])
            # set votes
            self.__train_set_votes_lock.acquire()
            self.__train_set_votes[msg.source] = tmp_votes
            self.__train_set_votes_lock.release()
            # Communicate to the training process that a vote has been received
            try:
                self.__wait_votes_ready_lock.release()
            except:
                pass

    def __models_agregated_callback(self, msg):
        if msg.round == self.round:
            self.__models_agregated[msg.source] = msg.args

    def __models_ready_callback(self, msg):
        if msg.round == self.round:
            self.__nei_status[msg.source] = int(msg.args[0])

    def __metrics_callback(self, msg):
        name = msg.source
        round = msg.round
        loss, metric = msg.args[0:2]
        loss = float(loss)
        self.learner.log_validation_metrics(loss, metric, round=round, name=name)

    ############################
    #  GRPC - Remote Services  #
    ############################

    def add_model(self, request, _):
        """
        GRPC service. It is called when a node wants to add a model to the network.
        """
        # Check if Learning is running
        if self.round is not None:
            # Check source
            if request.round != self.round:
                logging.error(
                    f"({self.addr}) Model Reception in a late round ({request.round} != {self.round})."
                )
                return node_pb2.ResponseMessage()

            # Check moment (not init and invalid round)
            if (
                not self.__model_initialized_lock.locked()
                and len(self.__train_set) == 0
            ):
                logging.error(
                    f"({self.addr}) Model Reception when there is no trainset"
                )
                return node_pb2.ResponseMessage()

            try:
                if not self.__model_initialized_lock.locked():
                    # Add model to aggregator
                    decoded_model = self.learner.decode_parameters(request.weights)
                    if self.learner.check_parameters(decoded_model):
                        models_added = self.aggregator.add_model(
                            decoded_model, request.contributors, request.weight
                        )
                        if models_added is not None:
                            # Communicate Aggregation
                            self._neighbors.broadcast_msg(
                                self._neighbors.build_msg(
                                    LearningNodeMessages.MODELS_AGGREGATED, models_added
                                )
                            )
                    else:
                        raise ModelNotMatchingError("Not matching models")
                else:
                    # Initialize model (try to handle concurrent initializations)
                    try:
                        self.__model_initialized_lock.release()
                        model = self.learner.decode_parameters(request.weights)
                        self.learner.set_parameters(model)
                        logging.info(f"({self.addr}) Model Weights Initialized")
                        # Communicate Initialization
                        self._neighbors.broadcast_msg(
                            self._neighbors.build_msg(
                                LearningNodeMessages.MODEL_INITIALIZED
                            )
                        )
                    except RuntimeError:
                        # unlock unlocked lock
                        pass

            # Warning: these stops can cause a denegation of service attack
            except DecodingParamsError as e:
                logging.error(f"({self.addr}) Error decoding parameters.")
                self.stop()

            except ModelNotMatchingError as e:
                logging.error(f"({self.addr}) Models not matching.")
                self.stop()

            except Exception as e:
                logging.error(f"({self.addr}) Unknown error adding model: {e}")
                self.stop()

        else:
            logging.debug(
                f"({self.addr}) Tried to add a model while learning is not running"
            )

        # Response
        return node_pb2.ResponseMessage()

    def handshake(self, request, _):
        """
        GRPC service. It is called when a node connects to another.
        """
        if self.round is not None:
            logging.info(
                f"({self.addr}) Cant connect to other nodes when learning is running."
            )
            return node_pb2.ResponseMessage(error="Cant connect: learning is running")
        else:
            return super().handshake(request, _)

    #########################
    #    Node Management    #
    #########################

    def connect(self, addr):
        """
        Connects a node to another. If learning is running, connections are not allowed.

        Args:
            addr (str): Address of the node to connect to.

        Returns:
            bool: True if the connection was successful, False otherwise.
        """
        # Check if learning is running
        if self.round is not None:
            logging.info(
                f"({self.addr}) Cant connect to other nodes when learning is running."
            )
            return False
        # Connect
        return super().connect(addr)

    def stop(self):
        """
        Stops the node. If learning is running, the local learning process is interrupted.
        """
        # Interrupt learning
        if self.round is not None:
            self.__stop_learning()
        # Close node
        super().stop()

    ##########################
    #    Learning Setters    #
    ##########################

    def set_data(self, data):
        """
        Set the data to be used in the learning process (by the learner).

        Args:
            data: Dataset to be used in the learning process.
        """
        self.learner.set_data(data)

    def set_model(self, model):
        """
        Set the model to be used in the learning process (by the learner).

        Args:
            model: Model to be used in the learning process.
        """
        self.learner.set_model(model)

    ###############################################
    #         Network Learning Management         #
    ###############################################

    def set_start_learning(self, rounds=1, epochs=1):
        """
        Start the learning process in the entire network.

        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
        """
        self.assert_running(True)

        if self.round is None:
            # Broadcast start Learning
            logging.info(f"({self.addr}) Broadcasting start learning...")
            self._neighbors.broadcast_msg(
                self._neighbors.build_msg(
                    LearningNodeMessages.START_LEARNING, [rounds, epochs]
                )
            )
            # Set model initializated
            self.__model_initialized_lock.release()
            # Broadcast initialize model
            self._neighbors.broadcast_msg(
                self._neighbors.build_msg(LearningNodeMessages.MODEL_INITIALIZED)
            )
            # Learning Thread
            self.__start_learning_thread(rounds, epochs)
        else:
            logging.info(f"({self.addr}) Learning already started")

    def set_stop_learning(self):
        """
        Stop the learning process in the entire network.
        """
        if self.round is not None:
            # send stop msg
            self._neighbors.broadcast_msg(
                self._neighbors.build_msg(LearningNodeMessages.STOP_LEARNING)
            )
            # stop learning
            self.__stop_learning()
        else:
            logging.info(f"({self.addr}) Learning already stopped")

    ##################################
    #         Local Learning         #
    ##################################

    def __start_learning_thread(self, rounds, epochs):
        learning_thread = threading.Thread(
            target=self.__start_learning, args=(rounds, epochs)
        )
        learning_thread.name = "learning_thread-" + self.addr
        learning_thread.daemon = True
        learning_thread.start()

    def __start_learning(self, rounds, epochs):
        self.__start_thread_lock.acquire()  # Used to avoid create duplicated training threads
        if self.round is None:
            self.round = 0
            self.totalrounds = rounds
            self.learner.create_new_exp()
            self.__start_thread_lock.release()
            begin = time.time()

            # Wait and gossip model inicialization
            logging.info(f"({self.addr}) Waiting initialization.")
            self.__model_initialized_lock.acquire()
            logging.info(f"({self.addr}) Gossiping model initialization.")
            self.__gossip_model_difusion(initialization=True)

            # Wait to guarantee new connection heartbeats convergence
            wait_time = Settings.WAIT_HEARTBEATS_CONVERGENCE - (time.time() - begin)
            if wait_time > 0:
                time.sleep(wait_time)

            # Train
            self.learner.set_epochs(epochs)
            self.__train_step()
        else:
            self.__start_thread_lock.release()

    def __stop_learning(self):
        logging.info(f"({self.addr}) Stopping learning")
        # Rounds
        self.round = None
        self.totalrounds = None
        # Leraner
        self.learner.interrupt_fit()
        # Aggregator
        self.aggregator.clear()
        # Try to free wait locks
        try:
            self.__wait_votes_ready_lock.release()
        except:
            pass

    #######################
    #    Trainig Steps    #
    #######################

    def __wait_aggregated_model(self):
        logging.info(f"({self.addr}) Waiting aregation.")

        params = self.aggregator.wait_and_get_aggregation()

        # Set parameters and communate it to the training process
        if params is not None:
            self.learner.set_parameters(params)
            logging.debug(
                f"({self.addr}) Broadcast aggregation done for round {self.round}"
            )
            # Share that aggregation is done
            self._neighbors.broadcast_msg(
                self._neighbors.build_msg(
                    LearningNodeMessages.MODELS_READY, [self.round]
                )
            )
        else:
            logging.error(f"({self.addr}) Aggregation finished with no parameters")
            self.stop()

    def __train_step(self):
        # Set train set
        if self.round is not None:
            self.__train_set = self.__vote_train_set()
            self.__train_set = self.__validate_train_set(self.__train_set)
            logging.info(f"{self.addr} Train set of {len(self.__train_set)} nodes: {self.__train_set}")

        # Determine if node is in the train set
        if self.addr in self.__train_set:
            # Full connect train set
            if self.round is not None:
                # Set Models To Aggregate
                self.aggregator.set_nodes_to_aggregate(self.__train_set)

            # Evaluate and send metrics
            if self.round is not None:
                self.__evaluate()

            # Train
            if self.round is not None:
                self.__train()

            # Aggregate Model
            if self.round is not None:
                models_added = self.aggregator.add_model(
                    self.learner.get_parameters(),
                    [self.addr],
                    self.learner.get_num_samples()[0],
                )
                # send model added msg ---->> redundant (a node always owns its model)
                self._neighbors.broadcast_msg(
                    self._neighbors.build_msg(
                        LearningNodeMessages.MODELS_AGGREGATED, models_added
                    )
                )
                self.__gossip_model_aggregation()

        else:
            # Set Models To Aggregate
            self.aggregator.set_waiting_aggregated_model()

        # Gossip aggregated model (also syncrhonizes nodes)
        if self.round is not None:
            self.__wait_aggregated_model()
            self.__gossip_model_difusion()

        # Finish round
        if self.round is not None:
            self.__on_round_finished()

    ################
    #    Voting    #
    ################

    def __vote_train_set(self):
        # Vote (at least itself)
        candidates = list(self.get_neighbors(only_direct=False))
        if self.addr not in candidates:
            candidates.append(self.addr)
        logging.debug(f"({self.addr}) {len(candidates)} candidates to train set")

        # Send vote
        samples = min(Settings.TRAIN_SET_SIZE, len(candidates))
        nodes_voted = random.sample(candidates, samples)
        weights = [
            math.floor(random.randint(0, 1000) / (i + 1)) for i in range(samples)
        ]
        votes = list(zip(nodes_voted, weights))

        # Adding votes
        self.__train_set_votes_lock.acquire()
        self.__train_set_votes[self.addr] = dict(votes)
        self.__train_set_votes_lock.release()

        # Send and wait for votes
        logging.info(f"({self.addr}) Sending train set vote.")
        logging.debug(f"({self.addr}) Self Vote: {votes}")
        self._neighbors.broadcast_msg(
            self._neighbors.build_msg(
                LearningNodeMessages.VOTE_TRAIN_SET, list(sum(votes, tuple()))
            )
        )
        logging.debug(f"({self.addr}) Waiting other node votes.")

        # Get time
        count = 0
        begin = time.time()

        while True:
            # If the trainning has been interrupted, stop waiting
            if self.round is None:
                logging.info(f"({self.addr}) Stopping on_round_finished process.")
                return []

            # Update time counters (timeout)
            count = count + (time.time() - begin)
            begin = time.time()
            timeout = count > Settings.VOTE_TIMEOUT

            # Clear non candidate votes
            self.__train_set_votes_lock.acquire()
            nc_votes = {
                k: v for k, v in self.__train_set_votes.items() if k in candidates
            }
            self.__train_set_votes_lock.release()

            # Determine if all votes are received
            votes_ready = set(candidates) == set(nc_votes.keys())
            if votes_ready or timeout:
                if timeout and not votes_ready:
                    logging.info(
                        f"({self.addr}) Timeout for vote aggregation. Missing votes from {set(candidates) - set(nc_votes.keys())}"
                    )

                results = {}
                for node_vote in list(nc_votes.values()):
                    for i in range(len(node_vote)):
                        k = list(node_vote.keys())[i]
                        v = list(node_vote.values())[i]
                        if k in results:
                            results[k] += v
                        else:
                            results[k] = v

                # Order by votes and get TOP X
                results = sorted(
                    results.items(), key=lambda x: x[0], reverse=True
                )  # to equal solve of draw
                results = sorted(results, key=lambda x: x[1], reverse=True)
                top = min(len(results), Settings.TRAIN_SET_SIZE)
                results = results[0:top]
                results = {k: v for k, v in results}
                votes = list(results.keys())

                # Clear votes
                self.__train_set_votes = {}
                logging.info(f"({self.addr}) Computed {len(nc_votes)} votes.")
                return votes

            # Wait for votes or refresh every 2 seconds
            self.__wait_votes_ready_lock.acquire(timeout=2)

    def __validate_train_set(self, train_set):
        # Verify if node set is valid (can happend that a node was down when the votes were being processed)
        for tsn in train_set:
            if tsn not in self.get_neighbors(only_direct=False):
                if tsn != self.addr:
                    train_set.remove(tsn)
        return train_set

    ############################
    #    Train and Evaluate    #
    ############################

    def __train(self):
        logging.info(f"({self.addr}) Training...")
        self.learner.fit()

    def __evaluate(self):
        logging.info(f"({self.addr}) Evaluating...")
        results = self.learner.evaluate()
        if results is not None:
            logging.info(
                f"({self.addr}) Evaluated. Losss: {results[0]}, Metric: {results[1]}."
            )
            # Send metrics
            logging.info(f"({self.addr}) Broadcasting metrics.")
            self._neighbors.broadcast_msg(
                self._neighbors.build_msg(
                    LearningNodeMessages.METRICS,
                    [results[0], results[1]],
                    round=self.round,
                )
            )

    ######################
    #    Round finish    #
    ######################

    def __on_round_finished(self):
        # Set Next Round
        self.aggregator.clear()
        self.learner.finalize_round()  # revisar x si esto pueiera quedar mejor
        self.round = self.round + 1
        # Clear node aggregation
        self.__models_agregated = {}

        # Next Step or Finish
        logging.info(
            f"({self.addr}) Round {self.round} of {self.totalrounds} finished."
        )
        if self.round < self.totalrounds:
            self.__train_step()
        else:
            # At end, all nodes compute metrics
            self.__evaluate()
            # Finish
            self.round = None
            self.totalrounds = None
            self.__model_initialized_lock.acquire()
            logging.info(f"({self.addr}) Training finished!!.")

    #########################
    #    Model Gossiping    #
    #########################

    def get_agregated_models(self, node):
        """
        Get the models that have been aggregated by a given node in the actual round.

        Args:
            node (str): Node to get the aggregated models from.
        """
        try:
            return self.__models_agregated[node]
        except KeyError:
            return []

    def __gossip_model_aggregation(self):
        # Anonymous functions
        candidate_condition = lambda node: (
            (node not in self.aggregator.get_agregated_models())
            and (node in self.__train_set)
        )
        status_function = lambda node: (
            node,
            self.get_agregated_models(node),
        )
        model_function = lambda node: self.aggregator.get_partial_aggregation(
            self.get_agregated_models(node)
        )

        # Gossip
        self.__gossip_model(candidate_condition, status_function, model_function)

    def __gossip_model_difusion(self, initialization=False):
        # Wait a model (init or aggregated)
        if initialization:
            candidate_condition = lambda node: node not in self.__nei_status.keys()
        else:
            logging.info(f"({self.addr}) Gossiping aggregated model.")
            candidate_condition = lambda node: self.__nei_status[node] < self.round

        # Status fn
        status_function = lambda nc: nc
        # Model fn -> At diffusion, contributors are not relevant
        model_function = lambda _: (
            self.learner.get_parameters(),
            self.aggregator.get_agregated_models(),
            1,
        )

        # Gossip
        self.__gossip_model(candidate_condition, status_function, model_function)

    def __gossip_model(
        self,
        candidate_condition,
        status_function,
        model_function,
        period=Settings.GOSSIP_MODELS_PERIOD,
    ):
        # Initialize list with status of nodes in the last X iterations
        last_x_status = []
        j = 0

        while True:
            # Get time to calculate frequency
            t = time.time()

            # If the trainning has been interrupted, stop waiting
            if self.round is None:
                logging.info(f"({self.addr}) Stopping model gossip process.")
                return

            # Get nodes wich need models
            neis = [n for n in self.get_neighbors() if candidate_condition(n)]
            logging.debug(f"({self.addr} Gossip remaining nodes: {neis}")

            # Determine end of gossip
            if neis == []:
                logging.info(f"({self.addr}) Gossip finished.")
                return

            # Save state of neighbors. If nodes are not responding gossip will stop
            if len(last_x_status) != Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS:
                last_x_status.append([status_function(n) for n in neis])
            else:
                last_x_status[j] = str([status_function(n) for n in neis])
                j = (j + 1) % Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS

                # Check if las messages are the same
                for i in range(len(last_x_status) - 1):
                    if last_x_status[i] != last_x_status[i + 1]:
                        break
                    logging.info(
                        f"({self.addr}) Gossiping exited for {Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS} equal reounds."
                    )
                    return

            # Select a random subset of neighbors
            samples = min(Settings.GOSSIP_MODELS_PER_ROUND, len(neis))
            neis = random.sample(neis, samples)

            # Generate and Send Model Partial Aggregations (model, node_contributors)
            for nei in neis:
                model, contributors, weight = model_function(nei)

                # Send Partial Aggregation
                if model is not None:
                    logging.info(f"({self.addr}) Gossiping model to {nei}.")
                    encoded_model = self.learner.encode_parameters(params=model)
                    self._neighbors.send_model(
                        nei, self.round, encoded_model, contributors, weight
                    )

            # Sleep to allow periodicity
            sleep_time = max(0, period - (t - time.time()))
            time.sleep(sleep_time)
