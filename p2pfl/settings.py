#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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

"""Module to define constants for the p2pfl system."""

###################
# Global Settings #
###################


class Settings:
    """Class to define global settings for the p2pfl system."""

    ######
    # GENERAL
    ######
    GRPC_TIMEOUT: float = 10
    """
    Maximum time (seconds) to wait for a gRPC request.
    """
    LOG_LEVEL: str = "INFO"
    """
    Log level for the system.
    """
    LOG_DIR: str = "logs"
    """
    Directory to save logs.
    """

    ######
    # HEARTBEAT
    ######
    HEARTBEAT_PERIOD: float = 2
    """
    Period (seconds) to send heartbeats.
    """
    HEARTBEAT_TIMEOUT: float = 5
    """
    Timeout (seconds) for a node to be considered dead.
    """

    ######
    # GOSSIP
    ######
    GOSSIP_PERIOD: float = 0.1
    """
    Period (seconds) for the gossip protocol.
    """
    TTL: int = 10
    """
    Time to live (TTL) for a message in the gossip protocol.
    """
    GOSSIP_MESSAGES_PER_PERIOD: int = 100
    """
    Number of messages to send in each gossip period.
    """
    AMOUNT_LAST_MESSAGES_SAVED = 100
    """
    Number of last messages saved in the gossip protocol (avoid multiple message processing).
    """
    GOSSIP_MODELS_PERIOD: int = 1
    """
    Period of gossiping models (times by second).
    """
    GOSSIP_MODELS_PER_ROUND: int = 2
    """
    Amount of equal rounds to exit gossiping. Careful, a low value can cause an early stop of gossiping.
    """
    GOSSIP_EXIT_ON_X_EQUAL_ROUNDS: int = 10
    """
    Amount of equal rounds to exit gossiping. Careful, a low value can cause an early stop of gossiping.
    """

    ######
    # TRAINING
    ######
    TRAIN_SET_SIZE: int = 4
    """
    Size of the training set.
    """
    VOTE_TIMEOUT: int = 60
    """
    Timeout (seconds) for a node to wait for a vote.
    """
    AGGREGATION_TIMEOUT: int = 300
    """
    Timeout (seconds) for a node to wait for other models. Timeout starts when the first model is added.
    """
    WAIT_HEARTBEATS_CONVERGENCE: float = 0.2 * HEARTBEAT_TIMEOUT
    """
    Time (seconds) to wait for the heartbeats to converge before a learning round starts.
    """

    ######
    # WEB
    ######
    RESOURCE_MONITOR_PERIOD: int = 1
    """
    Period (seconds) to send resource monitor information.
    """
