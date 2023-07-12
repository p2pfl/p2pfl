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
from p2pfl.settings import Settings


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

"""
Module to define constants for the p2pfl system.
"""

###################
# Global Settings #
###################


def set_test_settings():
    Settings.GRPC_TIMEOUT = 2
    Settings.HEARTBEAT_PERIOD = 2
    Settings.HEARTBEAT_TIMEOUT = 5
    Settings.GOSSIP_PERIOD = 0
    Settings.TTL = 3
    Settings.GOSSIP_MESSAGES_PER_PERIOD = 100
    Settings.AMOUNT_LAST_MESSAGES_SAVED = 100
    Settings.GOSSIP_MODELS_PERIOD = 1
    Settings.GOSSIP_MODELS_PER_ROUND = 4
    Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 10
    Settings.TRAIN_SET_SIZE = 4
    Settings.VOTE_TIMEOUT = 5
    Settings.AGGREGATION_TIMEOUT = 5
    Settings.WAIT_HEARTBEATS_CONVERGENCE = 0.2 * Settings.HEARTBEAT_TIMEOUT


def wait_convergence(nodes, n_neis, wait=10, only_direct=False):
    acum = 0
    while True:
        begin = time.time()
        if all(
            [len(n.get_neighbors(only_direct=only_direct)) == n_neis for n in nodes]
        ):
            break
        time.sleep(0.1)
        acum += time.time() - begin
        if acum > wait:
            assert False


def full_connection(node, nodes):
    for n in nodes:
        node.connect(n.addr)
