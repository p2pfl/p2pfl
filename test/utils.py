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
import torch
from p2pfl.settings import Settings


def set_test_settings():
    Settings.BLOCK_SIZE = 8192
    Settings.NODE_TIMEOUT = 10
    Settings.VOTE_TIMEOUT = 10
    Settings.AGGREGATION_TIMEOUT = 10
    Settings.HEARTBEAT_PERIOD = 3
    Settings.HEARTBEATER_REFRESH_NEIGHBORS_BY_PERIOD = 2
    Settings.WAIT_HEARTBEATS_CONVERGENCE = 4
    Settings.TRAIN_SET_SIZE = 5
    Settings.TRAIN_SET_CONNECT_TIMEOUT = 5
    Settings.AMOUNT_LAST_MESSAGES_SAVED = 100
    Settings.GOSSIP_MESSAGES_FREC = 100
    Settings.GOSSIP_MESSAGES_PER_ROUND = 100
    Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 9
    Settings.GOSSIP_MODELS_FREC = 1
    Settings.GOSSIP_MODELS_PER_ROUND = 2


def wait_network_nodes(nodes):
    acum = 0
    while True:
        begin = time.time()
        if all([len(n.get_network_nodes()) == len(nodes) for n in nodes]):
            break
        time.sleep(0.1)
        acum += time.time() - begin
        if acum > 6:
            assert False


def wait_4_results(nodes):
    while True:
        time.sleep(1)
        finish = True
        for f in [node.round is None for node in nodes]:
            finish = finish and f

        if finish:
            break


def check_equal_models(nodes):
    model = None
    first = True
    for node in nodes:
        if first:
            model = node.learner.get_parameters()
            first = False
        else:
            for layer in model:
                a = torch.round(model[layer], decimals=2)
                b = torch.round(node.learner.get_parameters()[layer], decimals=2)
                assert torch.eq(a, b).all()
