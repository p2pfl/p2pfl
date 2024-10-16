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

import pytest
from unittest.mock import MagicMock, patch
from p2pfl.simulation.actor_pool import SuperActorPool

def test_super_actor_pool_initialization():
    pool = SuperActorPool(resources={"num_cpus": 2})
    assert pool.num_actors > 0

def test_super_actor_pool_singleton():
    pool1 = SuperActorPool(resources={"num_cpus": 2})
    pool2 = SuperActorPool(resources={"num_cpus": 2})
    assert pool1 is pool2

def test_add_actor():
    pool = SuperActorPool(resources={"num_cpus": 2})
    initial_num_actors = pool.num_actors
    pool.add_actor(2)
    assert pool.num_actors == initial_num_actors + 2

def test_submit_task():
    # TODO
    pass

def test_flag_future_as_ready():
    pool = SuperActorPool(resources={"num_cpus": 2})
    pool._reset_addr_to_future_dict("addr1")
    pool._flag_future_as_ready("addr1")
    assert pool._addr_to_future["addr1"]["ready"] is True

def test_fetch_future_result():
    pool = SuperActorPool(resources={"num_cpus": 2})
    pool._reset_addr_to_future_dict("addr1")
    mock_future = MagicMock()
    pool._addr_to_future["addr1"]["future"] = mock_future
    with patch('ray.get', return_value=("addr1", "result")):
        result = pool._fetch_future_result("addr1")
    assert result == "result"

def test_flag_actor_for_removal():
    pool = SuperActorPool(resources={"num_cpus": 2})
    mock_actor_id = "actor_id_hex"
    pool._flag_actor_for_removal(mock_actor_id)
    assert mock_actor_id in pool.actor_to_remove

def test_check_and_remove_actor_from_pool():
    pool = SuperActorPool(resources={"num_cpus": 2})
    mock_actor = MagicMock()
    mock_actor._actor_id.hex.return_value = "actor_id_hex"
    pool._flag_actor_for_removal("actor_id_hex")
    result = pool._check_and_remove_actor_from_pool(mock_actor)
    assert result is False

def test_check_actor_fits_in_pool():
    pool = SuperActorPool(resources={"num_cpus": 2})
    pool.num_actors = 10
    with patch('p2pfl.simulation.utils.pool_size_from_resources', return_value=5):
        result = pool._check_actor_fits_in_pool()
    assert result is False