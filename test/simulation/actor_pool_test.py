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
"""Test the Ray Actor pool."""

import contextlib
from unittest.mock import MagicMock, patch

import pytest

with contextlib.suppress(ImportError):
    from p2pfl.learning.frameworks.simulation.actor_pool import SuperActorPool


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset the singleton instance of the SuperActorPool class."""
    SuperActorPool._instance = None


def test_super_actor_pool_singleton():
    """Test the singleton behavior of the SuperActorPool class."""
    pool1 = SuperActorPool()
    pool2 = SuperActorPool()
    assert pool1 is pool2


def test_flag_future_as_ready():
    """
    Test the _flag_future_as_ready method of the SuperActorPool class.

    1. Create a SuperActorPool instance.
    2. Reset the addr_to_future dictionary with a mock address.
    3. Call the _flag_future_as_ready method with the mock address.
    4. Check that the future is flagged as ready.

    """
    pool = SuperActorPool()
    pool._reset_addr_to_future_dict("addr1")
    pool._flag_future_as_ready("addr1")
    assert pool._addr_to_future["addr1"]["ready"] is True


def test_fetch_future_result():
    """
    Test the _fetch_future_result method of the SuperActorPool class.

    1. Create a SuperActorPool instance.
    2. Reset the addr_to_future dictionary with a mock address.
    3. Create a mock future and set it in the addr_to_future dictionary.
    4. Mock the ray.get function to return the address and the result.
    5. Call the _fetch_future_result method with the mock address.
    6. Check that the result is the expected one.

    """
    pool = SuperActorPool()
    pool._reset_addr_to_future_dict("addr1")
    mock_future = MagicMock()
    pool._addr_to_future["addr1"]["future"] = mock_future
    with patch("ray.get", return_value=("addr1", "result")):
        result = pool._fetch_future_result("addr1")

    assert result == ("addr1", "result")


def test_flag_actor_for_removal():
    """
    Test the _flag_actor_for_removal method of the SuperActorPool class.

    1. Create a SuperActorPool instance.
    2. Create a mock actor id.
    3. Call the _flag_actor_for_removal method with the mock actor id.
    4. Check that the actor id is in the actor_to_remove dictionary.

    """
    pool = SuperActorPool()
    mock_actor_id = "actor_id_hex"
    pool._flag_actor_for_removal(mock_actor_id)
    assert mock_actor_id in pool.actor_to_remove


def test_check_and_remove_actor_from_pool():
    """
    Test the _check_and_remove_actor_from_pool method of the SuperActorPool class.

    1. Create a SuperActorPool instance.
    2. Create a mock actor.
    3. Call the _check_and_remove_actor_from_pool method with the mock actor.
    4. Check that the actor was removed from the pool.
    5. Check that the result is True.

    """
    pool = SuperActorPool()
    mock_actor = MagicMock()
    mock_actor._actor_id.hex.return_value = "actor_id_hex"
    pool._flag_actor_for_removal("actor_id_hex")
    result = pool._check_and_remove_actor_from_pool(mock_actor)
    assert result is False


def test_submit_learner_job_with_idle_actor():
    """
    Test the submit_learner_job method of the SuperActorPool class when there are idle actors.

    1. Create a SuperActorPool instance.
    2. Create a mock function and a mock learner.
    3. Create a mock address.
    4. Call the submit_learner_job method with the mock function and the mock address and learner.
    5. Check that the submit method was called with the correct arguments.

    """
    pool = SuperActorPool()
    pool._idle_actors = [MagicMock()]
    mock_fn = MagicMock()
    mock_learner = MagicMock()
    mock_addr = "addr1"

    with patch.object(pool, "submit") as mock_submit:
        pool.submit_learner_job(mock_fn, (mock_addr, mock_learner))

    mock_submit.assert_called_once_with(mock_fn, (mock_addr, mock_learner))


def test_submit_learner_job_with_no_idle_actor():
    """
    Test the submit_learner_job method of the SuperActorPool class when there are no idle actors.

    1. Create a SuperActorPool instance.
    2. Create a mock function and a mock learner.
    3. Create a mock address.
    4. Call the submit_learner_job method with the mock function and the mock address and learner.
    5. Check that the job was added to the pending submits.

    """
    pool = SuperActorPool()
    pool._idle_actors = []
    mock_fn = MagicMock()
    mock_learner = MagicMock()
    mock_addr = "addr1"

    pool.submit_learner_job(mock_fn, (mock_addr, mock_learner))

    assert pool._pending_submits == [(mock_fn, (mock_addr, mock_learner))]


def test_process_unordered_future():
    """
    Test the process_unordered_future method of the SuperActorPool class.

    1. Create a SuperActorPool instance.
    2. Create a mock future and set it in the future_to_actor dictionary.
    3. Mock the ray.wait function to return the mock future.
    4. Call the process_unordered_future method.
    5. Check that the future was processed correctly.

    """
    pool = SuperActorPool()
    mock_future = MagicMock()
    mock_actor = MagicMock()
    mock_addr = "addr1"
    pool._future_to_actor[mock_future] = (0, mock_actor, mock_addr)
    pool._reset_addr_to_future_dict(mock_addr)

    with (
        patch("ray.wait", return_value=([mock_future], [])),
        patch.object(pool, "_check_and_remove_actor_from_pool", return_value=True),
        patch.object(pool, "_return_actor") as mock_return_actor,
    ):
        pool.process_unordered_future()

    mock_return_actor.assert_called_once_with(mock_actor)
    assert pool._addr_to_future[mock_addr]["ready"] is True


def test_get_learner_result():
    """
    Test the get_learner_result method of the SuperActorPool class.

    1. Create a SuperActorPool instance.
    2. Create a mock address and set it in the addr_to_future dictionary.
    3. Mock the process_unordered_future method to set the future as ready.
    4. Mock the fetch_future_result method to return the result.
    5. Call the get_learner_result method with the mock address.
    6. Check that the result is the expected one.

    """
    pool = SuperActorPool()
    pool._reset_addr_to_future_dict("addr1")
    mock_future = MagicMock()
    pool._addr_to_future["addr1"]["future"] = mock_future
    with patch("ray.get", return_value=("addr1", "result")):
        result = pool.get_learner_result("addr1", timeout=None)

    assert result == ("addr1", "result")
