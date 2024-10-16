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

"""
SuperActorPool extends ActorPool to manage a pool of VirtualLearnerActor instances
for asynchronous distributed computing using Ray.

Attributes:
    _instance (SuperActorPool): Singleton instance of SuperActorPool.
    _lock (threading.Lock): Lock for thread-safe instance creation.
    resources (dict): Resources for actor creation.
    _addr_to_future (dict): Mapping from actor address to future information.
    actor_to_remove (set): Set of actor IDs scheduled for removal.
    num_actors (int): Number of active actors in the pool.
    lock (threading.RLock): Reentrant lock for thread-safe operations.
    initialized (bool): Flag indicating initialization status.

Methods:
    __init__(self, resources=None, actor_list=None):
        Initializes the SuperActorPool with optional initial actors.

    create_actor(self):
        Creates a new VirtualLearnerActor instance using provided resources.

    add_actor(self, num_actors):
        Adds a specified number of actors to the pool.

    submit(self, fn, value):
        Submits a task to an idle actor in the pool.

    submit_learner_job(self, actor_fn, job):
        Submits a learner job to the pool, handling pending submits if no idle actors are available.

    _flag_future_as_ready(self, addr):
        Flags the future associated with the given address as ready.

    _reset_addr_to_future_dict(self, addr):
        Resets the future dictionary for a given address.

    _is_future_ready(self, addr):
        Checks if the future associated with the given address is ready.

    _fetch_future_result(self, addr):
        Fetches the result of a future associated with the given address.

    _flag_actor_for_removal(self, actor_id_hex):
        Flags a specified actor for removal from the pool.

    _check_and_remove_actor_from_pool(self, actor):
        Checks if an actor should be removed from the pool based on flagged removals.

    _check_actor_fits_in_pool(self):
        Checks if the current number of actors in the pool is within resource limits.

    process_unordered_future(self, timeout=None):
        Processes the next unordered future result from the pool.

    get_learner_result(self, addr, timeout=None):
        Retrieves the learner result associated with the given address.
"""

import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from p2pfl.learning.learner import NodeLearner
from p2pfl.simulation.actor import VirtualLearnerActor
from p2pfl.simulation.utils import pool_size_from_resources, check_client_resources

import ray
from ray import ObjectRef
from ray.util.actor_pool import ActorPool

class SuperActorPool(ActorPool):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SuperActorPool, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, resources = None, actor_list: Optional[List[Type[VirtualLearnerActor]]] = None):
        """
        Initialize SuperActorPool.

        Args:
            resources (dict, optional): Resources for actor creation. Defaults to None.
            actor_list (List[Type[VirtualLearnerActor]], optional): List of pre-created actor instances. Defaults to None.
        """
        if not hasattr(self, 'initialized'):  # To avoid reinitialization
            self.resources = check_client_resources(resources)

            if actor_list is None:
                num_actors = pool_size_from_resources(self.resources)
                actors = [self.create_actor() for _ in range(num_actors)]
            else:
                actors = actor_list

            super().__init__(actors)

            # A dict that maps addr to another dict containing: a reference to the remote job
            # and its status (i.e. whether it is ready or not)
            self._addr_to_future: Dict[
                str, Dict[str, Union[bool, Optional[ObjectRef[Any]]]]
            ] = {}
            self.actor_to_remove: Set[str] = set()  # a set of actor ids to be removed
            self.num_actors = len(actors)

            self.lock = threading.RLock()

            self.initialized = True  # Mark as initialized

    def __reduce__(self):
        return SuperActorPool, (
            self.resources,
            self._idle_actors,
        )

    def create_actor(self):
        """
        Creates a new VirtualLearnerActor instance using provided resources.

        Returns:
            VirtualLearnerActor ray handler: New actor instance.
        """
        return VirtualLearnerActor.options(**self.resources).remote()

    def add_actor(self, num_actors: int) -> None:
        """
        Adds a specified number of actors to the pool.

        Args:
            num_actors (int): Number of actors to add.
        """
        with self.lock:
            new_actors = [self.create_actor() for _ in range(num_actors)]
            self._idle_actors.extend(new_actors)
            self.num_actors += num_actors
            print(f"Created {num_actors} actors")

    def submit(self, fn: Any, value: Tuple[str,NodeLearner]) -> None:
        """
        Submits a task to an idle actor in the pool.

        Args:
            fn (Any): Function to be executed by the actor.
            value (Tuple[str, NodeLearner]): Tuple containing address and learner information.
        """
        addr, learner = value
        actor = self._idle_actors.pop()

        if self._check_and_remove_actor_from_pool(actor):
            future = fn(actor, learner, addr)
            future_key = tuple(future) if isinstance(future, list) else future
            self._future_to_actor[future_key] = (self._next_task_index, actor, addr)
            self._next_task_index += 1
            self._addr_to_future[addr]["future"] = future_key

    def submit_learner_job(self, actor_fn: Any, job: Tuple[str,NodeLearner]) -> None:
        """
        Submits a learner job to the pool, handling pending submits if no idle actors are available.

        Args:
            actor_fn (Any): Function to be executed by the actor.
            job (Tuple[str, NodeLearner]): Tuple containing address and learner information.
        """
        addr, _ = job 
        with self.lock:
            self._reset_addr_to_future_dict(addr)
            if self._idle_actors:
                self.submit(actor_fn, job)
            else:
                self._pending_submits.append((actor_fn, job))

    def _flag_future_as_ready(self, addr: str) -> None:
        """
        Flags the future associated with the given address as ready.

        Args:
            addr (str): Address of the future to flag.
        """
        self._addr_to_future[addr]["ready"] = True

    def _reset_addr_to_future_dict(self, addr: str) -> None:
        """
        Resets the future dictionary for a given address.

        Args:
            addr (str): Address to reset in the future dictionary.
        """
        if addr not in self._addr_to_future:
            self._addr_to_future[addr] = {}
        self._addr_to_future[addr]["future"] = None
        self._addr_to_future[addr]["ready"] = False

    def _is_future_ready(self, addr: str) -> bool:
        """
        Checks if the future associated with the given address is ready.

        Args:
            addr (str): Address of the future to check.

        Returns:
            bool: True if future is ready, False otherwise.
        """
        if addr not in self._addr_to_future:
            print("This shouldn't be happening")
            return False
        return self._addr_to_future[addr]["ready"]

    def _fetch_future_result(self, addr: str) -> Tuple[Any, Any]:
        """
        Fetches the result of a future associated with the given address.

        Args:
            addr (str): Address of the future to fetch.

        Returns:
            Tuple[Any, Any]: Address and result fetched from the future.
        """
        try:
            future = self._addr_to_future[addr]["future"]
            res_addr, result = ray.get(future)
        except ray.exceptions.RayActorError as ex:
            print(ex)
            if hasattr(ex, "actor_id"):
                self._flag_actor_for_removal(ex.actor_id)
            raise ex
        assert res_addr == addr
        self._reset_addr_to_future_dict(addr)
        return result

    def _flag_actor_for_removal(self, actor_id_hex: str) -> None:
        """
        Flags a specified actor for removal from the pool.

        Args:
            actor_id_hex (str): ID of the actor to be removed.
        """
        with self.lock:
            self.actor_to_remove.add(actor_id_hex)
            print(f"Actor({actor_id_hex}) will be removed from pool.")

    def _check_and_remove_actor_from_pool(self, actor: VirtualLearnerActor) -> bool:
        """
        Checks if an actor should be removed from the pool based on flagged removals.

        Args:
            actor (VirtualLearnerActor): Actor instance to check.

        Returns:
            bool: True if the actor should not be removed, False otherwise.
        """
        with self.lock:
            actor_id = actor._actor_id.hex()
            if actor_id in self.actor_to_remove:
                self.actor_to_remove.remove(actor_id)
                self.num_actors -= 1
                print(f"REMOVED actor {actor_id} from pool")
                return False
            return True

    def _check_actor_fits_in_pool(self) -> bool:
        """
        Checks if the current number of actors in the pool is within resource limits.

        Returns:
            bool: True if the number of actors is within limits, False otherwise.
        """
        num_actors_updated = pool_size_from_resources(self.resources)
        if num_actors_updated < self.num_actors:
            self.num_actors -= 1
            return False
        return True

    def process_unordered_future(self, timeout: Optional[float] = None) -> None:
        """
        Processes the next unordered future result from the pool.

        Args:
            timeout (float, optional): Timeout for processing the future. Defaults to None.

        Raises:
            StopIteration: If no more results are available.
            TimeoutError: If the future processing times out.
        """
        if not self.has_next():
            raise StopIteration("No more results to get")
        res, _ = ray.wait(list(self._future_to_actor), num_returns=1, timeout=timeout)
        if res:
            [future] = res
        else:
            raise TimeoutError("Timed out waiting for result")
        with self.lock:
            _, actor, addr = self._future_to_actor.pop(future, (None, None, -1))
            if actor is not None:
                if self._check_actor_fits_in_pool():
                    if self._check_and_remove_actor_from_pool(actor):
                        self._return_actor(actor)
                    self._flag_future_as_ready(addr)
                else:
                    actor.terminate.remote()

    def get_learner_result(self, addr: str, timeout: Optional[float]) -> Tuple[Any, Any]:
        """
        Retrieves the learner result associated with the given address.

        Args:
            addr (str): Address of the learner result to retrieve.
            timeout (float, optional): Timeout for retrieving the result. Defaults to None.

        Returns:
            Tuple[Any, Any]: Address and result of the learner job.
        """
        while self.has_next() and not self._is_future_ready(addr):
            try:
                self.process_unordered_future(timeout=timeout)
            except StopIteration:
                break
        return self._fetch_future_result(addr)