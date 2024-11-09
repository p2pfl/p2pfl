#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/p2pfl).
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

"""A factory for returning the correct callback for the given aggregator."""

from typing import Any, Callable, Dict, List, Type

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.learner import NodeLearner


class CallbackFactory:
    """Factory for creating callbacks based on learner framework and aggregator requirements."""

    _callback_registry: Dict[tuple, List[Callable[[], Any]]] = {}

    @classmethod
    def register_callback(cls, framework: str, callback_key: str, callback_constructor: Callable[[], Any]):
        """
        Register a callback constructor for a given learner framework and callback key.

        Args:
            framework (str): The framework of the learner (e.g., 'pytorch', 'tensorflow').
            callback_key (str): The unique identifier for the callback (e.g., 'scaffold').
            callback_constructor (Callable[[Any], Callback]): A callable that returns an instance of a Callback.

        """
        key = (framework, callback_key.lower())
        if key not in cls._callback_registry:
            cls._callback_registry[key] = []
        cls._callback_registry[key].append(callback_constructor)

    @classmethod
    def create_callbacks(cls, learner: Type[NodeLearner], aggregator: Aggregator) -> List[Any]:
        """
        Create the callbacks required by the aggregator for the given learner.

        Args:
            learner (Any): The learner instance (must implement `get_framework`).
            aggregator (Any): The aggregator instance.

        """
        try:
            framework = learner.get_framework()
            if not framework:
                raise ValueError(f"Learner '{learner}' did not specify a framework.")

            required_callback_keys = aggregator.get_required_callbacks()
            if not required_callback_keys:
                return []

            callbacks = []
            for callback_key in required_callback_keys:
                key = (framework, callback_key.lower())
                constructors = cls._callback_registry.get(key, [])
                if not constructors:
                    continue
                for constructor in constructors:
                    callback_instance = constructor()
                    callbacks.append(callback_instance)
            return callbacks
        except Exception as e:
            raise e
