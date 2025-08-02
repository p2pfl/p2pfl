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

"""P2PFLCallback factory."""

from p2pfl.learning.aggregators.aggregator import Aggregator
from p2pfl.learning.frameworks import Framework
from p2pfl.learning.frameworks.callback import P2PFLCallback

###
#   FACTORY
###


class CallbackFactory:
    """Factory for creating callbacks based on learner framework and aggregator requirements."""

    _callback_registry: dict[str, list[type[P2PFLCallback]]] = {}

    @classmethod
    def register_callback(cls, learner: str, callback: type[P2PFLCallback]):
        """
        Register a callback constructor for a given learner framework and callback key.

        Args:
            learner: The learner instance (e.g., 'KerasLearner', 'LightningLearner').
            callback: A callback.

        """
        # Register the learner
        if learner not in cls._callback_registry:
            cls._callback_registry[learner] = []

        # Register the callback
        if callback in cls._callback_registry[learner]:
            raise ValueError(f"Callback {callback} already registered for {learner}.")
        cls._callback_registry[learner].append(callback)

    @classmethod
    def create_callbacks(cls, framework: str, aggregator: Aggregator) -> list[P2PFLCallback]:
        """
        Create the callbacks required by the aggregator for the given learner.

        Args:
            framework: The framework of the learner.
            aggregator (Any): The aggregator instance.

        """
        required_callbacks = aggregator.get_required_callbacks()
        if not required_callbacks:
            return []

        # Get the callbacks for the learner
        if framework not in cls._callback_registry:
            raise ValueError(f"No callbacks registered for {framework}.")
        learner_callbacks = cls._callback_registry[framework]
        learner_callbacks_dict = {callback.get_name(): callback for callback in learner_callbacks}

        # Check if all required callbacks are registered
        for rc in required_callbacks:
            if rc not in learner_callbacks_dict:
                raise ValueError(f"Callback {rc} not registered for {framework}.")

        # Get and build the required callback keys
        return [learner_callbacks_dict[rc]() for rc in required_callbacks]


###
#   REGISTER CALLBACKS
###

try:
    from p2pfl.learning.frameworks.pytorch.callbacks.scaffold_callback import SCAFFOLDCallback as SCAFFOLDCallbackPT

    CallbackFactory.register_callback(learner=Framework.PYTORCH.value, callback=SCAFFOLDCallbackPT)
except ImportError:
    pass

try:
    from p2pfl.learning.frameworks.pytorch.callbacks.fedprox_callback import FedProxCallback as FedProxCallbackPT

    CallbackFactory.register_callback(learner=Framework.PYTORCH.value, callback=FedProxCallbackPT)
except ImportError:
    pass

try:
    from p2pfl.learning.frameworks.tensorflow.callbacks.scaffold_callback import SCAFFOLDCallback as SCAFFOLDCallbackTF

    CallbackFactory.register_callback(learner=Framework.TENSORFLOW.value, callback=SCAFFOLDCallbackTF)
except ImportError:
    pass
