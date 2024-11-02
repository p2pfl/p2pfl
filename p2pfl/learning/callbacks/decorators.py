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

"""Registry for associating callbacks required by each aggregator with corresponding learners."""


from .callback_factory import CallbackFactory


def register(callback_key: str, framework: str):
    """
    Register a callback class with the CallbackFactory.

    Explicitly specifies the framework to which the callback belongs.

    Args:
        callback_key (str): The unique identifier for the callback (e.g., 'scaffold').
        framework (str): The framework name (e.g., 'pytorch', 'tensorflow').

    Returns:
        Callable: The decorator function.

    """
    def decorator(cls):
        try:
            CallbackFactory.register_callback(
                framework=framework.lower(),
                callback_key=callback_key.lower(),
                callback_constructor=lambda aggregator: cls(aggregator)
            )
        except Exception as e:
            raise e
        return cls
    return decorator
