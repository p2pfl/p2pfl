#
# This file is part of the p2pfl distribution
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

"""Module to define constants for the p2pfl system."""

import os
from dataclasses import dataclass
from typing import Any

from p2pfl.utils.singleton import SingletonMeta

#######################
# Settings by section #
#######################


@dataclass
class General:
    """General system settings."""

    SEED: int | None = None
    """Seed for random number generation."""
    GRPC_TIMEOUT: float = 10.0
    """Maximum time (seconds) to wait for a gRPC request."""
    LOG_LEVEL: str = "INFO"
    """Log level for the system."""
    LOG_DIR: str = "logs"
    """Directory to save logs."""
    MAX_LOG_RUNS: int = 10
    """Maximum number of run log files to keep."""
    DISABLE_RAY: bool = False
    """Disable Ray for local testing."""
    RESOURCE_MONITOR_PERIOD: int = 10
    """Period (seconds) to send resource monitor information."""


@dataclass
class Heartbeat:
    """Heartbeat settings."""

    PERIOD: float = 2.0
    """Period (seconds) to send heartbeats."""
    TIMEOUT: float = 5.0
    """Timeout (seconds) for a node to be considered dead."""
    WAIT_CONVERGENCE: float = PERIOD
    """Time (seconds) to wait for the heartbeats to converge before a learning round starts."""
    EXCLUDE_BEAT_LOGS: bool = True
    """Exclude heartbeat logs."""


@dataclass
class Gossip:
    """Gossip protocol settings."""

    PERIOD: float = 0.1
    """Period (seconds) for the gossip protocol."""
    TTL: int = 10
    """Time to live (TTL) for a message in the gossip protocol."""
    MESSAGES_PER_PERIOD: int = 100
    """Number of messages to send in each gossip period."""
    AMOUNT_LAST_MESSAGES_SAVED: int = 100
    """Number of last messages saved in the gossip protocol (avoid multiple message processing)."""
    MODELS_PERIOD: int = 1
    """Period of gossiping models (times by second)."""
    MODELS_PER_ROUND: int = 2
    """Amount of equal rounds to exit gossiping. Careful, a low value can cause an early stop of gossiping."""
    EXIT_ON_X_EQUAL_ROUNDS: int = 10
    """Amount of equal rounds to exit gossiping. Careful, a low value can cause an early stop of gossiping."""
    MODE_EXPECTATION_TIMEOUT: float = 60.0
    """Timeout (seconds) to wait for a model to be received."""


@dataclass
class SSL:
    """SSL certificate settings."""

    USE_SSL: bool = True
    """Use SSL on experiments."""
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    CA_CRT: str = os.path.join(BASE_DIR, "certificates", "ca.crt")
    """CA certificate."""
    SERVER_CRT: str = os.path.join(BASE_DIR, "certificates", "server.crt")
    """Server certificate."""
    CLIENT_CRT: str = os.path.join(BASE_DIR, "certificates", "client.crt")
    """Client certificate."""
    SERVER_KEY: str = os.path.join(BASE_DIR, "certificates", "server.key")
    """Server private key."""
    CLIENT_KEY: str = os.path.join(BASE_DIR, "certificates", "client.key")
    """Client private key."""


@dataclass
class Training:
    """Training process settings."""

    VOTE_TIMEOUT: int = 60
    """Timeout (seconds) for a node to wait for a vote."""
    AGGREGATION_TIMEOUT: int = 300
    """Timeout (seconds) for a node to wait for other models. Timeout starts when the first model is added."""
    DEFAULT_BATCH_SIZE: int = 128
    """Default batch size for training."""
    RAY_ACTOR_POOL_SIZE: int = 4


###################
# Global Settings #
###################


class Settings(metaclass=SingletonMeta):
    """Class to define global settings for the p2pfl system."""

    general = General()
    """General settings."""
    heartbeat = Heartbeat()
    """Heartbeat settings."""
    gossip = Gossip()
    """Gossip protocol settings."""
    ssl = SSL()
    """SSL certificate settings."""
    training = Training()
    """Training process settings."""

    @classmethod
    def set_from_dict(cls, settings_dict: dict[str, dict[str, Any]]):
        """
        Update settings from a dictionary.

        The dictionary should be nested, with the first level keys
        corresponding to the nested setting classes (e.g., "General", "Heartbeat")
        and the second level keys corresponding to the setting attributes
        within those classes (e.g., "GRPC_TIMEOUT", "HEARTBEAT_PERIOD").

        Args:
            settings_dict: Dictionary with the settings to update.

        """
        for category, category_settings in settings_dict.items():
            category_lower = category.lower()
            if hasattr(cls, category_lower):
                nested_class = getattr(cls, category_lower)
                for setting_name, setting_value in category_settings.items():
                    if hasattr(nested_class, setting_name.upper()):  # Assuming settings are uppercase in dataclasses
                        setattr(nested_class, setting_name.upper(), setting_value)
                        # print(f"    {setting_name}: {setting_value} ✅")
                    else:
                        print(f"❌ {category_lower}.{setting_name}: {setting_value} not found in settings")
            else:
                print(f"❌ {category} not found in settings")
