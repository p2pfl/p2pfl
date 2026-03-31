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

"""Manage Ray placement groups."""

import threading
import warnings

import ray
from ray.util.placement_group import PlacementGroup, placement_group, remove_placement_group


class PlacementGroupManager:
    """Singleton manager for a shared Ray placement group using all available resources by default."""

    _instance: "PlacementGroupManager | None" = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls, bundles: list[dict[str, float]] | None = None, num_actors: int | None = None) -> "PlacementGroupManager":
        """
        Create or return the singleton instance.

        Args:
            bundles: Explicit resource bundles for the placement group.
            num_actors: Number of actors — divides resources evenly into per-actor bundles.

        Returns:
            The singleton PlacementGroupManager instance.

        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, bundles: list[dict[str, float]] | None = None, num_actors: int | None = None) -> None:
        """
        Initialize the placement group manager.

        Args:
            bundles: Explicit resource bundles for the placement group.
            num_actors: Number of actors — divides resources evenly into per-actor bundles.

        """
        if self._initialized:
            if bundles is not None or num_actors is not None:
                warnings.warn(
                    "PlacementGroupManager is a singleton — ignoring bundles/num_actors "
                    "because it was already initialized. Call shutdown() first to reconfigure.",
                    stacklevel=2,
                )
            return

        available = ray.cluster_resources()
        filtered = {k: v for k, v in available.items() if not k.startswith("node:") and not k.startswith("object_store")}
        cpu = filtered.pop("CPU", 0)
        gpu = filtered.pop("GPU", 0)

        if bundles is not None:
            # Explicit bundles provided — use as-is
            self.bundles = bundles
        elif num_actors is not None and num_actors > 0:
            # Divide resources evenly across actors
            cpu_per = max(cpu / num_actors, 0.5)
            bundle: dict[str, float] = {"CPU": cpu_per}
            if gpu > 0:
                bundle["GPU"] = gpu / num_actors
            self.bundles = [dict(bundle) for _ in range(num_actors)]
        else:
            # Default: single bundle with all resources
            self.bundles = []
            if gpu > 0:
                self.bundles.append({"CPU": cpu, "GPU": gpu})
            elif cpu > 0:
                self.bundles.append({"CPU": cpu})
            for res, val in filtered.items():
                self.bundles.append({res: val})

        # Strategy: SPREAD for multi-node, PACK for single-node
        alive_nodes = [n for n in ray.nodes() if n.get("Alive")]
        strategy = "SPREAD" if len(alive_nodes) > 1 else "PACK"

        self.pg = placement_group(self.bundles, strategy=strategy)
        ray.get(self.pg.ready())
        self._initialized = True

    def get_placement_group(self) -> PlacementGroup:
        """Get the Ray placement group."""
        return self.pg

    def get_bundle_count(self) -> int:
        """Get the number of bundles in the placement group."""
        return len(self.bundles)

    def shutdown(self) -> None:
        """Remove the placement group and reset the singleton instance."""
        with self.__class__._lock:
            remove_placement_group(self.pg)
            self.__class__._instance = None
            self._initialized = False
