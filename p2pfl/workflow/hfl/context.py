#
# This file is part of the p2pfl (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2026 Pedro Guijas Bravo.
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
"""Typed context and per-peer state for HFL workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from p2pfl.workflow.engine.context import WorkflowContext

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


@dataclass
class HFLPeerState:
    """Per-peer mutable state for HFL workflow."""

    round_number: int = 0
    model: P2PFLModel | None = None
    aggregated_from: list[str] = field(default_factory=list)

    def reset_round(self) -> None:
        """Reset per-round mutable state."""
        self.model = None
        self.aggregated_from.clear()


@dataclass
class HFLContext(WorkflowContext):
    """
    Typed context for Hierarchical Federated Learning.

    Extends WorkflowContext with role-specific state for hierarchical
    topology (workers -> edges -> root).
    """

    # Role: "worker", "edge", or "root"
    role: str = "worker"

    # Topology (configured before learning starts)
    edge_addr: str | None = None  # For workers: assigned edge
    worker_addrs: list[str] = field(default_factory=list)  # For edges: managed workers
    root_addr: str | None = None  # For edges: root node address
    child_edge_addrs: list[str] = field(default_factory=list)  # For root: child edge addresses
    edge_trains: bool = True  # Whether edge nodes train on their own data

    # Per-peer tracking
    peers: dict[str, HFLPeerState] = field(default_factory=dict)
