#
# This file is part of the p2pfl distribution (see https://github.com/pguijas/p2pfl).
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
"""Workflow factory with registry for built-in and custom workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from p2pfl.workflow.engine.workflow import Workflow

_workflow_registry: dict[str, type[Workflow[Any]]] = {}


def _load_builtins() -> None:
    """Load built-in workflows into registry (lazy imports avoid circular deps)."""
    from p2pfl.workflow.async_dfl.workflow import AsyncDFL
    from p2pfl.workflow.basic_dfl.workflow import BasicDFL
    from p2pfl.workflow.hfl.workflow import HFL

    _workflow_registry.setdefault("basic", BasicDFL)
    _workflow_registry.setdefault("async", AsyncDFL)
    _workflow_registry.setdefault("hfl", HFL)


_load_builtins()


def register_workflow(name: str, workflow_class: type[Workflow[Any]]) -> None:
    """
    Register a workflow type by name.

    Registered workflows can be created via :func:`create_workflow` and used
    across the network (the name is serialized in gossip messages).

    Args:
        name: Unique string key for this workflow (e.g. ``"fedprox"``).
        workflow_class: The workflow class to instantiate.

    Raises:
        ValueError: If a workflow with the same name is already registered.

    """
    if name in _workflow_registry:
        raise ValueError(f"Workflow '{name}' is already registered")
    _workflow_registry[name] = workflow_class


def create_workflow(workflow_name: str) -> Workflow[Any]:
    """
    Create a workflow instance by registered name.

    Args:
        workflow_name: The registered name of the workflow (e.g. ``"basic"``, ``"async"``).

    Returns:
        A new workflow instance.

    Raises:
        ValueError: If the workflow name is not registered.

    """
    if workflow_name not in _workflow_registry:
        valid = ", ".join(sorted(_workflow_registry.keys()))
        raise ValueError(f"Unknown workflow '{workflow_name}'. Registered: {valid}")
    return _workflow_registry[workflow_name]()


def list_workflows() -> list[str]:
    """
    List all registered workflow names.

    Returns:
        Sorted list of registered workflow names.

    """
    return sorted(_workflow_registry.keys())
