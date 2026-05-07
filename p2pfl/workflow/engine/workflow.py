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
"""Workflow base class."""

from __future__ import annotations

import asyncio
import contextlib
import enum
import inspect
import random
import re
import time
from abc import abstractmethod
from collections.abc import Callable
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any, Generic

from p2pfl.management.logger import logger
from p2pfl.management.logger.experiment_observer import ExperimentLoggerObserver
from p2pfl.workflow.engine.context import TContext
from p2pfl.workflow.engine.experiment import Experiment
from p2pfl.workflow.engine.message import MessageEntry
from p2pfl.workflow.engine.stage import Stage
from p2pfl.workflow.validation import validate_workflow

if TYPE_CHECKING:
    from p2pfl.communication.protocols.communication_protocol import CommunicationProtocol
    from p2pfl.learning.aggregators.aggregator import Aggregator
    from p2pfl.learning.frameworks.learner import Learner


class WorkflowStatus(enum.Enum):
    """Status of a workflow run."""

    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Check if the workflow reached any conclusive state."""
        return self in (WorkflowStatus.FINISHED, WorkflowStatus.CANCELLED, WorkflowStatus.FAILED)


class Workflow(Generic[TContext]):
    """
    Base class for learning workflows.

    Subclasses must implement:
    - ``get_stages()``: returns ``list[Stage[TContext]]``
    - ``create_context()``: builds the typed context from run parameters

    ``initial_stage`` is derived from the first element of ``get_stages()``.
    Override as a class attribute to use a different entry point.

    Stage names are derived automatically from each stage class (see
    ``Stage.__init_subclass__``).  Override ``Stage.name`` as a class
    attribute to customize.

    Example::

        class BasicDFL(Workflow[BasicDFLContext]):
            context_class = BasicDFLContext

            def get_stages(self) -> list[Stage[BasicDFLContext]]:
                return [SetupStage(), VotingStage(), TrainStage(), FinishStage()]
    """

    context_class: type[TContext]

    @property
    def initial_stage(self) -> str:
        """Return the name of the first stage from ``get_stages()``."""
        stages = self.get_stages()
        if not stages:
            raise ValueError("get_stages() returned an empty list")
        return stages[0].name

    def __init__(self) -> None:
        """Initialize the workflow."""
        self.status: WorkflowStatus = WorkflowStatus.IDLE
        self.error: Exception | None = None
        self._task: asyncio.Task[Experiment] | None = None
        self._stage_map: dict[str, Stage[TContext]] = {}
        self._current_stage: Stage[TContext] | None = None
        self._handlers: dict[str, list[tuple[Callable[..., Any], MessageEntry]]] = {}
        self.stage_timings: list[dict[str, Any]] = []

    ############################
    #    Abstract interface    #
    ############################

    @abstractmethod
    def get_stages(self) -> list[Stage[TContext]]:
        """Return the list of stages for this workflow."""
        ...

    def create_context(
        self,
        address: str,
        learner: Learner,
        aggregator: Aggregator,
        cp: CommunicationProtocol,
        generator: random.Random,
        experiment: Experiment,
    ) -> TContext:
        """
        Build the typed context from run parameters.

        Uses ``context_class`` to construct the context with the base fields.
        Override for custom initialization.
        """
        return self.context_class(
            address=address,
            learner=learner,
            aggregator=aggregator,
            cp=cp,
            generator=generator,
            experiment=experiment,
        )

    #####################
    #    Composition    #
    #####################

    def _compose(self, ctx: TContext) -> None:
        """Wire stages, build handler map, and validate the graph."""
        self._stage_map = {s.name: s for s in self.get_stages()}
        self._handlers.clear()

        self._validate_graph()

        self._bind_context(ctx)
        self._collect_handlers()
        self._validate_during_names()

    def _bind_context(self, ctx: TContext) -> None:
        """Set the workflow context on each stage."""
        for stage in self._stage_map.values():
            stage.ctx = ctx

    def _validate_graph(self) -> None:
        """Validate the workflow stage graph via AST inspection."""
        result = validate_workflow(self._stage_map, self.initial_stage)
        if not result.is_valid:
            errors_str = "\n".join(f"  - {e}" for e in result.errors)
            raise ValueError(f"Invalid workflow graph:\n{errors_str}")

    def _collect_handlers(self) -> None:
        """Collect @on_message handlers from stages, storing bound callables."""
        for stage in self._stage_map.values():
            for cls in type(stage).__mro__:
                if cls is Stage or cls is object:
                    break
                for msg_name, entry in cls.__dict__.get("_message_registry", {}).items():
                    # Default stage handlers to their own stage if `during` not specified
                    if entry.during is None:
                        entry = MessageEntry(entry.method_name, entry.is_weights, frozenset({stage.name}))
                    bound = getattr(stage, entry.method_name)
                    if entry.is_weights and "weights" not in inspect.signature(bound).parameters:
                        raise ValueError(
                            f"Handler '{msg_name}' on {type(stage).__name__} is declared with weights=True "
                            f"but its signature lacks a 'weights' parameter."
                        )
                    self._register_handler(bound, msg_name, entry)

    def _expand_during(self, during: frozenset[str] | None) -> set[str]:
        """Expand a ``during`` set by resolving regex patterns against registered stage names."""
        if during is None:
            return set(self._stage_map.keys())
        expanded: set[str] = set()
        for pattern in during:
            if pattern in self._stage_map:
                expanded.add(pattern)
            else:
                expanded.update(s for s in self._stage_map if re.fullmatch(pattern, s))
        return expanded

    def _register_handler(self, callback: Callable[..., Any], msg_name: str, entry: MessageEntry) -> None:
        """Register a handler, checking for collisions with overlapping ``during`` sets."""
        if msg_name in self._handlers:
            for existing_cb, existing_entry in self._handlers[msg_name]:
                existing_expanded = self._expand_during(existing_entry.during)
                new_expanded = self._expand_during(entry.during)
                if existing_expanded & new_expanded:
                    existing_owner = type(getattr(existing_cb, "__self__", existing_cb)).__name__
                    new_owner = type(getattr(callback, "__self__", callback)).__name__
                    raise ValueError(
                        f"Handler collision: message '{msg_name}' is registered on both "
                        f"{existing_owner} and {new_owner} "
                        f"with overlapping or unscoped `during` sets. "
                        f"Use non-overlapping `during` to scope handlers to specific stages."
                    )
            self._handlers[msg_name].append((callback, entry))
        else:
            self._handlers[msg_name] = [(callback, entry)]

    def _validate_during_names(self) -> None:
        """
        Check that all ``during`` stage names in handlers reference existing stages.

        Supports regex patterns in ``during`` (e.g. ``"learning_.*"``).
        A pattern is valid if it matches at least one registered stage name.
        """
        available_stages = sorted(self._stage_map.keys())
        errors: list[str] = []
        for msg_name, entries in self._handlers.items():
            for _, entry in entries:
                if entry.during is not None:
                    for pattern in sorted(entry.during):
                        # Exact match
                        if pattern in self._stage_map:
                            continue
                        # Regex match — valid if at least one stage matches
                        if any(re.fullmatch(pattern, s) for s in self._stage_map):
                            continue
                        suggestions = get_close_matches(pattern, self._stage_map.keys(), n=1)
                        hint = f" Did you mean '{suggestions[0]}'?" if suggestions else ""
                        errors.append(
                            f"Handler '{msg_name}' has `during={{'{pattern}'}}` but '{pattern}' "
                            f"does not match any stage. Available: {', '.join(available_stages)}.{hint}"
                        )
        if errors:
            raise ValueError("\n".join(errors))

    #############
    #    Run    #
    #############

    async def run(
        self,
        experiment: Experiment,
        address: str,
        learner: Learner,
        aggregator: Aggregator,
        cp: CommunicationProtocol,
        generator: random.Random,
        on_ready: asyncio.Event | None = None,
    ) -> Experiment:
        """
        Run the workflow with an explicit Experiment and context parameters.

        The caller is responsible for constructing the ``Experiment`` instance.

        Args:
            experiment: A fully constructed Experiment describing this run.
            address: The node's network address.
            learner: The learner instance for training.
            aggregator: The aggregator instance for model aggregation.
            cp: The communication protocol for network operations.
            generator: Random number generator for reproducibility.
            on_ready: Event set when the workflow transitions to RUNNING,
                before the first stage executes.

        Returns:
            The Experiment with tracked data after completion.

        """
        self.error = None
        observer = ExperimentLoggerObserver(address)

        # 1. Build typed context
        logger.debug(address, "Workflow: creating context...")
        ctx = self.create_context(
            address=address,
            learner=learner,
            aggregator=aggregator,
            cp=cp,
            generator=generator,
            experiment=experiment,
        )

        # 2. Compose stages, wire context, build handler map
        logger.debug(address, "Workflow: composing stages...")
        self._compose(ctx)

        # 3. Attach observer and execute stage loop
        logger.debug(address, "Workflow: starting stage loop...")
        experiment.add_observer(observer)
        self.status = WorkflowStatus.RUNNING
        if on_ready is not None:
            on_ready.set()
            await asyncio.sleep(0)
        logger.experiment_started(ctx.address, experiment)
        try:
            await self._run(ctx)
            self.status = WorkflowStatus.FINISHED
            logger.info(ctx.address, "🏁 Learning finished.")
        except asyncio.CancelledError:
            if self.status != WorkflowStatus.FINISHED:
                self.status = WorkflowStatus.CANCELLED
            logger.info(address, "🛑 Learning cancelled.")
            raise
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            self.error = e
            logger.error(address, f"Learning failed: {e}")
            raise
        finally:
            experiment.remove_observer(observer)
            logger.experiment_ended(address, experiment, self.status.value)

        return ctx.experiment

    async def _run(self, ctx: TContext) -> None:
        """Run the workflow as a sequential stage loop."""
        self.stage_timings = []

        # Setup stage
        stage = self._stage_map[self.initial_stage]
        self._current_stage = stage
        ctx.experiment.current_stage = self.initial_stage
        t0 = time.perf_counter()
        stage_name: str | None = await stage.run()
        dt = time.perf_counter() - t0
        self.stage_timings.append(
            {
                "node": ctx.address,
                "stage": self.initial_stage,
                "round": ctx.experiment.round,
                "duration": dt,
                "start": t0,
            }
        )

        self.validate_experiment(ctx)

        # Remaining stages
        while stage_name is not None:
            current_name = stage_name
            stage = self._stage_map[stage_name]
            self._current_stage = stage
            ctx.experiment.current_stage = stage_name
            t0 = time.perf_counter()
            stage_name = await stage.run()
            dt = time.perf_counter() - t0
            self.stage_timings.append(
                {
                    "node": ctx.address,
                    "stage": current_name,
                    "round": ctx.experiment.round,
                    "duration": dt,
                    "start": t0,
                }
            )
        self._current_stage = None

    def validate_experiment(self, ctx: TContext) -> None:
        """Override to resolve dynamic defaults and validate experiment params after setup."""

    #########################
    #    Task Management    #
    #########################

    async def start(self, *args: Any, on_ready: asyncio.Event | None = None, **kwargs: Any) -> None:
        """
        Launch the workflow as a background ``asyncio.Task``.

        Takes the same arguments as ``run()``, plus an optional ``on_ready``
        event that is set once the workflow status transitions to RUNNING
        (before the first stage executes).

        Args:
            *args: Positional arguments forwarded to ``run()``.
            on_ready: Event set when the workflow becomes RUNNING.
            **kwargs: Keyword arguments forwarded to ``run()``.

        Raises:
            RuntimeError: If a workflow task is already running.

        """
        if self._task is not None and not self._task.done():
            raise RuntimeError("Workflow is already running")
        kwargs["on_ready"] = on_ready
        self._task = asyncio.create_task(self.run(*args, **kwargs))

    async def stop(self) -> None:
        """
        Cancel the workflow task and wait for it to finish.

        Safe to call when no task is running (no-op), when the task is
        already done (retrieves the exception to suppress the asyncio
        'exception was never retrieved' warning), or when actively running.
        """
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        elif self._task is not None and self._task.done():
            # Retrieve exception to suppress asyncio "exception never retrieved" warning
            if not self._task.cancelled():
                with contextlib.suppress(Exception):
                    self._task.exception()
        self._task = None

    async def wait(self) -> Experiment:
        """
        Await workflow completion and return the experiment.

        Propagates any exception raised during the workflow run.

        Returns:
            The Experiment with tracked data after completion.

        Raises:
            RuntimeError: If the workflow has not been started.

        """
        if self._task is None:
            raise RuntimeError("Workflow not started")
        return await self._task

    ####################
    #    Properties    #
    ####################

    @property
    def current_stage_name(self) -> str | None:
        """Get the name of the currently executing stage, or ``None`` if idle."""
        return self._current_stage.name if self._current_stage is not None else None

    @property
    def experiment(self) -> Experiment | None:
        """Get the experiment from the current stage's context, or ``None`` if idle."""
        return self._current_stage.ctx.experiment if self._current_stage is not None else None

    ##############################
    #    Message Registration    #
    ##############################

    def get_messages(self) -> dict[str, MessageEntry]:
        """
        Get all declared message entries.

        After ``_compose()``, returns entries from the live handler map.
        Before ``_compose()``, scans class-level registries (safe to call early).
        """
        if self._handlers:
            return {name: items[0][1] for name, items in self._handlers.items()}

        # Pre-compose fallback: extract types to read class-level _message_registry
        stage_classes = [type(s) for s in self._stage_map.values()] if self._stage_map else [type(s) for s in self.get_stages()]

        result: dict[str, MessageEntry] = {}
        for stage_cls in stage_classes:
            for cls in stage_cls.__mro__:
                if cls is Stage or cls is object:
                    break
                for msg_name, entry in cls.__dict__.get("_message_registry", {}).items():
                    result.setdefault(msg_name, entry)
        return result
