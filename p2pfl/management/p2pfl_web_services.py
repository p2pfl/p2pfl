#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""Communication with P2PFL Web Services (via REST API)."""

from __future__ import annotations

import base64
import datetime
import gzip
import json
import os
import threading
import time
from typing import Any

import httpx

from p2pfl.settings import Settings


class P2pflWebServices:
    """
    Class that manages the communication with the p2pfl web services.

    All buffered data (logs, metrics, messages, experiment updates, system
    metrics) is accumulated in a single buffer and sent by a background
    thread every ``Settings.general.WEB_BATCH_INTERVAL`` seconds as gzipped
    ``POST /batch`` requests of at most ``Settings.general.WEB_BATCH_SIZE``
    entries. Failed sends are retried with exponential backoff.

    Each entry in the batch carries a ``type`` field so the backend can
    dispatch accordingly.

    Immediate (non-batched) operations like ``register_node`` and
    ``create_experiment`` use synchronous HTTP helpers.

    Args:
        url: The base URL of the web services API.
        key: The API key to access the services.

    """

    def __init__(self, url: str, key: str) -> None:
        """Initialize the p2pfl web services."""
        self._base_url = url.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "x-api-key": key,
        }
        # Maps node address -> server-assigned node id
        self.node_id: dict[str, int] = {}
        # Maps experiment name -> server-assigned experiment id
        self._exp_id: dict[str, int] = {}
        # Maps node address -> experiment name
        self._node_exp: dict[str, str] = {}

        # Single batch buffer — lock needed: singleton logger is called from multiple threads
        self._buffer: list[dict] = []
        self._lock = threading.Lock()
        self._flush_thread: threading.Thread | None = None
        self._running = False
        self._wake = threading.Event()  # set by stop() to end the wait early

        self._last_overflow_warn: float = 0.0
        self._awake = False  # set once the server has answered (see _wait_until_awake)

        # Sync HTTP client for immediate operations (register, unregister, sync flush)
        self._client = httpx.Client(headers=self._headers, timeout=5.0)

    # --- Lifecycle ---

    def _ensure_flush_thread(self) -> None:
        """Lazily start the background thread that sends the buffer."""
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        self._running = True
        self._wake.clear()
        t = threading.Thread(target=self._flush_loop, daemon=True, name="p2pfl-web-flush")
        t.start()
        self._flush_thread = t

    def _flush_loop(self) -> None:
        """
        Send the buffer every ``WEB_BATCH_INTERVAL`` seconds until stopped.

        A failed request puts the unsent entries back at the front of the
        buffer and waits with exponential backoff (2 s up to 60 s).
        """
        backoff = 2.0
        max_backoff = 60.0
        consecutive_failures = 0
        while self._running:
            self._wake.wait(Settings.general.WEB_BATCH_INTERVAL)
            if not self._running:
                return
            pending = self._drain()
            try:
                while pending:
                    self._post_batch(pending[: Settings.general.WEB_BATCH_SIZE])
                    pending = pending[Settings.general.WEB_BATCH_SIZE :]
            except Exception as e:
                if pending:
                    with self._lock:
                        self._buffer[:0] = pending
                consecutive_failures += 1
                if consecutive_failures == 1:
                    print(f"[P2PFL Web Services] Send failed: {e}, retrying in {backoff:.0f}s...")
                elif consecutive_failures % 10 == 0:
                    print(
                        f"[P2PFL Web Services] Web services still unreachable "
                        f"({consecutive_failures} consecutive failures), retrying in {backoff:.0f}s..."
                    )
                self._wake.wait(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue
            if consecutive_failures > 0:
                print(f"[P2PFL Web Services] Reconnected after {consecutive_failures} failed attempt(s).")
            backoff = 2.0
            consecutive_failures = 0

    def stop(self) -> None:
        """Stop the flush thread, flush remaining data, close client."""
        self._running = False
        self._wake.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5)
        self._flush_thread = None
        self.flush()
        self._client.close()

    # --- Batching internals ---

    def _enqueue(self, entry: dict) -> None:
        """Append an entry to the batch buffer. Never blocks."""
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) > Settings.general.WEB_MAX_BUFFER_SIZE:
                overflow = len(self._buffer) - Settings.general.WEB_MAX_BUFFER_SIZE
                del self._buffer[:overflow]
                now = time.monotonic()
                if now - self._last_overflow_warn >= 30.0:
                    self._last_overflow_warn = now
                    print(
                        f"[P2PFL Web Services] Buffer full — dropped {overflow} oldest {'entry' if overflow == 1 else 'entries'} "
                        f"(buffer capped at {Settings.general.WEB_MAX_BUFFER_SIZE})."
                    )
        self._ensure_flush_thread()

    def _drain(self) -> list[dict]:
        """Atomically drain the buffer."""
        with self._lock:
            entries = self._buffer
            self._buffer = []
        return entries

    def flush(self) -> None:
        """Flush all buffered entries synchronously (used during shutdown)."""
        entries = self._drain()
        if entries:
            self._sync_send(entries)

    def _post_batch(self, entries: list[dict]) -> None:
        """``POST /batch`` with a gzipped JSON body (raises on failure)."""
        body = gzip.compress(json.dumps(entries).encode(), compresslevel=6)
        response = self._client.post(
            self._base_url + "/batch",
            content=body,
            headers={"Content-Encoding": "gzip"},
            timeout=10,
        )
        response.raise_for_status()

    def _sync_send(self, entries: list[dict]) -> None:
        """Send everything now, in ``WEB_BATCH_SIZE`` chunks. Best-effort (drops on failure)."""
        size = Settings.general.WEB_BATCH_SIZE
        for start in range(0, len(entries), size):
            chunk = entries[start : start + size]
            try:
                self._post_batch(chunk)
            except Exception as e:
                print(f"[P2PFL Web Services] Dropped batch ({len(chunk)} entries): {e}")

    # --- HTTP helpers (sync, for immediate operations) ---

    def _wait_until_awake(self) -> None:
        """
        Wait for a sleeping server before the first synchronous call.

        Free hosting plans put the server to sleep when idle and it takes up to
        a minute to wake up, longer than the 5 s timeout of the calls a node
        makes at start-up (register, join the experiment). Polls ``GET /health``
        (any HTTP answer counts) for up to ``Settings.general.WEB_WAKE_TIMEOUT``
        seconds; afterwards the normal calls report any error.
        """
        if self._awake:
            return
        deadline = time.monotonic() + Settings.general.WEB_WAKE_TIMEOUT
        warned = False
        while True:
            try:
                self._client.get(self._base_url + "/health", timeout=10)
                break
            except httpx.HTTPError:
                if time.monotonic() >= deadline:
                    return
                if not warned:
                    print("[P2PFL Web Services] Waiting for the web services to wake up (up to a minute)...")
                    warned = True
                time.sleep(2)
        self._awake = True

    def _get(self, path: str, *, timeout: int = 5) -> dict:
        response = self._client.get(self._base_url + path, timeout=timeout)
        response.raise_for_status()
        result: dict = response.json()
        return result

    def _post(self, path: str, data: Any, *, timeout: int = 5) -> dict:
        response = self._client.post(self._base_url + path, json=data, timeout=timeout)
        response.raise_for_status()
        if response.status_code == 204:
            return {}
        result: dict = response.json()
        return result

    def _patch(self, path: str, data: Any, *, timeout: int = 5) -> None:
        response = self._client.patch(self._base_url + path, json=data, timeout=timeout)
        response.raise_for_status()

    def _delete(self, path: str, *, timeout: int = 5) -> None:
        response = self._client.delete(self._base_url + path, timeout=timeout)
        response.raise_for_status()

    # --- Nodes ---

    def register_node(self, node: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Register a node.

        Args:
            node: The node address.
            metadata: Optional system metadata (OS, hardware, GPU, geolocation).

        """
        self._wait_until_awake()
        try:
            data: dict[str, Any] = {"address": node}
            if metadata is not None:
                data["metadata"] = metadata
            result = self._post("/nodes", data)
            self.node_id[node] = result["id"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 409:
                # Node already registered — fetch its ID
                result = self._get(f"/nodes/{node}")
                self.node_id[node] = result["id"]
            else:
                print(f"[P2PFL Web Services] Failed to register node '{node}': {e}")
                print(f"  Check that '{self._base_url}' is valid (P2PFL_WEB_LOGGER_URL or ~/.p2pfl_env)")
                raise
        except httpx.HTTPError as e:
            print(f"[P2PFL Web Services] Failed to register node '{node}': {e}")
            print(f"  Check that '{self._base_url}' is valid (P2PFL_WEB_LOGGER_URL or ~/.p2pfl_env)")
            raise

    def update_node_status(self, node: str, state: str) -> None:
        """
        Update a node's lifecycle state.

        Args:
            node: The node address.
            state: The new state ("idle", "learning", "finished", "failed", "cancelled").

        """
        try:
            self._patch(f"/nodes/{node}/status", {"state": state})
        except httpx.HTTPError as e:
            print(f"[P2PFL Web Services] Failed to update node status for '{node}': {e}")

    def unregister_node(self, node: str) -> None:
        """
        Unregister a node.

        Args:
            node: The node address.

        """
        try:
            self._delete(f"/nodes/{node}")
        except httpx.HTTPError as e:
            print(f"[P2PFL Web Services] Failed to unregister node '{node}': {e}")

    # --- Experiments ---

    def create_experiment(self, node_address: str, **data: Any) -> int:
        """
        Create an experiment on the web services.

        ``node_address`` is separate because it is not part of the Experiment
        object — it identifies which node is joining the experiment (sent as
        a query parameter).  ``**data`` should come from
        ``experiment.to_dict()``.

        Args:
            node_address: The node participating in the experiment.
            **data: Experiment fields (exp_name, total_rounds, workflow, …).

        Returns:
            The server-assigned experiment ID.

        """
        exp_name: str = data.get("exp_name", "")
        self._node_exp[node_address] = exp_name
        self._wait_until_awake()

        # Always POST (sync, not batched) — the node needs the experiment
        # ID before it can proceed. The backend is idempotent on exp_name.
        result = self._post(f"/experiments?node_address={node_address}", data)
        exp_id = result["id"]
        self._exp_id[exp_name] = exp_id
        return exp_id

    def update_experiment(self, exp_name: str, node_address: str, **changes: Any) -> None:
        """
        Buffer an experiment state update.

        Args:
            exp_name: Experiment name (for ID lookup).
            node_address: The node reporting the change.
            **changes: Fields that changed (round, status, current_stage, …).

        """
        exp_id = self._exp_id.get(exp_name)
        if exp_id is None:
            return
        self._enqueue({"type": "experiment_update", "experiment_id": exp_id, "node_address": node_address, **changes})

    # --- Batched data methods ---

    def send_log(self, time: datetime.datetime, node: str, level: int | str, message: str) -> None:
        """
        Buffer a log message.

        Args:
            time: The time of the message.
            node: The node address.
            level: The log level.
            message: The message.

        """
        self._enqueue(
            {
                "type": "log",
                "timestamp": time.isoformat() if isinstance(time, datetime.datetime) else str(time),
                "node_address": node,
                "level": str(level),
                "message": message,
            }
        )

    def send_local_metric(self, exp: str, round: int, metric: str, node: str, value: float, step: int) -> None:
        """
        Buffer a local metric.

        Args:
            exp: The experiment name.
            round: The round.
            metric: The metric name.
            node: The node address.
            value: The metric value.
            step: The training step.

        """
        exp_id = self._exp_id.get(exp)
        if exp_id is None:
            return
        self._enqueue(
            {
                "type": "metric",
                "experiment_id": exp_id,
                "node_address": node,
                "metric_name": metric,
                "round": round,
                "step": step,
                "value": value,
                "metric_type": "local",
            }
        )

    def send_global_metric(self, exp: str, round: int, metric: str, node: str, value: float) -> None:
        """
        Buffer a global metric.

        Args:
            exp: The experiment name.
            round: The round.
            metric: The metric name.
            node: The node address.
            value: The metric value.

        """
        exp_id = self._exp_id.get(exp)
        if exp_id is None:
            return
        self._enqueue(
            {
                "type": "metric",
                "experiment_id": exp_id,
                "node_address": node,
                "metric_name": metric,
                "round": round,
                "value": value,
                "metric_type": "global",
            }
        )

    def send_system_metric(self, node: str, metric: str, value: float, time: datetime.datetime) -> None:
        """
        Buffer a system metric.

        Args:
            node: The node address.
            metric: The metric name.
            value: The value.
            time: The timestamp.

        """
        self._enqueue(
            {
                "type": "system_metric",
                "node_address": node,
                "timestamp": time.isoformat(),
                "metric": metric,
                "value": value,
            }
        )

    def send_communication_log(
        self,
        node: str,
        timestamp: datetime.datetime,
        direction: str,
        cmd: str,
        source_dest: str,
        package_type: str,
        package_size: int,
        round_num: int | None = None,
        additional_info: dict | None = None,
    ) -> None:
        """
        Buffer a communication log.

        Args:
            node: The node address.
            timestamp: The timestamp of the communication.
            direction: Direction of communication ("sent" or "received").
            cmd: The command or message type.
            source_dest: Source (if receiving) or destination (if sending) node.
            package_type: Type of package ("message" or "weights").
            package_size: Size of the package in bytes.
            round_num: The federated learning round number.
            additional_info: Additional information as a dictionary.

        """
        # Skip heartbeat messages — they flood the buffer with no diagnostic value
        if cmd == "beat":
            return
        exp_name = self._node_exp.get(node)
        exp_id = self._exp_id.get(exp_name) if exp_name else None
        if exp_id is None:
            return
        self._enqueue(
            {
                "type": "message",
                "experiment_id": exp_id,
                "node_address": node,
                "cmd": cmd,
                "direction": direction,
                "peer": source_dest,
                "round": round_num,
                "size_bytes": package_size,
                "metadata": {"package_type": package_type, **(additional_info or {})},
            }
        )

    def upload_profiling(self, exp_name: str, profile_dir: str) -> None:
        """
        Upload profiling .pstat files for an experiment.

        Args:
            exp_name: The experiment name.
            profile_dir: Local directory containing .pstat files.

        """
        exp_id = self._exp_id.get(exp_name)
        if exp_id is None:
            return

        files: list[dict[str, str]] = []
        for fname in os.listdir(profile_dir):
            if fname.endswith(".pstat"):
                fpath = os.path.join(profile_dir, fname)
                with open(fpath, "rb") as f:
                    files.append(
                        {
                            "filename": fname,
                            "data": base64.b64encode(f.read()).decode("ascii"),
                        }
                    )

        if not files:
            return

        try:
            self._post(
                f"/experiments/{exp_id}/profiling",
                {"files": files},
                timeout=30,
            )
        except Exception as e:
            print(f"[P2PFL Web Services] Failed to upload profiling data: {e}")
