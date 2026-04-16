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

"""Node monitor."""

import asyncio
import contextlib
import datetime
import platform
import subprocess
import sys
from collections.abc import Callable

import psutil  # type: ignore

from p2pfl.settings import Settings


class NodeMonitor:
    """Node monitor that periodically collects system resource metrics."""

    MAX_LOG_ENTRIES = 500

    def __init__(self) -> None:
        """Initialize the node monitor."""
        self.period = Settings.general.RESOURCE_MONITOR_PERIOD
        self.logs: dict[datetime.datetime, dict[str, float]] = {}
        self._task: asyncio.Task[None] | None = None
        self._callbacks: list[Callable[[datetime.datetime, dict[str, float]], None]] = []

    @property
    def running(self) -> bool:
        """Whether the monitor task is currently running."""
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start monitoring. Must be called from a running event loop."""
        if self.running:
            return
        with contextlib.suppress(RuntimeError):
            self._task = asyncio.get_running_loop().create_task(self._loop())

    def add_callback(self, cb: Callable[[datetime.datetime, dict[str, float]], None]) -> None:
        """Register a callback invoked each time metrics are collected."""
        self._callbacks.append(cb)

    def remove_callback(self, cb: Callable[[datetime.datetime, dict[str, float]], None]) -> None:
        """Unregister a callback (no-op if not registered)."""
        with contextlib.suppress(ValueError):
            self._callbacks.remove(cb)

    async def _loop(self) -> None:
        """Periodically collect system resource metrics."""
        while True:
            now = datetime.datetime.now(tz=datetime.timezone.utc)
            resources = self._report_system_resources()
            self.logs[now] = resources
            # Evict oldest entries when over the cap
            if len(self.logs) > self.MAX_LOG_ENTRIES:
                excess = len(self.logs) - self.MAX_LOG_ENTRIES
                for key in list(self.logs)[:excess]:
                    del self.logs[key]
            for cb in self._callbacks:
                cb(now, resources)
            await asyncio.sleep(self.period)

    def stop(self) -> None:
        """Cancel the monitoring task."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    def get_logs(self) -> dict[datetime.datetime, dict[str, float]]:
        """Get the collected resource logs."""
        return self.logs

    def _report_system_resources(self) -> dict[str, float]:
        """Report the system resources."""
        res: dict[str, float] = {}
        res["cpu"] = psutil.cpu_percent()
        res["ram"] = psutil.virtual_memory().percent

        # GPU metrics via nvidia-smi (best-effort)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                for i, line in enumerate(result.stdout.strip().splitlines()):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 3:
                        res[f"gpu{i}_util"] = float(parts[0])
                        res[f"gpu{i}_vram"] = (float(parts[1]) / float(parts[2])) * 100 if float(parts[2]) > 0 else 0.0
        except Exception:
            pass

        return res


def _detect_gpus() -> list[dict]:
    """Best-effort GPU detection: CUDA → nvidia-smi fallback → Apple MPS."""
    gpus: list[dict] = []

    # Try PyTorch CUDA
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    {
                        "name": props.name,
                        "memory_mb": props.total_memory // (1024 * 1024),
                        "backend": "cuda",
                    }
                )
            return gpus
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    gpus.append({"name": parts[0], "memory_mb": int(float(parts[1])), "backend": "cuda"})
            if gpus:
                return gpus
    except Exception:
        pass

    # Apple MPS
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpus.append({"name": "Apple MPS", "backend": "mps"})
    except Exception:
        pass

    return gpus


def _get_geolocation() -> dict | None:
    """Fetch geolocation from ip-api.com if enabled in settings."""
    if not Settings.general.WEB_GEOLOCATION:
        return None
    try:
        import httpx

        # ip-api.com free tier only supports HTTP (HTTPS requires paid plan)
        resp = httpx.get("http://ip-api.com/json", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "country": data.get("country"),
                "region": data.get("regionName"),
                "city": data.get("city"),
                "lat": data.get("lat"),
                "lon": data.get("lon"),
                "isp": data.get("isp"),
            }
    except Exception:
        pass
    return None


def collect_node_metadata() -> dict:
    """Collect system metadata for node registration. Never raises."""
    try:
        mem = psutil.virtual_memory()
        metadata: dict = {
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "hardware": {
                "cpu_count": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "ram_total_mb": mem.total // (1024 * 1024),
            },
            "runtime": {
                "python_version": platform.python_version(),
                "p2pfl_python": sys.executable,
            },
        }

        gpus = _detect_gpus()
        if gpus:
            metadata["gpus"] = gpus

        geo = _get_geolocation()
        if geo is not None:
            metadata["geolocation"] = geo

        return metadata
    except Exception:
        return {}
