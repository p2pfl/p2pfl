"""Tests for P2pflWebServices and WebP2PFLogger."""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import logging
import time
from datetime import timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from p2pfl.management.logger.decorators.web_logger import DictFormatter, P2pflWebLogHandler, WebP2PFLogger
from p2pfl.management.p2pfl_web_services import P2pflWebServices
from p2pfl.settings import Settings
from p2pfl.workflow.engine.experiment import Experiment

###
# Helpers
###


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    """Build a mock httpx.Response with raise_for_status wired up."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        error = httpx.HTTPStatusError("error", request=MagicMock(), response=resp)
        resp.raise_for_status.side_effect = error
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_services() -> tuple[P2pflWebServices, MagicMock]:
    """Create a P2pflWebServices with its internal httpx.Client mocked."""
    svc = P2pflWebServices("http://localhost:8000", "test-key")
    mock_client = MagicMock(spec=httpx.Client)
    svc._client = mock_client
    return svc, mock_client


###
# P2pflWebServices – constructor
###


class TestWebServicesInit:
    """Web Services Init tests."""

    def test_strips_trailing_slash(self):
        """Test strips trailing slash."""
        svc = P2pflWebServices("http://example.com/api/", "k")
        assert svc._base_url == "http://example.com/api"
        svc._client.close()

    def test_headers_contain_api_key(self):
        """Test headers contain api key."""
        svc = P2pflWebServices("http://x", "my-key")
        assert svc._headers["x-api-key"] == "my-key"
        assert svc._headers["Content-Type"] == "application/json"
        svc._client.close()


###
# P2pflWebServices – HTTP helpers
###


class TestHTTPHelpers:
    """H T T P Helpers tests."""

    def test_get_returns_json(self):
        """Test get returns json."""
        svc, client = _make_services()
        client.get.return_value = _mock_response(200, {"ok": True})
        result = svc._get("/foo")
        assert result == {"ok": True}
        client.get.assert_called_once()

    def test_post_returns_json(self):
        """Test post returns json."""
        svc, client = _make_services()
        client.post.return_value = _mock_response(200, {"id": 42})
        result = svc._post("/bar", {"x": 1})
        assert result == {"id": 42}
        client.post.assert_called_once()

    def test_post_204_returns_empty_dict(self):
        """Test post 204 returns empty dict."""
        svc, client = _make_services()
        resp = _mock_response(204)
        client.post.return_value = resp
        result = svc._post("/bar", {})
        assert result == {}

    def test_patch_calls_raise_for_status(self):
        """Test patch calls raise for status."""
        svc, client = _make_services()
        client.patch.return_value = _mock_response(200)
        svc._patch("/baz", {"state": "idle"})
        client.patch.assert_called_once()

    def test_delete_calls_raise_for_status(self):
        """Test delete calls raise for status."""
        svc, client = _make_services()
        client.delete.return_value = _mock_response(200)
        svc._delete("/node/x")
        client.delete.assert_called_once()

    def test_get_propagates_http_error(self):
        """Test get propagates http error."""
        svc, client = _make_services()
        client.get.return_value = _mock_response(500)
        with pytest.raises(httpx.HTTPStatusError):
            svc._get("/fail")

    def test_post_propagates_http_error(self):
        """Test post propagates http error."""
        svc, client = _make_services()
        client.post.return_value = _mock_response(500)
        with pytest.raises(httpx.HTTPStatusError):
            svc._post("/fail", {})


###
# P2pflWebServices – register / unregister node
###


class TestNodeRegistration:
    """Node Registration tests."""

    def test_register_node_stores_id(self):
        """Test register node stores id."""
        svc, client = _make_services()
        client.post.return_value = _mock_response(200, {"id": 7})
        svc.register_node("node-1")
        assert svc.node_id["node-1"] == 7
        client.post.assert_called_once()
        # Verify POST body contains address
        _, kwargs = client.post.call_args
        assert kwargs["json"]["address"] == "node-1"

    def test_register_node_with_metadata(self):
        """Test register node with metadata."""
        svc, client = _make_services()
        client.post.return_value = _mock_response(200, {"id": 8})
        meta = {"os": {"system": "Linux"}}
        svc.register_node("node-2", metadata=meta)
        assert svc.node_id["node-2"] == 8
        _, kwargs = client.post.call_args
        assert kwargs["json"]["metadata"] == meta

    def test_register_node_409_fetches_existing(self):
        """Test register node 409 fetches existing."""
        svc, client = _make_services()
        # POST returns 409
        client.post.return_value = _mock_response(409)
        # GET fallback returns the existing node
        client.get.return_value = _mock_response(200, {"id": 99})
        svc.register_node("existing-node")
        assert svc.node_id["existing-node"] == 99
        client.get.assert_called_once()

    def test_register_node_other_http_error_raises(self):
        """Test register node other http error raises."""
        svc, client = _make_services()
        client.post.return_value = _mock_response(500)
        with pytest.raises(httpx.HTTPStatusError):
            svc.register_node("node-x")

    def test_register_node_connection_error_raises(self):
        """Test register node connection error raises."""
        svc, client = _make_services()
        client.post.side_effect = httpx.ConnectError("refused")
        with pytest.raises(httpx.HTTPError):
            svc.register_node("node-y")

    def test_unregister_node_calls_delete(self):
        """Test unregister node calls delete."""
        svc, client = _make_services()
        client.delete.return_value = _mock_response(200)
        svc.unregister_node("node-1")
        client.delete.assert_called_once()
        assert "/nodes/node-1" in client.delete.call_args[0][0]

    def test_unregister_node_swallows_http_error(self):
        """Unregister should print but not raise on failure."""
        svc, client = _make_services()
        client.delete.return_value = _mock_response(404)
        # Should not raise
        svc.unregister_node("gone-node")


###
# P2pflWebServices – update_node_status
###


class TestUpdateNodeStatus:
    """Update Node Status tests."""

    def test_update_node_status_calls_patch(self):
        """Test update node status calls patch."""
        svc, client = _make_services()
        client.patch.return_value = _mock_response(200)
        svc.update_node_status("node-1", "learning")
        client.patch.assert_called_once()
        args, kwargs = client.patch.call_args
        assert "/nodes/node-1/status" in args[0]
        assert kwargs["json"] == {"state": "learning"}

    def test_update_node_status_swallows_error(self):
        """Test update node status swallows error."""
        svc, client = _make_services()
        client.patch.return_value = _mock_response(500)
        # Should not raise
        svc.update_node_status("node-1", "failed")


###
# P2pflWebServices – experiments
###


class TestExperiments:
    """Experiments tests."""

    def test_create_experiment_stores_id_and_node_mapping(self):
        """Test create experiment stores id and node mapping."""
        svc, client = _make_services()
        client.post.return_value = _mock_response(200, {"id": 42})
        exp_id = svc.create_experiment("node-1", exp_name="exp-a", total_rounds=5)
        assert exp_id == 42
        assert svc._exp_id["exp-a"] == 42
        assert svc._node_exp["node-1"] == "exp-a"

    def test_update_experiment_enqueues_when_exp_known(self):
        """Test update experiment enqueues when exp known."""
        svc, client = _make_services()
        svc._exp_id["exp-a"] = 42
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.update_experiment("exp-a", "node-1", round=2, status="running")
            mock_enqueue.assert_called_once()
            entry = mock_enqueue.call_args[0][0]
            assert entry["type"] == "experiment_update"
            assert entry["experiment_id"] == 42
            assert entry["round"] == 2
            assert entry["status"] == "running"

    def test_update_experiment_noop_when_exp_unknown(self):
        """Test update experiment noop when exp unknown."""
        svc, client = _make_services()
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.update_experiment("unknown-exp", "node-1", round=0)
            mock_enqueue.assert_not_called()


###
# P2pflWebServices – batched data methods
###


class TestBatchedData:
    """Batched Data tests."""

    def test_send_log_enqueues(self):
        """Test send log enqueues."""
        svc, _ = _make_services()
        now = datetime.datetime(2025, 1, 1, tzinfo=timezone.utc)
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.send_log(now, "node-1", "INFO", "hello world")
            entry = mock_enqueue.call_args[0][0]
            assert entry["type"] == "log"
            assert entry["node_address"] == "node-1"
            assert entry["level"] == "INFO"
            assert entry["message"] == "hello world"
            assert entry["timestamp"] == now.isoformat()

    def test_send_log_with_string_time(self):
        """Test send log with string time."""
        svc, _ = _make_services()
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.send_log("2025-01-01", "node-1", 20, "msg")  # type: ignore[arg-type]
            entry = mock_enqueue.call_args[0][0]
            assert entry["timestamp"] == "2025-01-01"
            assert entry["level"] == "20"

    def test_send_local_metric_enqueues_when_exp_known(self):
        """Test send local metric enqueues when exp known."""
        svc, _ = _make_services()
        svc._exp_id["exp-a"] = 10
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.send_local_metric("exp-a", 1, "loss", "node-1", 0.5, step=3)
            entry = mock_enqueue.call_args[0][0]
            assert entry["type"] == "metric"
            assert entry["metric_type"] == "local"
            assert entry["experiment_id"] == 10
            assert entry["step"] == 3
            assert entry["value"] == 0.5

    def test_send_local_metric_noop_when_exp_unknown(self):
        """Test send local metric noop when exp unknown."""
        svc, _ = _make_services()
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.send_local_metric("missing", 0, "loss", "n", 0.0, 0)
            mock_enqueue.assert_not_called()

    def test_send_global_metric_enqueues_when_exp_known(self):
        """Test send global metric enqueues when exp known."""
        svc, _ = _make_services()
        svc._exp_id["exp-b"] = 20
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.send_global_metric("exp-b", 2, "accuracy", "node-2", 0.95)
            entry = mock_enqueue.call_args[0][0]
            assert entry["type"] == "metric"
            assert entry["metric_type"] == "global"
            assert entry["round"] == 2
            assert "step" not in entry

    def test_send_global_metric_noop_when_exp_unknown(self):
        """Test send global metric noop when exp unknown."""
        svc, _ = _make_services()
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.send_global_metric("missing", 0, "acc", "n", 0.0)
            mock_enqueue.assert_not_called()

    def test_send_system_metric_enqueues(self):
        """Test send system metric enqueues."""
        svc, _ = _make_services()
        now = datetime.datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.send_system_metric("node-1", "cpu", 42.5, now)
            entry = mock_enqueue.call_args[0][0]
            assert entry["type"] == "system_metric"
            assert entry["metric"] == "cpu"
            assert entry["value"] == 42.5
            assert entry["timestamp"] == now.isoformat()


###
# P2pflWebServices – communication log
###


class TestCommunicationLog:
    """Communication Log tests."""

    def test_send_communication_log_enqueues(self):
        """Test send communication log enqueues."""
        svc, _ = _make_services()
        svc._exp_id["exp-a"] = 10
        svc._node_exp["node-1"] = "exp-a"
        now = datetime.datetime(2025, 1, 1, tzinfo=timezone.utc)
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.send_communication_log(
                node="node-1",
                timestamp=now,
                direction="sent",
                cmd="aggregate",
                source_dest="node-2",
                package_type="weights",
                package_size=1024,
                round_num=3,
                additional_info={"foo": "bar"},
            )
            entry = mock_enqueue.call_args[0][0]
            assert entry["type"] == "message"
            assert entry["experiment_id"] == 10
            assert entry["cmd"] == "aggregate"
            assert entry["direction"] == "sent"
            assert entry["size_bytes"] == 1024
            assert entry["metadata"]["package_type"] == "weights"
            assert entry["metadata"]["foo"] == "bar"

    def test_skips_heartbeat_messages(self):
        """Test skips heartbeat messages."""
        svc, _ = _make_services()
        svc._exp_id["exp-a"] = 10
        svc._node_exp["node-1"] = "exp-a"
        now = datetime.datetime.now(tz=timezone.utc)
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.send_communication_log("node-1", now, "sent", "beat", "node-2", "message", 50)
            mock_enqueue.assert_not_called()

    def test_noop_when_no_experiment(self):
        """Test noop when no experiment."""
        svc, _ = _make_services()
        # node-1 has no experiment mapping
        now = datetime.datetime.now(tz=timezone.utc)
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.send_communication_log("node-1", now, "sent", "cmd", "node-2", "message", 100)
            mock_enqueue.assert_not_called()

    def test_noop_when_exp_name_not_in_exp_id(self):
        """Test noop when exp name not in exp id."""
        svc, _ = _make_services()
        svc._node_exp["node-1"] = "exp-unknown"
        # exp_id does not have "exp-unknown"
        now = datetime.datetime.now(tz=timezone.utc)
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.send_communication_log("node-1", now, "sent", "cmd", "n2", "message", 50)
            mock_enqueue.assert_not_called()

    def test_additional_info_defaults_to_empty(self):
        """Test additional info defaults to empty."""
        svc, _ = _make_services()
        svc._exp_id["exp-a"] = 10
        svc._node_exp["node-1"] = "exp-a"
        now = datetime.datetime.now(tz=timezone.utc)
        with patch.object(svc, "_enqueue") as mock_enqueue:
            svc.send_communication_log("node-1", now, "sent", "cmd", "n2", "message", 50)
            entry = mock_enqueue.call_args[0][0]
            assert entry["metadata"] == {"package_type": "message"}


###
# P2pflWebServices – buffer / enqueue / drain / flush
###


class TestBufferMechanics:
    """Buffer Mechanics tests."""

    def test_enqueue_and_drain(self):
        """Test enqueue and drain."""
        svc, _ = _make_services()
        # Suppress flush thread creation
        with patch.object(svc, "_ensure_flush_thread"):
            svc._enqueue({"type": "log", "msg": "a"})
            svc._enqueue({"type": "log", "msg": "b"})
        entries = svc._drain()
        assert len(entries) == 2
        # Second drain is empty
        assert svc._drain() == []

    def test_enqueue_respects_max_buffer_size(self):
        """Test enqueue respects max buffer size."""
        svc, _ = _make_services()
        Settings.general.WEB_MAX_BUFFER_SIZE = 3
        with patch.object(svc, "_ensure_flush_thread"):
            for i in range(5):
                svc._enqueue({"i": i})
        entries = svc._drain()
        assert len(entries) == 3
        # Oldest entries should have been dropped; newest kept
        assert entries[0]["i"] == 2
        assert entries[2]["i"] == 4

    def test_enqueue_overflow_prints_warning(self, capsys):
        """Buffer overflow prints a warning with dropped count."""
        svc, _ = _make_services()
        svc._last_overflow_warn = 0.0
        Settings.general.WEB_MAX_BUFFER_SIZE = 3
        with patch.object(svc, "_ensure_flush_thread"):
            for i in range(5):
                svc._enqueue({"i": i})
        captured = capsys.readouterr()
        assert "Buffer full" in captured.out
        assert "dropped 1 oldest entry" in captured.out

    def test_enqueue_overflow_warning_is_rate_limited(self, capsys):
        """Overflow warning fires at most once per 30s."""
        svc, _ = _make_services()
        Settings.general.WEB_MAX_BUFFER_SIZE = 2
        with patch.object(svc, "_ensure_flush_thread"):
            # First overflow
            for i in range(4):
                svc._enqueue({"i": i})
            # Second overflow — should be suppressed (same monotonic window)
            for i in range(4):
                svc._enqueue({"i": i})
        captured = capsys.readouterr()
        assert captured.out.count("Buffer full") == 1

    def test_flush_calls_sync_send(self):
        """Test flush calls sync send."""
        svc, client = _make_services()
        with patch.object(svc, "_ensure_flush_thread"):
            svc._enqueue({"type": "test"})
        client.post.return_value = _mock_response(200)
        svc.flush()
        # POST to /batch
        client.post.assert_called_once()
        url = client.post.call_args[0][0]
        assert url.endswith("/batch")

    def test_flush_noop_when_empty(self):
        """Test flush noop when empty."""
        svc, client = _make_services()
        svc.flush()
        client.post.assert_not_called()

    def test_batches_are_gzipped(self):
        """The /batch body is gzipped JSON with Content-Encoding: gzip."""
        import gzip as _gzip

        svc, client = _make_services()
        with patch.object(svc, "_ensure_flush_thread"):
            svc._enqueue({"type": "log", "message": "hi"})
        client.post.return_value = _mock_response(200)
        svc.flush()
        kwargs = client.post.call_args.kwargs
        assert kwargs["headers"] == {"Content-Encoding": "gzip"}
        assert json.loads(_gzip.decompress(kwargs["content"])) == [{"type": "log", "message": "hi"}]

    def test_flush_sends_in_batch_size_chunks(self):
        """A large buffer is sent in WEB_BATCH_SIZE chunks."""
        svc, client = _make_services()
        client.post.return_value = _mock_response(200)
        with patch.object(Settings.general, "WEB_BATCH_SIZE", 2):
            svc._sync_send([{"type": "log", "i": i} for i in range(5)])
        assert client.post.call_count == 3

    def test_flush_loop_sends_periodically(self):
        """The background thread posts the buffer every WEB_BATCH_INTERVAL."""
        svc, client = _make_services()
        client.post.return_value = _mock_response(200)
        with patch.object(Settings.general, "WEB_BATCH_INTERVAL", 0.01):
            svc._enqueue({"type": "log"})
            deadline = time.monotonic() + 2
            while client.post.call_count == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            svc._running = False
            svc._wake.set()
            svc._flush_thread.join(timeout=2)
        assert client.post.call_count >= 1
        assert svc._buffer == []

    def test_flush_loop_requeues_on_failure(self):
        """A failed send puts the entries back at the front of the buffer."""
        svc, client = _make_services()
        client.post.side_effect = httpx.ConnectError("down")
        with patch.object(Settings.general, "WEB_BATCH_INTERVAL", 0.01):
            svc._enqueue({"type": "log", "i": 1})
            deadline = time.monotonic() + 2
            while client.post.call_count == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            svc._running = False
            svc._wake.set()
            svc._flush_thread.join(timeout=2)
        assert {"type": "log", "i": 1} in svc._buffer

    def test_stop_wakes_the_flush_thread(self):
        """stop() ends the interval wait at once instead of after WEB_BATCH_INTERVAL."""
        svc, client = _make_services()
        client.post.return_value = _mock_response(200)
        with patch.object(Settings.general, "WEB_BATCH_INTERVAL", 60):
            svc._enqueue({"type": "log"})
            started = time.monotonic()
            svc.stop()
        assert time.monotonic() - started < 5
        client.post.assert_called_once()  # the final flush


###
# P2pflWebServices – upload_profiling
###


class TestUploadProfiling:
    """Upload Profiling tests."""

    def test_uploads_pstat_files(self, tmp_path):
        """Test uploads pstat files."""
        svc, client = _make_services()
        svc._exp_id["exp-a"] = 5
        # Create fake .pstat files
        (tmp_path / "profile1.pstat").write_bytes(b"data1")
        (tmp_path / "profile2.pstat").write_bytes(b"data2")
        (tmp_path / "readme.txt").write_text("ignore me")

        client.post.return_value = _mock_response(200)
        svc.upload_profiling("exp-a", str(tmp_path))
        client.post.assert_called_once()
        url = client.post.call_args[0][0]
        assert "/experiments/5/profiling" in url
        payload = client.post.call_args[1]["json"]
        assert len(payload["files"]) == 2

    def test_noop_when_exp_unknown(self, tmp_path):
        """Test noop when exp unknown."""
        svc, client = _make_services()
        svc.upload_profiling("unknown", str(tmp_path))
        client.post.assert_not_called()

    def test_noop_when_no_pstat_files(self, tmp_path):
        """Test noop when no pstat files."""
        svc, client = _make_services()
        svc._exp_id["exp-a"] = 5
        (tmp_path / "other.txt").write_text("nope")
        svc.upload_profiling("exp-a", str(tmp_path))
        client.post.assert_not_called()


###
# P2pflWebServices – lifecycle (stop)
###


class TestLifecycle:
    """Lifecycle tests."""

    def test_stop_flushes_and_closes(self):
        """Test stop flushes and closes."""
        svc, client = _make_services()
        with patch.object(svc, "flush") as mock_flush:
            svc.stop()
            mock_flush.assert_called_once()
        client.close.assert_called_once()


###
# WebP2PFLogger tests
###


class TestWebP2PFLogger:
    """Tests for the WebP2PFLogger decorator — mocks inner logger and web services."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Ensure web env vars are not set and testing mode is on."""
        monkeypatch.setenv("P2PFL_TESTING", "1")
        monkeypatch.delenv("P2PFL_WEB_LOGGER_URL", raising=False)
        monkeypatch.delenv("P2PFL_WEB_LOGGER_KEY", raising=False)

    def _make_logger(self, *, connected: bool = True):
        """Create a WebP2PFLogger with a mocked inner logger and optionally a mocked web services."""
        inner = MagicMock(spec=P2PFLoggerStub)
        inner.node_monitor = MagicMock()
        inner.node_monitor.add_callback = MagicMock()
        inner.node_monitor.remove_callback = MagicMock()

        logger = WebP2PFLogger.__new__(WebP2PFLogger)
        logger._p2pfl_logger = inner
        logger.node_monitor = inner.node_monitor
        logger._p2pfl_web_services = None
        logger._ended_experiments = set()
        logger._node_callbacks = {}
        logger._pending_futures = []

        logger._lifecycle_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        if connected:
            logger._p2pfl_web_services = MagicMock(spec=P2pflWebServices)

        return logger, inner

    # -- connect --

    def test_connect_creates_services_from_args(self, monkeypatch):
        """Test connect creates services from args."""
        inner = MagicMock(spec=P2PFLoggerStub)
        inner.node_monitor = MagicMock()

        logger = WebP2PFLogger.__new__(WebP2PFLogger)
        logger._p2pfl_logger = inner
        logger.node_monitor = inner.node_monitor
        logger._p2pfl_web_services = None
        logger._ended_experiments = set()
        logger._node_callbacks = {}
        logger._pending_futures = []

        logger._lifecycle_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        with patch("p2pfl.management.logger.decorators.web_logger.P2pflWebServices") as MockWS:
            MockWS.return_value = MagicMock()
            logger.connect(p2pfl_web_url="http://test", p2pfl_web_key="key123")
            MockWS.assert_called_once_with("http://test", "key123")
            assert logger._p2pfl_web_services is not None

    def test_connect_noop_when_missing_url_or_key(self):
        """Test connect noop when missing url or key."""
        logger, _ = self._make_logger(connected=False)
        logger.connect(p2pfl_web_url="http://x")
        assert logger._p2pfl_web_services is None

    def test_connect_noop_when_already_connected(self):
        """Test connect noop when already connected."""
        logger, _ = self._make_logger(connected=True)
        original = logger._p2pfl_web_services
        logger.connect(p2pfl_web_url="http://x", p2pfl_web_key="k")
        assert logger._p2pfl_web_services is original

    # -- register_node / unregister_node --

    def test_register_node_delegates_to_inner_and_submits_background(self):
        """Test register node delegates to inner and submits background."""
        logger, inner = self._make_logger(connected=True)

        # collect_node_metadata is imported lazily inside the function, so
        # patch at its defining module
        with patch("p2pfl.management.node_monitor.collect_node_metadata", return_value={"os": "test"}):
            logger.register_node("node-1")

        inner.register_node.assert_called_once_with("node-1")
        inner.node_monitor.add_callback.assert_called_once()
        # A future should be pending
        assert len(logger._pending_futures) == 1

    def test_register_node_no_web_still_delegates(self):
        """Test register node no web still delegates."""
        logger, inner = self._make_logger(connected=False)
        logger.register_node("node-1")
        inner.register_node.assert_called_once_with("node-1")
        assert len(logger._pending_futures) == 0

    def test_unregister_node_removes_callback_and_delegates(self):
        """Test unregister node removes callback and delegates."""
        logger, inner = self._make_logger(connected=True)
        # Simulate a registered callback
        cb = MagicMock()
        logger._node_callbacks["node-1"] = cb
        logger.unregister_node("node-1")
        inner.node_monitor.remove_callback.assert_called_once_with(cb)
        inner.unregister_node.assert_called_once_with("node-1")
        assert "node-1" not in logger._node_callbacks

    def test_unregister_node_no_web_still_delegates(self):
        """Test unregister node no web still delegates."""
        logger, inner = self._make_logger(connected=False)
        logger.unregister_node("node-1")
        inner.unregister_node.assert_called_once_with("node-1")

    # -- experiment_started --

    def test_experiment_started_creates_experiment_on_web(self):
        """Test experiment started creates experiment on web."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        exp = Experiment(exp_name="exp-1", total_rounds=5)

        logger.experiment_started("node-1", exp)

        ws.create_experiment.assert_called_once()
        ws.update_node_status.assert_called_once_with("node-1", "learning")
        inner.experiment_started.assert_called_once_with("node-1", exp)

    def test_experiment_started_swallows_web_error(self):
        """Test experiment started swallows web error."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        ws.create_experiment.side_effect = Exception("network error")
        exp = Experiment(exp_name="exp-1", total_rounds=5)

        # Should not raise
        logger.experiment_started("node-1", exp)
        inner.experiment_started.assert_called_once()

    def test_experiment_started_no_web_only_delegates(self):
        """Test experiment started no web only delegates."""
        logger, inner = self._make_logger(connected=False)
        exp = Experiment(exp_name="exp-1", total_rounds=3)
        logger.experiment_started("node-1", exp)
        inner.experiment_started.assert_called_once()

    # -- experiment_ended --

    def test_experiment_ended_updates_and_flushes(self):
        """Test experiment ended updates and flushes."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        exp = Experiment(exp_name="exp-1", total_rounds=5)

        logger.experiment_ended("node-1", exp, "finished")

        ws.update_experiment.assert_called_once_with("exp-1", "node-1", status="finished")
        ws.update_node_status.assert_called_once_with("node-1", "idle")
        ws.flush.assert_called_once()
        inner.experiment_ended.assert_called_once_with("node-1", exp, "finished")

    def test_experiment_ended_sets_failed_state(self):
        """Test experiment ended sets failed state."""
        logger, _ = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        exp = Experiment(exp_name="exp-1", total_rounds=5)

        logger.experiment_ended("node-1", exp, "failed")
        ws.update_node_status.assert_called_once_with("node-1", "failed")

    def test_experiment_ended_sets_cancelled_state(self):
        """Test experiment ended sets cancelled state."""
        logger, _ = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        exp = Experiment(exp_name="exp-1", total_rounds=5)

        logger.experiment_ended("node-1", exp, "cancelled")
        ws.update_node_status.assert_called_once_with("node-1", "cancelled")

    def test_experiment_ended_deduplicates(self):
        """Test experiment ended deduplicates."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        exp = Experiment(exp_name="exp-1", total_rounds=5)

        logger.experiment_ended("node-1", exp, "finished")
        logger.experiment_ended("node-1", exp, "finished")

        # Second call should be a no-op for web services
        assert ws.update_experiment.call_count == 1
        # But parent is always called
        assert inner.experiment_ended.call_count == 2

    def test_experiment_ended_swallows_web_error(self):
        """Test experiment ended swallows web error."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        ws.update_experiment.side_effect = Exception("boom")
        exp = Experiment(exp_name="exp-1", total_rounds=5)

        logger.experiment_ended("node-1", exp, "finished")
        inner.experiment_ended.assert_called_once()

    # -- on_experiment_change --

    def test_on_experiment_change_forwards_to_web(self):
        """Test on experiment change forwards to web."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        exp = Experiment(exp_name="exp-1", total_rounds=5)
        inner.get_nodes.return_value = {"node-1": {"Experiment": exp}}

        logger.on_experiment_change("node-1", "round", 3)

        inner.on_experiment_change.assert_called_once_with("node-1", "round", 3)
        ws.update_experiment.assert_called_once_with("exp-1", "node-1", round=3)

    def test_on_experiment_change_noop_when_node_not_found(self):
        """Test on experiment change noop when node not found."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        inner.get_nodes.return_value = {}

        logger.on_experiment_change("unknown", "round", 1)
        ws.update_experiment.assert_not_called()

    def test_on_experiment_change_noop_when_no_experiment(self):
        """Test on experiment change noop when no experiment."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        inner.get_nodes.return_value = {"node-1": {}}

        logger.on_experiment_change("node-1", "round", 1)
        ws.update_experiment.assert_not_called()

    def test_on_experiment_change_no_web(self):
        """Test on experiment change no web."""
        logger, inner = self._make_logger(connected=False)
        logger.on_experiment_change("node-1", "round", 1)
        inner.on_experiment_change.assert_called_once()

    # -- log_metric --

    def test_log_metric_local(self):
        """Test log metric local."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        exp = Experiment(exp_name="exp-1", total_rounds=5)
        inner.get_nodes.return_value = {"node-1": {"Experiment": exp, "round": 2}}

        logger.log_metric("node-1", "loss", 0.5, step=10, round=2)

        inner.log_metric.assert_called_once()
        ws.send_local_metric.assert_called_once_with("exp-1", 2, "loss", "node-1", 0.5, 10)

    def test_log_metric_global(self):
        """Test log metric global."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        exp = Experiment(exp_name="exp-1", total_rounds=5)
        inner.get_nodes.return_value = {"node-1": {"Experiment": exp, "round": 3}}

        logger.log_metric("node-1", "accuracy", 0.95, step=None, round=3)

        ws.send_global_metric.assert_called_once_with("exp-1", 3, "accuracy", "node-1", 0.95)

    def test_log_metric_uses_stored_round_when_none(self):
        """Test log metric uses stored round when none."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        exp = Experiment(exp_name="exp-1", total_rounds=5)
        inner.get_nodes.return_value = {"node-1": {"Experiment": exp, "round": 7}}

        logger.log_metric("node-1", "acc", 0.9, step=None, round=None)

        ws.send_global_metric.assert_called_once_with("exp-1", 7, "acc", "node-1", 0.9)

    def test_log_metric_skips_when_node_not_registered(self):
        """Test log metric skips when node not registered."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        inner.get_nodes.return_value = {}

        logger.log_metric("unknown", "loss", 0.5, step=1, round=1)

        ws.send_local_metric.assert_not_called()
        ws.send_global_metric.assert_not_called()

    def test_log_metric_no_web(self):
        """Test log metric no web."""
        logger, inner = self._make_logger(connected=False)
        logger.log_metric("node-1", "loss", 0.5, step=1, round=1)
        inner.log_metric.assert_called_once()

    # -- log_communication --

    def test_log_communication_sends_to_web(self):
        """Test log communication sends to web."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services

        logger.log_communication(
            node="node-1",
            direction="sent",
            cmd="aggregate",
            source_dest="node-2",
            package_type="weights",
            package_size=2048,
            round_num=5,
        )

        inner.log_communication.assert_called_once()
        ws.send_communication_log.assert_called_once()
        kwargs = ws.send_communication_log.call_args[1]
        assert kwargs["node"] == "node-1"
        assert kwargs["cmd"] == "aggregate"
        assert kwargs["package_size"] == 2048

    def test_log_communication_no_web(self):
        """Test log communication no web."""
        logger, inner = self._make_logger(connected=False)
        logger.log_communication("n", "sent", "cmd", "n2", "message", 100)
        inner.log_communication.assert_called_once()

    def test_log_communication_swallows_web_error(self):
        """Test log communication swallows web error."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        ws.send_communication_log.side_effect = Exception("network fail")

        # Should not raise
        logger.log_communication("n", "sent", "cmd", "n2", "message", 100)
        inner.log_communication.assert_called_once()

    # -- finish / reset / cleanup --

    def test_finish_flushes_web(self):
        """Test finish flushes web."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        logger.finish()
        ws.flush.assert_called_once()
        inner.finish.assert_called_once()

    def test_finish_no_web(self):
        """Test finish no web."""
        logger, inner = self._make_logger(connected=False)
        logger.finish()
        inner.finish.assert_called_once()

    def test_reset_clears_ended_experiments(self):
        """Test reset clears ended experiments."""
        logger, inner = self._make_logger(connected=True)
        logger._ended_experiments.add(("exp-1", "node-1"))
        logger.reset()
        assert len(logger._ended_experiments) == 0
        inner.reset.assert_called_once()

    def test_cleanup_stops_web_services(self):
        """Test cleanup stops web services."""
        logger, inner = self._make_logger(connected=True)
        ws = logger._p2pfl_web_services
        logger.cleanup()
        ws.stop.assert_called_once()
        inner.cleanup.assert_called_once()

    def test_cleanup_waits_for_pending_futures(self):
        """Test cleanup waits for pending futures."""
        logger, inner = self._make_logger(connected=True)
        fut = MagicMock()
        logger._pending_futures = [fut]
        logger.cleanup()
        fut.result.assert_called_once_with(timeout=10)

    def test_cleanup_no_web(self):
        """Test cleanup no web."""
        logger, inner = self._make_logger(connected=False)
        logger.cleanup()
        inner.cleanup.assert_called_once()


###
# DictFormatter / P2pflWebLogHandler
###


class TestDictFormatterAndHandler:
    """Dict Formatter And Handler tests."""

    def test_dict_formatter_extracts_fields(self):
        """Test dict formatter extracts fields."""
        formatter = DictFormatter()
        record = logging.LogRecord("p2pfl", logging.INFO, "", 0, "test message", (), None)
        record.node = "node-1"  # type: ignore[attr-defined]
        result = formatter.format(record)
        assert result["level"] == "INFO"
        assert result["node"] == "node-1"
        assert result["message"] == "test message"
        assert isinstance(result["timestamp"], datetime.datetime)

    def test_dict_formatter_raises_without_node(self):
        """Test dict formatter raises without node."""
        formatter = DictFormatter()
        record = logging.LogRecord("p2pfl", logging.INFO, "", 0, "msg", (), None)
        with pytest.raises(ValueError, match="node"):
            formatter.format(record)

    def test_log_handler_sends_to_web_services(self):
        """Test log handler sends to web services."""
        mock_ws = MagicMock(spec=P2pflWebServices)
        handler = P2pflWebLogHandler(mock_ws)
        record = logging.LogRecord("p2pfl", logging.WARNING, "", 0, "warning msg", (), None)
        record.node = "node-1"  # type: ignore[attr-defined]
        handler.emit(record)

        mock_ws.send_log.assert_called_once()
        args = mock_ws.send_log.call_args[0]
        assert args[1] == "node-1"
        assert args[2] == "WARNING"
        assert args[3] == "warning msg"


###
# Stub used for MagicMock spec on the inner logger
###


class P2PFLoggerStub:
    """Minimal stub providing the interface that WebP2PFLogger calls on its wrapped logger."""

    node_monitor: MagicMock

    def register_node(self, address: str) -> None:
        """Register node."""

    def unregister_node(self, address: str) -> None:
        """Unregister node."""

    def experiment_started(self, node: str, experiment: object) -> None:
        """Experiment started."""

    def experiment_ended(self, address: str, experiment: object, status: str) -> None:
        """Experiment ended."""

    def on_experiment_change(self, address: str, field_name: str, value: object) -> None:
        """On experiment change."""

    def log(self, level: int, node: str, message: str) -> None:
        """Log."""

    def log_metric(self, **kwargs: object) -> None:
        """Log metric."""

    def log_communication(self, **kwargs: object) -> None:
        """Log communication."""

    def get_nodes(self) -> dict:
        """Get nodes."""
        return {}

    def add_handler(self, handler: object) -> None:
        """Add handler."""

    def finish(self) -> None:
        """Finish."""

    def cleanup(self) -> None:
        """Cleanup."""

    def reset(self) -> None:
        """Reset."""

    def set_level(self, level: object) -> None:
        """Set level."""

    def get_level(self) -> int:
        """Get level."""
        return 0

    def info(self, node: str, message: str) -> None:
        """Info."""

    def debug(self, node: str, message: str) -> None:
        """Debug."""

    def warning(self, node: str, message: str) -> None:
        """Warning."""
