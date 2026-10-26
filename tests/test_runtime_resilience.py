from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from asr_server.adapters.base import AudioPath, TranscriptionResult
from asr_server.adapters.mock import MockAsrAdapter
from asr_server.config import Settings, load_settings
from asr_server.errors import AsrError
from asr_server.lifecycle import ModelLifecycleManager, is_recoverable_worker_error
from asr_server.main import create_app
from asr_server.registry import default_models
from asr_server.scheduler import OperationScheduler
from asr_server.workers.transport import ProcessRpcTransport
from tests.test_worker_transport import make_transport


class UnreapableProcess:
    pid = 987654
    exitcode = 0

    def __init__(self) -> None:
        self.alive = True
        self.closed = False

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float) -> None:
        pass

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def test_unreapable_generation_blocks_every_new_worker_until_exit() -> None:
    owner, following = make_transport("ok"), make_transport("ok")
    process = UnreapableProcess()
    owner._process = process
    try:
        owner.close(force=True)
        assert owner.pid == process.pid
        assert not process.closed
        with pytest.raises(AsrError, match="previous worker") as caught:
            following.start()
        assert caught.value.code == "worker_isolated"
        assert not is_recoverable_worker_error(caught.value)
        assert following.pid is None
        process.alive = False
        owner.close(force=True)
        following.start()
        assert following.request("work", timeout_seconds=1) == "ok"
    finally:
        process.alive = False
        owner.close(force=True)
        following.close(force=True)


def test_short_chunk_timeout_has_separate_deadline_and_reaps_worker(tmp_path: Any) -> None:
    transport = make_transport("hang")
    transport.name = "Qwen fixture"
    transport.configure_deadlines(chunk=0.03, alignment=0.1, native=1, observation=1)
    transport.start()
    with pytest.raises(AsrError) as caught:
        transport.request("transcribe", timeout_seconds=10, audio=AudioPath(tmp_path / "borrowed.wav", 0, 60))
    assert caught.value.details["phase"] == "chunk"
    assert caught.value.details["timeout_seconds"] == 0.03
    assert caught.value.details["worker_reaped"]
    assert not transport.is_alive


class RecoveringAdapter(MockAsrAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.fail = True
        self.calls = 0

    async def transcribe(self, *args: Any, **kwargs: Any) -> TranscriptionResult:
        self.calls += 1
        if self.fail:
            raise AsrError(503, "worker_unavailable", "injected EOF", {"worker_fatal": True, "worker_reaped": True})
        return TranscriptionResult("recovered", 1.0, "zh", [])


async def test_fatal_circuit_is_bounded_and_closes_only_after_real_success(monkeypatch: Any) -> None:
    now = [100.0]
    monkeypatch.setattr("asr_server.lifecycle.monotonic", lambda: now[0])
    adapter = RecoveringAdapter()
    manager = ModelLifecycleManager(default_models(), lambda _: adapter, idle_unload_seconds=0)
    async def infer() -> None:
        await manager.transcribe(b"fixture", model_id=None, backend="auto", language="auto", timestamps="none")
    for _ in range(3):
        with pytest.raises(AsrError):
            await infer()
    assert adapter.calls == 3
    for _ in range(3):
        assert not manager.inference_readiness()["ready"]
    with pytest.raises(AsrError) as caught:
        await infer()
    assert caught.value.code == "worker_circuit_open"
    assert adapter.calls == 3
    now[0] += 61
    assert manager.inference_readiness()["ready"]
    assert manager.inference_readiness()["ready"]  # reading does not consume probe
    adapter.fail = False
    await infer()
    assert manager.worker_diagnostics()["models"][0]["consecutive_failures"] == 0
    await manager.shutdown()


async def test_drain_keeps_pending_operations_and_rejects_new_admission() -> None:
    scheduler = OperationScheduler(4)
    entered, release, following = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def first() -> None:
        entered.set()
        await release.wait()
    async def second() -> None:
        following.set()
    first_op = await scheduler.submit("first", first)
    await entered.wait()
    second_op = await scheduler.submit("second", second)
    scheduler.set_draining(True)
    release.set()
    await first_op.future
    assert not scheduler.readiness()["ready"]
    assert not following.is_set()
    with pytest.raises(AsrError) as caught:
        await scheduler.submit("new", second)
    assert caught.value.code == "service_draining"
    assert scheduler.position("second") == 1
    scheduler.set_draining(False)
    await second_op.future
    await scheduler.shutdown(1, second)


async def test_readyz_preserves_health_and_read_access_during_drain() -> None:
    app = create_app(Settings(workbench_enabled=False))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/readyz")).status_code == 200
            app.state.job_manager.scheduler.set_draining(True)
            ready = await client.get("/readyz")
            assert ready.status_code == 503
            assert ready.json() == {"status": "unavailable", "reasons": ["draining"]}
            assert (await client.get("/health")).json()["status"] == "ok"
            assert (await client.get("/v1/models")).status_code == 200
            rejected = await client.post("/v1/models/qwen3-asr-1.7b/load", json={})
            assert rejected.status_code == 503
            assert rejected.json()["error"]["code"] == "service_draining"


def test_phase_deadlines_keep_legacy_envelope_until_explicitly_configured(monkeypatch: Any) -> None:
    monkeypatch.setenv("ASR_WORKER_INFERENCE_TIMEOUT_SECONDS", "1700")
    settings = load_settings()
    assert settings.worker_chunk_timeout_seconds == 1700
    assert settings.worker_native_timeout_seconds == 1700
    monkeypatch.setenv("ASR_WORKER_CHUNK_TIMEOUT_SECONDS", "400")
    monkeypatch.setenv("ASR_WORKER_ALIGNMENT_TIMEOUT_SECONDS", "600")
    settings = load_settings()
    assert settings.worker_chunk_timeout_seconds == 400
    assert settings.worker_alignment_timeout_seconds == 600
    assert settings.worker_native_timeout_seconds == 1700


async def test_scheduler_restarts_only_with_no_unconfirmed_owner() -> None:
    scheduler = OperationScheduler(3)
    async def finished() -> None:
        pass
    scheduler.worker = asyncio.create_task(finished())
    await scheduler.worker
    old_worker = scheduler.worker
    scheduler.running = "unconfirmed-owner"
    with pytest.raises(AsrError) as caught:
        await scheduler.start()
    assert caught.value.code == "scheduler_unavailable"
    assert scheduler.worker is old_worker
    scheduler.running = None
    await scheduler.start()
    operation = await scheduler.submit("healthy", finished)
    await operation.future
    await scheduler.shutdown(1, finished)


def test_gateway_watchdog_detects_stalled_event_loop_independently() -> None:
    import subprocess
    import sys
    script = """
import asyncio
import time
from asr_server.main import EventLoopWatchdog
async def main():
    watchdog = EventLoopWatchdog(0.05)
    watchdog.start()
    time.sleep(2)
asyncio.run(main())
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=5)
    assert result.returncode == 70
