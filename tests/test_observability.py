from __future__ import annotations

import json
import asyncio
import logging
import os
import queue
import sqlite3
import tarfile
from multiprocessing.connection import Connection
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from asr_server.adapters.base import AudioInput, TranscriptionResult
from asr_server.adapters.mock import MockAsrAdapter
from asr_server.config import Settings
from asr_server.engines import EngineRegistration
from asr_server.errors import AsrError
from asr_server.main import create_app
from asr_server.observability import (
    BoundedQueueHandler, ExecutionContext, MAX_RECORD_BYTES, bind_context,
    JsonFormatter, _PrivateRotatingFileHandler, configure_logging, emit, event_record, safe_exception,
)
from asr_server.workers.transport import ProcessRpcTransport
from asr_server.workers.codec import result_from_payload
from asr_server.registry import default_models
from scripts.asr_diagnostics import create_bundle, trace_job


def diagnostics_worker(conn: Connection) -> None:
    try:
        while True:
            request = conn.recv()
            if request["op"] == "ping":
                conn.send({"id": request["id"], "ok": True, "result": {"pid": os.getpid(), "protocol_version": 1}})
            else:
                assert isinstance(request["execution_context"]["job_id"], str)
                try:
                    raise ValueError("do not log this private transcript, password or path /private/input.wav")
                except ValueError as exc:
                    conn.send({"id": request["id"], "ok": False,
                               "error": {"status_code": 503, "code": "inference_failed", "message": "failed"},
                               "diagnostics": safe_exception(exc), "received_context": request.get("execution_context")})
    except EOFError:
        return
    finally:
        conn.close()


class RpcFailureAdapter(MockAsrAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.transport = ProcessRpcTransport(name="test-rpc", target=diagnostics_worker, target_args=(),
                                             terminate_timeout_seconds=0.2, kill_timeout_seconds=0.2)

    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        self.transport.start()

    async def unload(self, cuda_empty_cache: bool) -> None:
        self.transport.close(force=True)

    async def abort(self) -> None:
        self.transport.close(force=True)

    async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str,
                         context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
        return result_from_payload(self.transport.request("transcribe", timeout_seconds=1))


async def test_http_background_job_and_worker_failure_keep_one_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_TEMP_DIR", str(tmp_path / "temporary"))
    definition = replace(default_models()["qwen3-asr-0.6b"], id="test-rpc", default=True,
                         engine_id="test-rpc", pipelines=("legacy",))
    engine = EngineRegistration("test-rpc", lambda model, settings: RpcFailureAdapter(), "test:rpc")
    app = create_app(Settings(adapter="qwen", workbench_enabled=False, min_free_disk_mb=0,
                              workbench_data_dir=tmp_path / "workbench", log_dir=tmp_path / "logs"),
                     models={definition.id: definition}, engines={engine.engine_id: engine})
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            audio = Path(__file__).resolve().parents[1] / "test-fixtures" / "audio" / "test_short.wav"
            created = await client.post("/v1/audio/transcription-jobs", files={"file": ("input.wav", audio.read_bytes())},
                                        data={"context": "private_hotword_should_not_be_logged"})
            assert created.status_code == 202, created.text
            job_id = created.json()["id"]
            request_id = created.headers["x-request-id"]
            job = {}
            for _ in range(100):
                job = (await client.get(f"/v1/jobs/{job_id}")).json()
                if job["status"] in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.01)
            assert job["status"] == "failed", job
            assert job["error"]["code"] == "inference_failed"
    trace = trace_job(job_id, tmp_path / "logs")
    timeline = trace["timeline"]
    assert isinstance(timeline, list)
    required = {"request.completed", "job.queued", "job.started", "worker.rpc_failed", "job.failed"}
    assert required.issubset({event["event"] for event in timeline})
    related = [event for event in timeline if event["event"] in required]
    assert len({event["trace_id"] for event in related}) == 1
    assert {event["request_id"] for event in related} == {request_id}
    assert {event["job_id"] for event in related if "job_id" in event} == {job_id}
    rpc = next(event for event in related if event["event"] == "worker.rpc_failed")
    assert rpc["rpc_request_id"] == 2 and rpc["worker_generation"] == 1 and rpc["error_chain"]
    assert "private_hotword" not in (tmp_path / "logs" / "events.jsonl").read_text()


def test_background_worker_failure_trace_and_content_free_bundle(tmp_path: Path) -> None:
    runtime = configure_logging(log_dir=tmp_path / "logs", level="DEBUG")
    context = ExecutionContext.new(job_id="job_test", model_id="test-model", engine_id="test")
    transport = ProcessRpcTransport(name="test", target=diagnostics_worker, target_args=(),
                                    terminate_timeout_seconds=0.2, kill_timeout_seconds=0.2)
    try:
        with bind_context(context):
            emit("request.completed", route="/v1/audio/transcription-jobs", status_code=202,
                 text="do not log this", authorization="secret", path="/private/input.wav")
        # Explicit rebind models what a background FIFO owner must do.
        with bind_context(ExecutionContext.from_wire(context.to_wire())):
            emit("job.started")
            transport.start()
            with pytest.raises(AsrError) as failure:
                transport.request("transcribe", timeout_seconds=1)
            assert "error_chain" not in failure.value.details
            emit("job.failed", level=logging.ERROR, error_code=failure.value.code)
    finally:
        transport.close(force=True)
        runtime.close()
    log = (tmp_path / "logs" / "events.jsonl").read_text()
    assert "private transcript" not in log and "/private" not in log and "secret" not in log
    timeline = trace_job("job_test", tmp_path / "logs")
    events = timeline["timeline"]
    assert isinstance(events, list)
    assert {event["trace_id"] for event in events} == {context.trace_id}
    assert {event["request_id"] for event in events} == {context.request_id}
    failure_event = next(event for event in events if event["event"] == "worker.rpc_failed")
    assert failure_event["rpc_request_id"] == 2 and failure_event["worker_generation"] == 1
    assert failure_event["error_type"] == "ValueError"
    assert failure_event["error_chain"][0]["frames"][-1]["function"] == "diagnostics_worker"
    assert all(len(line.encode()) <= MAX_RECORD_BYTES for line in log.splitlines())
    data = tmp_path / "data"
    data.mkdir()
    with sqlite3.connect(data / "workbench.sqlite3") as db:
        db.execute("CREATE TABLE jobs (id TEXT, payload TEXT)")
        db.execute("INSERT INTO jobs VALUES (?,?)", ("job_test", json.dumps({"id": "job_test", "status": "failed", "result": {"text": "private transcript"}, "password_hash": "secret"})))
    output = tmp_path / "diagnostics.tar.gz"
    create_bundle("job_test", tmp_path / "logs", data, output)
    with tarfile.open(output) as archive:
        bundle = b"".join(stream.read() for member in archive.getmembers() if (stream := archive.extractfile(member)) is not None)
    assert b"private transcript" not in bundle and b"password_hash" not in bundle and b"secret" not in bundle
    with pytest.raises(ValueError, match="size limit"):
        create_bundle("job_test", tmp_path / "logs", data, tmp_path / "too-small.tar.gz", max_bytes=10)
    assert not (tmp_path / "too-small.tar.gz").exists()


def test_log_queue_is_bounded_reserves_failures_and_counts_loss() -> None:
    pending: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=8)
    handler = BoundedQueueHandler(pending, reserve=2)
    for _ in range(100):
        record = logging.LogRecord("asr.events", logging.DEBUG, "", 0, "", (), None)
        record.asr_event = event_record("worker.rpc_completed")
        handler.handle(record)
    assert pending.qsize() == 6 and handler.dropped_count == 94
    failure = logging.LogRecord("asr.events", logging.ERROR, "", 0, "", (), None)
    failure.asr_event = event_record("job.failed", error_code="inference_failed")
    handler.handle(failure)
    assert pending.qsize() == 7
    for _ in range(10):
        handler.handle(failure)
    assert pending.qsize() == 8 and handler.dropped_count == 103
    assert getattr(failure, "asr_event")["log_queue_dropped_total"] >= 94


def test_log_retention_expires_idle_files_and_rotates_by_utf8_bytes(tmp_path: Path) -> None:
    clock = [2_000_000_000.0]
    handler = _PrivateRotatingFileHandler(tmp_path / "events.jsonl", maxBytes=1024, backupCount=9,
                                          encoding="utf-8", clock=lambda: clock[0])
    handler.setFormatter(JsonFormatter())
    try:
        for _ in range(30):
            record = logging.LogRecord("asr.events", logging.INFO, "", 0, "", (), None)
            record.asr_event = {"ts": datetime.fromtimestamp(clock[0], timezone.utc).isoformat(), "event": "chunk.completed", "diagnostic": "中文" * 60}
            handler.handle(record)
        files = list(tmp_path.glob("events.jsonl*"))
        assert len(files) <= 10 and sum(path.stat().st_size for path in files) <= 10 * 1024
        assert all(path.stat().st_size <= 1024 and path.stat().st_mode & 0o777 == 0o600 for path in files)
        clock[0] += 86401
        handler.maintain()
        assert (tmp_path / "events.jsonl").stat().st_size == 0
        clock[0] += 7 * 86400
        handler.maintain()  # Same hook is invoked by the listener on idle ticks.
        assert not list(tmp_path.glob("events.jsonl.*"))
    finally:
        handler.close()


def test_recovery_events_preserve_bounded_diagnostics_without_content() -> None:
    event = event_record('quality.recovery_attempt', recovery_policy_version='recovery-v1', attempt_id='attempt_abc',
        call_key='chunk:4:language', requested_language='auto', effective_language='zh',
        source_chunk_indices=[1, 2, 3, 4], model_language_labels=['Chinese'] * 4,
        failures=3, cooldown_seconds=60, review_audio_seconds=180., budget_before_seconds=120.,
        budget_after_seconds=180., next_attempt_at='2026-10-26T08:00:00+00:00', accepted=True,
        text='private transcript', prompt='private prompt', audio='/private/source.wav',
        context='private context', hotwords=['private term'], credentials='private token')
    assert event['source_chunk_indices'] == [1, 2, 3]
    assert event['model_language_labels'] == ['Chinese'] * 3
    assert event['recovery_policy_version'] == 'recovery-v1' and event['attempt_id'] == 'attempt_abc'
    assert event['effective_language'] == 'zh' and event['accepted'] is True
    assert event['failures'] == 3 and event['cooldown_seconds'] == 60
    assert event['budget_before_seconds'] == 120. and event['budget_after_seconds'] == 180.
    assert event['next_attempt_at'] == '2026-10-26T08:00:00+00:00'
    assert 'private' not in json.dumps(event)


def test_recovery_events_drop_untrusted_labels_invalid_identifiers_and_numbers() -> None:
    event = event_record('quality.recovery_attempt', attempt_id='/private/path',
        requested_language='private transcript', effective_language='private_token',
        source_chunk_indices=[-1, 2, 10**30], model_language_labels=['Chinese', 'private transcript', 'English'],
        failures=10**1000, budget_after_seconds=float('nan'), cooldown_seconds=float('inf'),
        next_attempt_at='private transcript', language_evidence={'text': 'private transcript'})
    assert not {'attempt_id', 'requested_language', 'effective_language', 'source_chunk_indices',
                'model_language_labels', 'failures', 'budget_after_seconds', 'cooldown_seconds',
                'next_attempt_at', 'language_evidence'} & event.keys()
