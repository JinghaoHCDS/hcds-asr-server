from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from asr_server.errors import AsrError
from asr_server.audio.workspace import WorkspaceManager
from asr_server.config import Settings
from asr_server.workbench.executor import BoundedExecutor
from asr_server.workbench.security import LoginFailures
from asr_server.workbench.store import SCHEMA_VERSION, WorkbenchStore


def test_legacy_migration_summary_paging_and_capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"id": "old", "owner_id": "alice", "filename": "sample.wav", "status": "completed",
               "request": {"model": "test", "hotwords": "private"}, "progress": {"phase": "completed", "text": "private"},
               "result": {"text": "private" * 10000, "version": 1, "alignment_status": "aligned"}}
    encoded = json.dumps(payload)
    with sqlite3.connect(tmp_path / "workbench.sqlite3") as db:
        db.execute("CREATE TABLE jobs(sequence INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE,owner_id TEXT,status TEXT,reserved_bytes INTEGER,payload TEXT,idempotency_key TEXT,fingerprint TEXT,UNIQUE(owner_id,idempotency_key))")
        db.execute("INSERT INTO jobs(id,owner_id,status,reserved_bytes,payload) VALUES(?,?,?,?,?)", ("old", "alice", "completed", 100, encoded))
    store = WorkbenchStore(tmp_path)
    store.initialize()
    with store.connection() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert db.execute("SELECT payload FROM legacy_job_payloads WHERE job_id='old'").fetchone()[0] == encoded
        assert "private" * 100 not in db.execute("SELECT payload FROM jobs WHERE id='old'").fetchone()[0]
        for job_id, owner in [("new", "alice"), ("other", "bob")]:
            task = {**payload, "id": job_id, "owner_id": owner}
            db.execute("INSERT INTO jobs(id,owner_id,status,reserved_bytes,payload) VALUES(?,?,?,?,?)", (job_id, owner, "completed", 100, json.dumps(task)))
    first = store.job_summaries("alice", limit=1)
    assert first["jobs"][0]["id"] == "new"
    assert first["next_cursor"] is not None
    assert [task["id"] for task in store.job_summaries("alice", limit=1, before=first["next_cursor"])["jobs"]] == ["old"]
    assert "result" not in first["jobs"][0] and "text" not in first["jobs"][0]["progress"]
    assert first["jobs"][0]["has_result"] == 1
    job = store.job("new", "alice")
    old_revision = job["revision"]
    job["status"] = "failed"
    store.save_job(job)
    assert store.job_revision("new", "alice") > old_revision
    assert store.job_summaries("bob")["jobs"][0]["id"] == "other"
    with store.connection() as db:
        expected = db.execute("SELECT sum(reserved_bytes+length(CAST(payload AS BLOB))) FROM jobs").fetchone()[0]
        expected += sum(db.execute(f"SELECT coalesce(sum(length(CAST(payload AS BLOB))),0) FROM {table}").fetchone()[0] for table in ("versions", "job_chunks", "legacy_job_payloads"))
    assert store.usage() == expected
    store.set_physical_usage("asset", "alice", 90)
    store.set_physical_usage("asset", "alice", 90)
    assert store.physical_usage() == 90
    store.set_physical_usage("asset", "bob", 90)
    assert store.physical_usage("alice") == 0 and store.physical_usage("bob") == 90

    original = store.connection
    @contextmanager
    def summary_only() -> Iterator[sqlite3.Connection]:
        with original() as db:
            db.set_authorizer(lambda operation, table, column, _database, _source:
                sqlite3.SQLITE_DENY if operation == sqlite3.SQLITE_READ and table == "jobs" and column == "payload" else sqlite3.SQLITE_OK)
            yield db
    monkeypatch.setattr(store, "connection", summary_only)
    assert len(store.job_summaries("alice")["jobs"]) == 2
    assert store.usage() > 0
    monkeypatch.setattr(store, "connection", original)
    with store.connection() as db:
        db.execute("DELETE FROM jobs")
    assert store.usage() == 0
    store.close()


async def test_bounded_executor_keeps_running_slot_after_waiter_cancellation() -> None:
    executor = BoundedExecutor(1, 0, "test-repository")
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    def operation() -> None:
        started.set()
        release.wait(5)
        completed.set()
    task = asyncio.create_task(executor.run(operation))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(AsrError) as busy:
            await executor.run(lambda: None)
        assert busy.value.status_code == 429
        release.set()
        assert await asyncio.to_thread(completed.wait, 2)
    finally:
        release.set()
        await asyncio.to_thread(executor.close)


def test_login_failure_table_expires_and_bounds_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr("asr_server.workbench.security.time.monotonic", lambda: clock[0])
    failures = LoginFailures(capacity=2, ttl=60)
    for address in ("first", "second", "third"):
        failures.failed(address)
    assert len(failures.addresses) == 2 and failures.recent("first") == []
    clock[0] = 61
    assert failures.recent("third") == [] and not failures.addresses


def test_progress_and_result_bodies_are_stored_once_and_hydrated_on_detail(tmp_path: Path) -> None:
    store = WorkbenchStore(tmp_path)
    task: dict[str, Any] = {"id": "body-job", "owner_id": "alice", "status": "transcribing", "progress": {}, "result": None}
    with store.connection() as db:
        db.execute("INSERT INTO jobs(id,owner_id,status,reserved_bytes,payload) VALUES(?,?,?,?,?)", (task["id"], "alice", task["status"], 0, json.dumps(task)))
    store.save_chunk(task["id"], {"index": 0, "text": "chunk-body" * 1000}, 1)
    metadata = store.job_metadata(task["id"])
    assert "chunk-body" not in json.dumps(metadata)
    metadata["progress"]["phase"] = "reviewing"
    store.save_job(metadata)
    assert store.job(task["id"])["progress"]["chunks"][0]["text"] == "chunk-body" * 1000
    metadata["result"] = {"version": 1, "text": "result-body" * 1000, "alignment_status": "aligned"}
    metadata["status"] = "completed"
    store.save_version(metadata)
    compact = store.job_metadata(task["id"])
    assert "result-body" not in json.dumps(compact)
    store.save_job(compact)
    assert store.job(task["id"])["result"]["text"] == "result-body" * 1000
    store.save_chunk(task["id"], {"index": 1, "text": "late"}, 2)
    assert store.job(task["id"])["status"] == "completed"
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM versions").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM job_chunks").fetchone()[0] == 1
        db.execute("DELETE FROM jobs")
    assert store.usage() == 0
    store.close()


async def test_shared_disk_promises_handoff_and_materialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    free = [1000]
    monkeypatch.setattr("asr_server.audio.workspace.shutil.disk_usage", lambda _path: SimpleNamespace(free=free[0]))
    manager = WorkspaceManager(Settings(min_free_disk_mb=0))
    device = tmp_path.stat().st_dev
    await manager.set_persistent_reservations(tmp_path, {"upload": 700, "job": 200})
    assert manager._reserved_upload_bytes == 0
    with pytest.raises(AsrError) as no_space:
        await manager.reserve_resource(101, tmp_path)
    assert no_space.value.code == "storage_unavailable"
    transient = await manager.reserve_resource(100, tmp_path)
    with pytest.raises(AsrError):
        await manager.set_persistent_reservations(tmp_path, {"upload": 701, "job": 200})
    assert manager.pending_disk_bytes(tmp_path) == 1000
    await transient.release()
    decoder = await manager.handoff_persistent(tmp_path, "job", 200)
    assert manager.pending_disk_bytes(tmp_path) == 900  # Transfer, never double count.
    await decoder.materialize(150)
    free[0] -= 150  # Existing PCM bytes now appear in filesystem free space.
    assert manager.pending_disk_bytes(tmp_path) == 750
    await manager.set_persistent_reservations(tmp_path, {"upload": 700, "job": 50}, check_growth=False)
    assert manager.pending_disk_bytes(tmp_path) == 750
    with pytest.raises(AsrError):
        await manager.reserve_resource(101, tmp_path)
    await decoder.release_unused()
    await manager.set_persistent_reservations(tmp_path, {"upload": 700}, check_growth=False)
    await decoder.release()
    await decoder.release()
    assert manager.pending_disk_bytes(tmp_path) == 700
    assert not manager._persistent_handoffs[device] and manager._reserved_upload_bytes == 0
    available = await manager.reserve_resource(150, tmp_path)
    await available.release()
