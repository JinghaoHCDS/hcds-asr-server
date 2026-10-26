from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

from asr_server.errors import AsrError
from asr_server.workbench.store import SCHEMA_VERSION, WorkbenchStore, after_days, now
from scripts.workbench_release import ReleaseError, offline_lock
from scripts.workbench_reset_tasks import reset_tasks


def insert_job(store: WorkbenchStore, job: dict[str, Any], reservation: int) -> None:
    with store.connection() as db:
        db.execute("INSERT INTO jobs(id,owner_id,status,reserved_bytes,payload) VALUES(?,?,?,?,?)",
                   (job["id"], job["owner_id"], job["status"], reservation, json.dumps(job)))
    store.save_job(job)


def test_storage_categories_partition_materialized_files_and_unwritten_promises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = WorkbenchStore(tmp_path)
    task: dict[str, Any] = {"id": "job", "owner_id": "alice", "status": "transcribing",
                            "retained_file_bytes": {"source": 100, "audio.mp3": 200,
                                                    "operation/audio.wav": 500, "checkpoints/chunk.json": 100},
                            "audio_asset": {"state": "ready", "codec": "mp3", "size_bytes": 200},
                            "progress": {}, "result": None}
    insert_job(store, task, 1400)
    with store.connection() as db:
        db.execute("INSERT INTO uploads VALUES(?,?,?,?,?,?,?,NULL)", ("upload", "bob", "other.wav", 1000, 250, now(), after_days(1)))
    store.set_physical_usage("upload", "bob", 250)
    alice = store.storage_usage("alice")
    assert alice["saved_bytes"] == 300
    assert alice["temporary_bytes"] == 500
    assert alice["text_bytes"] > 100  # Durable checkpoint plus SQLite metadata.
    assert alice["reserved_bytes"] == 500
    assert alice["allocated_bytes"] == alice["used_bytes"] + 500
    assert store.future_disk_reservations() == {"job:job": 500, "upload:upload": 750}
    bob = store.storage_usage("bob")
    assert bob == {"saved_bytes": 250, "text_bytes": 0, "temporary_bytes": 0, "reserved_bytes": 750,
                   "used_bytes": 250, "allocated_bytes": 1000}
    assert store.usage() == alice["allocated_bytes"] + 1000
    assert store.job_summaries("alice")["jobs"][0]["audio_size_bytes"] == 200
    task["status"] = "completed"
    task["retained_file_bytes"] = {"audio.mp3": 200, "checkpoints/chunk.json": 100}
    store.save_job(task, reserved_bytes=300)
    completed = store.storage_usage("alice")
    assert completed["saved_bytes"] == 200 and completed["temporary_bytes"] == completed["reserved_bytes"] == 0
    assert store.future_disk_reservations() == {"upload:upload": 750}
    original = store.connection

    @contextmanager
    def no_payload_reads() -> Iterator[sqlite3.Connection]:
        with original() as db:
            db.set_authorizer(lambda action, table, column, _database, _source:
                              sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and table == "jobs" and column == "payload" else sqlite3.SQLITE_OK)
            yield db

    monkeypatch.setattr(store, "connection", no_payload_reads)
    assert store.storage_usage("alice") == completed
    monkeypatch.setattr(store, "connection", original)
    store.close()


def test_old_tasks_require_reset_and_offline_reset_preserves_complete_account_and_glossary(tmp_path: Path) -> None:
    store = WorkbenchStore(tmp_path)
    user = store.create_user("owner", "private-password", True)
    store.session("private-session", user["id"])
    store.update_glossary([{"canonical": "研究", "aliases": ["研旧"], "domain": "ai", "auto_normalize": True}], 1)
    task = {"id": "old", "owner_id": user["id"], "status": "completed", "result": {"version": 1, "text": "private-text"}, "retained_file_bytes": {"audio.wav": 10}}
    insert_job(store, task, 10)
    (tmp_path / "jobs/old").mkdir(parents=True)
    (tmp_path / "jobs/old/audio.wav").write_bytes(b"old-audio!")
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads/unsubmitted").write_bytes(b"unsubmitted")
    (tmp_path / "models").mkdir()
    (tmp_path / "models/do-not-touch").write_bytes(b"model-cache")
    with store.connection() as db:
        db.execute("PRAGMA user_version=3")
        before_users = [tuple(row) for row in db.execute("SELECT * FROM users")]
        before_metadata = [tuple(row) for row in db.execute("SELECT * FROM metadata")]
    store.close()
    rejected = WorkbenchStore(tmp_path)
    with pytest.raises(AsrError) as reset_required:
        rejected.initialize()
    assert reset_required.value.code == "schema_reset_required"
    rejected.close()
    preview = reset_tasks(tmp_path)
    assert not preview["applied"] and preview["task_rows"]["jobs"] == 1
    assert preview["file_bytes"] == 21
    assert (tmp_path / "jobs/old/audio.wav").exists()
    result = reset_tasks(tmp_path, apply=True)
    assert result["preserved_identities_match"] and result["schema_after"] == SCHEMA_VERSION == 5
    assert all(count == 0 for count in result["task_rows_after"].values())
    assert not (tmp_path / "jobs").exists() and not (tmp_path / "uploads").exists()
    assert (tmp_path / "models/do-not-touch").read_bytes() == b"model-cache"
    fresh = WorkbenchStore(tmp_path)
    with fresh.connection() as db:
        assert [tuple(row) for row in db.execute("SELECT * FROM users")] == before_users
        assert [tuple(row) for row in db.execute("SELECT * FROM metadata")] == before_metadata
    assert fresh.authenticate_session("private-session")["id"] == user["id"]
    assert fresh.storage_usage()["allocated_bytes"] == 0
    fresh.close()


def test_reset_refuses_live_store_and_symlink_before_deleting(tmp_path: Path) -> None:
    store = WorkbenchStore(tmp_path)
    store.initialize()
    (tmp_path / "jobs").mkdir()
    with offline_lock(tmp_path):
        with pytest.raises(ReleaseError, match="service is active"):
            reset_tasks(tmp_path, apply=True)
    outside = tmp_path.parent / "outside"
    outside.write_bytes(b"preserve")
    (tmp_path / "jobs/escape").symlink_to(outside)
    with pytest.raises(ReleaseError, match="symlink"):
        reset_tasks(tmp_path, apply=True)
    assert outside.read_bytes() == b"preserve" and (tmp_path / "jobs/escape").is_symlink()
    store.close()


def test_empty_schema3_upgrade_preserves_accounts(tmp_path: Path) -> None:
    store = WorkbenchStore(tmp_path)
    user = store.create_user("owner", "private-password")
    with store.connection() as db:
        db.execute("PRAGMA user_version=3")
    store.close()
    updated = WorkbenchStore(tmp_path)
    updated.initialize()
    assert updated.users() == [user]
    with updated.connection() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    updated.close()


def test_failed_file_cleanup_keeps_task_ledger_and_preserved_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = WorkbenchStore(tmp_path)
    store.create_user("owner", "private-password", True)
    task = {"id": "task", "owner_id": "alice", "status": "completed", "retained_file_bytes": {"audio.mp3": 10}}
    insert_job(store, task, 10)
    (tmp_path / "jobs/task").mkdir(parents=True)
    (tmp_path / "jobs/task/audio.mp3").write_bytes(b"saved-audio")
    users, glossary, usage = store.users(), store.glossary(), store.storage_usage()

    def failed_delete(_path: Path) -> None:
        raise PermissionError("simulated deletion failure")

    monkeypatch.setattr("scripts.workbench_reset_tasks.shutil.rmtree", failed_delete)
    with pytest.raises(PermissionError, match="deletion failure"):
        reset_tasks(tmp_path, apply=True)
    assert store.users() == users and store.glossary() == glossary
    assert store.storage_usage() == usage and store.job("task")["status"] == "completed"
    assert (tmp_path / "jobs/task/audio.mp3").read_bytes() == b"saved-audio"
    store.close()
