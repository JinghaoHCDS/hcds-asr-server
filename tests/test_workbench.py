from __future__ import annotations

import asyncio
import io
import json
import shutil
import sqlite3
import threading
import wave
from dataclasses import replace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterator
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from asr_server.audio.preprocess import NormalizedAudioPath
from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.main import create_app
from asr_server.workbench import service as service_module
from asr_server.workbench.security import COOKIE_NAME, verify_password
from asr_server.workbench.service import WorkbenchService
from asr_server.workbench.store import WorkbenchStore


def audio_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 32000)
    return output.getvalue()


@pytest.fixture
async def workbench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[FastAPI, WorkbenchService, AsyncClient]]:
    settings = replace(Settings(), workbench_data_dir=tmp_path / "workbench", min_free_disk_mb=0,
                       enable_qwen_alignment=True)
    app = create_app(settings)
    service: WorkbenchService = app.state.workbench
    service.store.create_user("admin", "correct-password", True)
    service.store.create_user("alice", "alice-password")
    service.store.create_user("bob", "bob-password")
    async def normalize(source: Path, output: Path, **kwargs: Any) -> NormalizedAudioPath:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        return NormalizedAudioPath(output, 0)
    async def probe(*args: Any, **kwargs: Any) -> float:
        return 2.0
    async def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        await kwargs["stage_callback"]("transcribing", {"completed_chunks": 1, "total_chunks": 1})
        return {"text": "你好世界", "raw_text": "你好世界", "duration": 2.0,
                "timestamps": [{"text": "你好世界", "start": 0.2, "end": 1.8, "source": "qwen3_forced_aligner"}],
                "segments": [{"text": "你好世界", "raw_text": "你好世界", "start": 0.2, "end": 1.8,
                              "timestamps": [{"text": "你好世界", "start": 0.2, "end": 1.8}]}],
                "quality": {"issues": [], "changes": []}}
    monkeypatch.setattr(service_module, "normalize_audio_path_to_wav_async", normalize)
    monkeypatch.setattr(service_module, "probe_audio_duration_path_async", probe)
    monkeypatch.setattr(service_module, "run_transcription_path", run)
    await app.state.job_manager.start()
    await service.start()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        yield app, service, client
    await service.shutdown()
    await app.state.job_manager.shutdown()
    await app.state.manager.shutdown()
    service.close()


async def login(client: AsyncClient, username: str = "alice", password: str = "alice-password") -> None:
    response = await client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"]


async def submit(client: AsyncClient, *, key: str = "sample", content: bytes | None = None) -> str:
    content = content or audio_bytes()
    upload = (await client.post("/api/uploads", json={"filename": "sample.wav", "size_bytes": len(content)})).json()
    response = await client.patch(f"/api/uploads/{upload['id']}", content=content, headers={"Upload-Offset": "0"})
    assert response.status_code == 200
    response = await client.post(f"/api/uploads/{upload['id']}/complete", json={"domain": "ai", "idempotency_key": key})
    assert response.status_code == 202, response.text
    return str(response.json()["id"])


async def completed(client: AsyncClient, job_id: str) -> dict[str, Any]:
    for _ in range(100):
        response = await client.get(f"/api/jobs/{job_id}")
        job: dict[str, Any] = response.json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            assert job["status"] == "completed", job
            return job
        await asyncio.sleep(0.01)
    pytest.fail("workbench job did not finish")


async def test_session_isolated_from_bearer_and_cross_origin(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    _app, _service, client = workbench
    assert (await client.get("/api/auth/me", headers={"Authorization": "Bearer pretend-legacy-token"})).status_code == 401
    cross = await client.post("/api/auth/login", json={"username": "alice", "password": "alice-password"}, headers={"Origin": "http://another-host"})
    assert cross.status_code == 403
    await login(client)
    assert (await client.get("/api/auth/me")).json()["user"]["username"] == "alice"
    assert (await client.get("/api/admin/users")).status_code == 403
    await client.post("/api/auth/logout")
    assert (await client.get("/api/auth/me")).status_code == 401


async def test_upload_resume_replay_and_owner_guard(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    _app, _service, client = workbench
    await login(client)
    upload = (await client.post("/api/uploads", json={"filename": "record.wav", "size_bytes": 8})).json()
    path = f"/api/uploads/{upload['id']}"
    assert (await client.patch(path, content=b"abcd", headers={"Upload-Offset": "0"})).json()["offset"] == 4
    assert (await client.patch(path, content=b"abcd", headers={"Upload-Offset": "0"})).json()["offset"] == 4
    assert (await client.patch(path, content=b"XXXX", headers={"Upload-Offset": "0"})).status_code == 409
    assert (await client.patch(path, content=b"efgh", headers={"Upload-Offset": "5"})).status_code == 400
    await login(client, "bob", "bob-password")
    assert (await client.get(path)).status_code == 404
    assert (await client.patch(path, content=b"efgh", headers={"Upload-Offset": "4"})).status_code == 404
    await login(client)
    assert (await client.patch(path, content=b"efgh", headers={"Upload-Offset": "4"})).json()["offset"] == 8


async def test_completed_task_audio_exports_and_user_isolation(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    _app, service, client = workbench
    await login(client)
    job_id = await submit(client)
    job = await completed(client, job_id)
    assert job["result"]["version"] == 1
    assert job["expires_at"] > job["completed_at"]
    assert not (service.job_dir(job_id) / "source").exists()
    audio = await client.get(f"/api/jobs/{job_id}/audio", headers={"Range": "bytes=0-43"})
    assert audio.status_code == 206
    assert len(audio.content) == 44
    for format_ in ("txt", "json", "srt", "vtt"):
        export = await client.get(f"/api/jobs/{job_id}/export?format={format_}&version=1")
        assert export.status_code == 200, export.text
        assert export.headers["x-transcript-version"] == "1"
        assert "你好世界" in export.text
    await login(client, "bob", "bob-password")
    assert (await client.get("/api/jobs")).json()["jobs"] == []
    for suffix in ("", "/audio", "/export"):
        assert (await client.get(f"/api/jobs/{job_id}{suffix}")).status_code == 404
    assert (await client.delete(f"/api/jobs/{job_id}")).status_code == 404


async def test_idempotency_matches_content_and_options(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    _app, _service, client = workbench
    await login(client)
    first = await submit(client, key="same")
    await completed(client, first)
    assert await submit(client, key="same") == first
    content = audio_bytes() + b"extra"
    upload = (await client.post("/api/uploads", json={"filename": "different.wav", "size_bytes": len(content)})).json()
    await client.patch(f"/api/uploads/{upload['id']}", content=content, headers={"Upload-Offset": "0"})
    result = await client.post(f"/api/uploads/{upload['id']}/complete", json={"domain": "ai", "idempotency_key": "same"})
    assert result.status_code == 409


async def test_glossary_snapshot_survives_admin_changes(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    _app, service, client = workbench
    await login(client)
    task_id = await submit(client)
    await completed(client, task_id)
    original = service.store.job(task_id)["request"]["glossary_snapshot"]
    await login(client, "admin", "correct-password")
    result = await client.put("/api/admin/glossary", json={"entries": [{"canonical": "新词", "aliases": [], "domain": "ai", "auto_normalize": False}]})
    assert result.status_code == 200
    assert result.json()["version"] == original["version"] + 1
    assert service.store.job(task_id)["request"]["glossary_snapshot"] == original


async def test_edit_blocks_stale_subtitle_and_cas(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    app, _service, client = workbench
    await login(client)
    job_id = await submit(client)
    job = await completed(client, job_id)
    segment = job["result"]["segments"][0]
    gate = asyncio.Event()
    async def align(*args: Any, **kwargs: Any) -> list[Any]:
        await gate.wait()
        from asr_server.adapters.base import TimestampItem
        return [TimestampItem(text=kwargs["text"], start=0.0, end=1.0, granularity="word")]
    monkeypatch.setattr(app.state.manager, "align_text", align)
    response = await client.patch(f"/api/jobs/{job_id}/segments/{segment['id']}", json={"text": "你好，世界", "base_version": 1})
    assert response.status_code == 200
    assert response.json()["result"]["alignment_status"] == "pending"
    assert response.json()["result"]["timestamps"] == []
    assert (await client.get(f"/api/jobs/{job_id}/export?format=srt")).status_code == 409
    assert (await client.get(f"/api/jobs/{job_id}/export?format=srt&version=1")).status_code == 200
    conflict = await client.patch(f"/api/jobs/{job_id}/segments/{segment['id']}", json={"text": "旧版本", "base_version": 1})
    assert conflict.status_code == 409
    gate.set()
    for _ in range(100):
        task = (await client.get(f"/api/jobs/{job_id}")).json()
        if task["result"]["alignment_status"] == "aligned":
            break
        await asyncio.sleep(0.01)
    assert (await client.get(f"/api/jobs/{job_id}/export?format=srt&version=2")).status_code == 200


async def test_quota_reservation_release_and_expiry(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    _app, service, client = workbench
    service.settings = replace(service.settings, workbench_user_quota_bytes=10)
    await login(client)
    upload = await client.post("/api/uploads", json={"filename": "first.wav", "size_bytes": 8})
    assert upload.status_code == 201
    assert (await client.post("/api/uploads", json={"filename": "second.wav", "size_bytes": 8})).status_code == 429
    assert (await client.delete(f"/api/uploads/{upload.json()['id']}")).status_code == 204
    assert (await client.post("/api/uploads", json={"filename": "second.wav", "size_bytes": 8})).status_code == 201
    with service.store.connection() as db:
        db.execute("UPDATE uploads SET expires_at='2000-01-01'")
    await service.cleanup()
    assert service.store.usage() == 0


async def test_admin_password_reset_revokes_sessions(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    app, _service, client = workbench
    await login(client)
    alice_cookie = client.cookies.get(COOKIE_NAME)
    assert alice_cookie is not None
    await login(client, "admin", "correct-password")
    users = (await client.get("/api/admin/users")).json()["users"]
    alice_id = next(user["id"] for user in users if user["username"] == "alice")
    assert (await client.patch(f"/api/admin/users/{alice_id}", json={"password": "new-password"})).status_code == 200
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver", cookies={COOKIE_NAME: alice_cookie}) as old:
        assert (await old.get("/api/auth/me")).status_code == 401


async def test_summary_etag_and_admin_queue_do_not_return_body(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    _app, service, client = workbench
    await login(client)
    job_id = await submit(client)
    await completed(client, job_id)
    summary = (await client.get("/api/jobs?view=summary&limit=1")).json()
    assert summary["jobs"][0]["id"] == job_id
    assert summary["next_cursor"] is None
    assert "result" not in summary["jobs"][0]
    assert "text" not in summary["jobs"][0]["progress"]
    detail = await client.get(f"/api/jobs/{job_id}")
    assert detail.json()["result"]["text"] == "你好世界"
    etag = detail.headers["etag"]
    unchanged = await client.get(f"/api/jobs/{job_id}", headers={"If-None-Match": etag})
    assert unchanged.status_code == 304 and not unchanged.content
    await login(client, "bob", "bob-password")
    assert not (await client.get(f"/api/jobs?view=summary&ids={job_id}")).json()["jobs"]
    assert (await client.get(f"/api/jobs/{job_id}", headers={"If-None-Match": etag})).status_code == 404

    saved = service.store.job(job_id)
    saved["status"] = "queued"
    service.store.save_job(saved)
    await login(client, "admin", "correct-password")
    queue = (await client.get("/api/admin/status")).json()["queue"]
    assert queue[0]["id"] == job_id
    assert not {"filename", "progress", "request", "result", "error"}.intersection(queue[0])
    assert (await client.get(f"/api/jobs/{job_id}", headers={"If-None-Match": etag})).status_code == 404


async def test_prefetch_cancel_releases_preparation_and_preserves_fifo(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    app, service, client = workbench
    app.state.job_manager.scheduler.prefetch = True
    inference_entered, inference_release, prepare_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_normalize = getattr(service_module, "normalize_audio_path_to_wav_async")
    original_run = getattr(service_module, "run_transcription_path")
    normalizations = 0
    calls: list[str] = []
    async def normalize(source: Path, output: Path, **kwargs: Any) -> NormalizedAudioPath:
        nonlocal normalizations
        normalizations += 1
        if normalizations == 2:
            output.write_bytes(b"partial decoder output")
            prepare_entered.set()
            await asyncio.Event().wait()
        normalized: NormalizedAudioPath = await original_normalize(source, output, **kwargs)
        return normalized
    async def infer(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(str(kwargs["normalized_asset"].owner))
        if len(calls) == 1:
            inference_entered.set()
            await inference_release.wait()
        result: dict[str, Any] = await original_run(*args, **kwargs)
        return result
    monkeypatch.setattr(service_module, "normalize_audio_path_to_wav_async", normalize)
    monkeypatch.setattr(service_module, "run_transcription_path", infer)
    await login(client)
    first = await submit(client, key="prefetch-first")
    await asyncio.wait_for(inference_entered.wait(), 2)
    second = await submit(client, key="prefetch-cancel")
    await asyncio.wait_for(prepare_entered.wait(), 2)
    cancelled = await client.post(f"/api/jobs/{second}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    assert (service.job_dir(second) / "source").exists()
    assert not (service.job_dir(second) / "audio.partial.wav").exists()
    assert app.state.workspace_manager._reserved_upload_bytes == 0
    third = await submit(client, key="prefetch-third")
    inference_release.set()
    await completed(client, first)
    await completed(client, third)
    assert calls == [first, third]
    assert normalizations == 3


async def test_workbench_and_temporary_disk_reservations_share_one_budget(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    app, service, client = workbench
    free = [1000]
    monkeypatch.setattr("asr_server.audio.workspace.shutil.disk_usage", lambda _path: SimpleNamespace(free=free[0]))
    manager = app.state.workspace_manager
    owner = service.store.user_for_login("alice")
    assert owner is not None
    upload = await service.create_upload(owner["id"], "reserved.wav", 700)
    assert manager.pending_disk_bytes(service.store.root) == 700
    transient = await manager.reserve_resource(300, service.store.root)
    with pytest.raises(AsrError) as busy:
        await service.create_upload(owner["id"], "cannot-fit.wav", 1)
    assert busy.value.code == "storage_unavailable"
    await transient.release()

    original = service.store.connection
    @contextmanager
    def reject_commit() -> Iterator[sqlite3.Connection]:
        with original() as db:
            db.set_authorizer(lambda operation, table, _column, _database, _source:
                sqlite3.SQLITE_DENY if operation == sqlite3.SQLITE_INSERT and table == "uploads" else sqlite3.SQLITE_OK)
            yield db
    existing = set((service.store.root / "uploads").iterdir())
    monkeypatch.setattr(service.store, "connection", reject_commit)
    with pytest.raises(sqlite3.DatabaseError):
        await service.create_upload(owner["id"], "commit-fails.wav", 100)
    monkeypatch.setattr(service.store, "connection", original)
    assert set((service.store.root / "uploads").iterdir()) == existing
    assert manager.pending_disk_bytes(service.store.root) == 700

    free[0] = 900
    await service.append_upload(owner["id"], upload["id"], 0, b"a" * 100)
    assert service.store.future_disk_reservations() == {"upload:" + upload["id"]: 600}
    assert manager.pending_disk_bytes(service.store.root) == 600
    with pytest.raises(AsrError):
        await manager.reserve_resource(301, service.store.root)
    await login(client)
    assert (await client.delete(f"/api/uploads/{upload['id']}")).status_code == 204
    assert not service.store.future_disk_reservations()
    assert manager.pending_disk_bytes(service.store.root) == 0


def test_store_reopens_with_hashed_credentials(tmp_path: Path) -> None:
    root = tmp_path / "data"
    first = WorkbenchStore(root)
    user = first.create_user("owner", "keep-secret", True)
    second = WorkbenchStore(root)
    with second.connection() as db:
        row = db.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
    assert row is not None
    assert "keep-secret" not in row["password_hash"]
    assert verify_password("keep-secret", row["password_hash"])
    assert not verify_password("wrong", row["password_hash"])


async def test_upload_recovers_uncommitted_tail_and_complete_options(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    _app, service, client = workbench
    await login(client)
    data = audio_bytes()
    upload = (await client.post("/api/uploads", json={"filename": "sample.wav", "size_bytes": len(data)})).json()
    path = f"/api/uploads/{upload['id']}"
    await client.patch(path, content=data[:100], headers={"Upload-Offset": "0"})
    with service.upload_path(upload["id"]).open("ab") as target:
        target.write(b"uncommitted-crash-tail")
    assert (await client.get(path)).json()["offset"] == 100
    response = await client.patch(path, content=data[100:], headers={"Upload-Offset": "100"})
    assert response.status_code == 200
    assert service.upload_path(upload["id"]).read_bytes() == data
    first = await client.post(path + "/complete", json={"domain": "ai", "idempotency_key": "replayed"})
    assert first.status_code == 202
    same = await client.post(path + "/complete", json={"domain": "ai", "idempotency_key": "replayed"})
    assert same.json()["id"] == first.json()["id"]
    changed = await client.post(path + "/complete", json={"domain": "general", "idempotency_key": "replayed"})
    assert changed.status_code == 409


async def test_delete_waits_for_active_alignment_without_resurrection(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    app, service, client = workbench
    await login(client)
    job_id = await submit(client)
    job = await completed(client, job_id)
    entered, release = asyncio.Event(), asyncio.Event()
    async def align(*args: Any, **kwargs: Any) -> list[Any]:
        entered.set()
        await release.wait()
        return []
    monkeypatch.setattr(app.state.manager, "align_text", align)
    response = await client.patch(f"/api/jobs/{job_id}/segments/segment-0", json={"text": "改字", "base_version": 1})
    assert response.status_code == 200
    await asyncio.wait_for(entered.wait(), 1)
    assert (await client.delete(f"/api/jobs/{job_id}")).status_code == 202
    assert (service.job_dir(job_id) / "audio.wav").exists()
    release.set()
    for _ in range(100):
        if (await client.get(f"/api/jobs/{job_id}")).status_code == 404:
            break
        await asyncio.sleep(0.01)
    assert (await client.get(f"/api/jobs/{job_id}")).status_code == 404
    assert not service.job_dir(job_id).exists()
    assert job["result"]["version"] == 1


async def test_pending_alignment_survives_full_queue(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    app, service, client = workbench
    await login(client)
    job_id = await submit(client)
    await completed(client, job_id)
    enqueue = app.state.job_manager.enqueue_external
    async def full(key: str, operation: Any, **kwargs: Any) -> None:
        raise AsrError(429, "job_queue_full", "full")
    monkeypatch.setattr(app.state.job_manager, "enqueue_external", full)
    response = await client.patch(f"/api/jobs/{job_id}/segments/segment-0", json={"text": "等待对齐", "base_version": 1})
    assert response.status_code == 200
    assert response.json()["result"]["alignment_status"] == "pending"
    monkeypatch.setattr(app.state.job_manager, "enqueue_external", enqueue)
    async def align(*args: Any, **kwargs: Any) -> list[Any]:
        from asr_server.adapters.base import TimestampItem
        return [TimestampItem(text="等待对齐", start=0.0, end=1.0, granularity="word")]
    monkeypatch.setattr(app.state.manager, "align_text", align)
    await service.fill_queue()
    for _ in range(100):
        job = (await client.get(f"/api/jobs/{job_id}")).json()
        if job["result"]["alignment_status"] == "aligned":
            break
        await asyncio.sleep(0.01)
    assert job["result"]["alignment_status"] == "aligned"


async def test_single_process_lock_and_task_recovery(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    app, service, client = workbench
    other = WorkbenchService(app, service.settings)
    with pytest.raises(AsrError) as caught:
        await other.start()
    assert caught.value.code == "data_directory_in_use"
    await login(client)
    job_id = await submit(client)
    await completed(client, job_id)
    saved = service.store.job(job_id)
    saved["status"] = "transcribing"
    saved["completed_at"] = None
    saved["expires_at"] = None
    service.store.save_job(saved)
    await service.shutdown()
    await app.state.job_manager.shutdown()
    service.close()
    # A new app and SQLite connection restore the same task identity and audio.
    restarted = create_app(service.settings)
    replacement: WorkbenchService = restarted.state.workbench
    await restarted.state.job_manager.start()
    await replacement.start()
    try:
        async with AsyncClient(transport=ASGITransport(app=restarted), base_url="http://testserver") as fresh:
            await login(fresh)
            recovered = await completed(fresh, job_id)
            assert recovered["id"] == job_id
            assert recovered["result"]["text"] == "你好世界"
    finally:
        await replacement.shutdown()
        await restarted.state.job_manager.shutdown()
        await restarted.state.manager.shutdown()
        replacement.close()


@pytest.mark.parametrize("interrupt", ["callback", "cancelled_error"])
async def test_user_cancel_wins_over_simultaneous_shutdown(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch, interrupt: str) -> None:
    app, service, client = workbench
    entered, release = asyncio.Event(), asyncio.Event()
    invocations: list[str] = []
    async def running(*args: Any, **kwargs: Any) -> dict[str, Any]:
        invocations.append("run")
        entered.set()
        await release.wait()
        if interrupt == "cancelled_error":
            raise asyncio.CancelledError
        await kwargs["before_chunk"](0, 1)
        raise AssertionError("cancelled task must stop at its callback")
    monkeypatch.setattr(service_module, "run_transcription_path", running)
    await login(client)
    job_id = await submit(client)
    await asyncio.wait_for(entered.wait(), 1)
    assert (await client.post(f"/api/jobs/{job_id}/cancel")).json()["status"] == "cancel_requested"
    await service.shutdown()
    release.set()
    for _ in range(100):
        if service.store.job(job_id)["status"] == "cancelled":
            break
        await asyncio.sleep(0.01)
    assert service.store.job(job_id)["status"] == "cancelled"
    await app.state.job_manager.shutdown()
    service.close()
    recovered = WorkbenchService(app, service.settings)
    await recovered.start()
    try:
        assert recovered.store.job(job_id)["status"] == "cancelled"
        assert invocations == ["run"]
    finally:
        await recovered.shutdown()
        recovered.close()


async def test_delete_intent_survives_cancellation_cleanup_await(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    _app, service, client = workbench
    running_entered, running_release = asyncio.Event(), asyncio.Event()
    cleanup_entered, cleanup_release = threading.Event(), threading.Event()
    original_rmtree = shutil.rmtree
    blocked = False
    async def running(*args: Any, **kwargs: Any) -> dict[str, Any]:
        running_entered.set()
        await running_release.wait()
        await kwargs["before_chunk"](0, 1)
        raise AssertionError("cancelled task should not complete")
    monkeypatch.setattr(service_module, "run_transcription_path", running)
    await login(client)
    job_id = await submit(client)
    await asyncio.wait_for(running_entered.wait(), 1)
    def gated_rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal blocked
        if Path(path) == service.job_dir(job_id) / "work" and not blocked:
            blocked = True
            cleanup_entered.set()
            assert cleanup_release.wait(3), "test cleanup gate was not released"
        original_rmtree(path, *args, **kwargs)
    monkeypatch.setattr(shutil, "rmtree", gated_rmtree)
    await client.post(f"/api/jobs/{job_id}/cancel")
    running_release.set()
    assert await asyncio.to_thread(cleanup_entered.wait, 2)
    try:
        assert (await client.delete(f"/api/jobs/{job_id}")).status_code == 202
        assert service.store.job(job_id)["delete_requested"] is True
    finally:
        cleanup_release.set()
    for _ in range(100):
        if (await client.get(f"/api/jobs/{job_id}")).status_code == 404:
            break
        await asyncio.sleep(0.01)
    assert (await client.get(f"/api/jobs/{job_id}")).status_code == 404
    assert not service.job_dir(job_id).exists()


async def test_delete_serializes_with_task_leaving_fifo(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    app, service, client = workbench
    hold_entered, hold_release, starting = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def hold() -> None:
        hold_entered.set()
        await hold_release.wait()
    await app.state.job_manager.enqueue_external("hold-for-delete-race", hold)
    await asyncio.wait_for(hold_entered.wait(), 1)
    await login(client)
    job_id = await submit(client)
    original_run = service.run_job
    async def starting_run(key: str) -> None:
        starting.set()
        await original_run(key)
    monkeypatch.setattr(service, "run_job", starting_run)
    original_cancel = app.state.job_manager.cancel_external
    async def racing_cancel(key: str) -> bool:
        if key == job_id:
            hold_release.set()
            await asyncio.wait_for(starting.wait(), 1)
            return False  # Worker has removed it from FIFO, awaiting start lock.
        return bool(await original_cancel(key))
    monkeypatch.setattr(app.state.job_manager, "cancel_external", racing_cancel)
    response = await client.delete(f"/api/jobs/{job_id}")
    assert response.status_code == 204
    await asyncio.sleep(0)
    assert (await client.get(f"/api/jobs/{job_id}")).status_code == 404
    assert not service.job_dir(job_id).exists()
    assert job_id not in service.active


async def test_request_hotwords_inherit_approved_domain_aliases(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    _app, service, client = workbench
    await login(client, "admin", "correct-password")
    response = await client.put("/api/admin/glossary", json={"entries": [
        {"canonical": "Qwen3-ASR", "aliases": ["千问 ASR"], "domain": "ai", "auto_normalize": True},
        {"canonical": "Qwen3-ASR", "aliases": [], "domain": "general", "auto_normalize": False},
    ]})
    assert response.status_code == 200
    snapshot = service.snapshot("ai", "自定义人名,qwen3-asr")
    assert snapshot["entries"][0]["canonical"] == "自定义人名"
    qwen = snapshot["entries"][1]
    assert qwen["canonical"] == "Qwen3-ASR"
    assert qwen["aliases"] == ["千问 ASR"]
    assert qwen["auto_normalize"] is True
    assert len(snapshot["entries"]) == 2


async def test_delete_intent_survives_failed_job_cleanup_await(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    _app, service, client = workbench
    running_entered, running_release = asyncio.Event(), asyncio.Event()
    cleanup_entered, cleanup_release = threading.Event(), threading.Event()
    original_rmtree = shutil.rmtree
    blocked = False
    async def failing(*args: Any, **kwargs: Any) -> dict[str, Any]:
        running_entered.set()
        await running_release.wait()
        raise AsrError(422, "generation_truncated", "test error before cleanup")
    monkeypatch.setattr(service_module, "run_transcription_path", failing)
    await login(client)
    job_id = await submit(client)
    await asyncio.wait_for(running_entered.wait(), 1)
    def gated_rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal blocked
        if Path(path) == service.job_dir(job_id) / "work" and not blocked:
            blocked = True
            cleanup_entered.set()
            assert cleanup_release.wait(3)
        original_rmtree(path, *args, **kwargs)
    monkeypatch.setattr(shutil, "rmtree", gated_rmtree)
    running_release.set()
    assert await asyncio.to_thread(cleanup_entered.wait, 2)
    try:
        assert (await client.delete(f"/api/jobs/{job_id}")).status_code == 202
    finally:
        cleanup_release.set()
    for _ in range(100):
        if (await client.get(f"/api/jobs/{job_id}")).status_code == 404:
            break
        await asyncio.sleep(0.01)
    assert (await client.get(f"/api/jobs/{job_id}")).status_code == 404
    assert not service.job_dir(job_id).exists()
