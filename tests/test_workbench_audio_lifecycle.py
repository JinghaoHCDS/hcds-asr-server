"""Real media + mock GPU contracts across operation lifetimes."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.main import create_app
from asr_server.workbench.service import WorkbenchService
from tests.helpers import make_wav

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")


async def wait_job(service: WorkbenchService, job_id: str) -> dict[str, Any]:
    operation = service.app.state.job_manager.scheduler.operations.get(job_id)
    if operation is not None:
        await asyncio.wait_for(asyncio.shield(operation.future), 10)
    return await service.store.run(service.store.job, job_id)


async def submit(service: WorkbenchService, owner: str, content: bytes) -> str:
    upload = await service.create_upload(owner, "fixture.wav", len(content))
    await service.append_upload(owner, upload["id"], 0, content)
    job = await service.complete_upload(owner, upload["id"], {"quality_profile": "legacy"})
    return str(job["id"])


async def test_mp3_survives_terminal_pcm_cleanup_retry_edit_and_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from asr_server.workbench import service as module
    original = getattr(module, "run_transcription_path")
    attempts = 0
    async def transient(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        assert kwargs["normalized_asset"].path.exists()
        if attempts == 1:
            raise AsrError(503, "gpu_unavailable", "temporary test failure")
        result: dict[str, Any] = await original(*args, **kwargs)
        return result
    monkeypatch.setattr(module, "run_transcription_path", transient)
    app = create_app(Settings(adapter="mock", workbench_data_dir=tmp_path / "data", min_free_disk_mb=0, enable_qwen_alignment=True))
    service: WorkbenchService = app.state.workbench
    user = service.store.create_user("audio-user", "audio-test-password")
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://testserver") as client:
            await client.post("/api/auth/login", json={"username": "audio-user", "password": "audio-test-password"})
            job_id = await submit(service, user["id"], make_wav(2))
            failed = await wait_job(service, job_id)
            root = service.job_dir(job_id)
            assert failed["status"] == "failed" and failed["audio_asset"]["state"] == "ready"
            assert not (root / "source").exists() and not (root / "operation").exists()
            mp3 = (root / "audio.mp3").read_bytes()
            usage = (await client.get("/api/storage")).json()
            assert usage["temporary_bytes"] == usage["reserved_bytes"] == 0
            assert usage["saved_bytes"] == len(mp3)
            await service.retry(user["id"], job_id)
            completed = await wait_job(service, job_id)
            assert completed["status"] == "completed"
            assert (root / "audio.mp3").read_bytes() == mp3
            assert not (root / "operation").exists() and not (root / "work").exists()
            response = await client.get(f"/api/jobs/{job_id}/audio", headers={"Range": "bytes=0-43"})
            assert response.status_code == 206 and response.headers["content-type"] == "audio/mpeg"
            target = completed["result"]["segments"][0]
            await service.edit(user["id"], job_id, target["id"], target["text"] + " edited", 1)
            operation = app.state.job_manager.scheduler.operations.get("align:" + job_id)
            if operation:
                await asyncio.wait_for(asyncio.shield(operation.future), 10)
            edited = await service.store.run(service.store.job, job_id)
            assert edited["result"]["alignment_status"] == "aligned"
            assert not (root / "operation").exists() and (root / "audio.mp3").read_bytes() == mp3
            assert (await client.get("/api/storage")).json()["temporary_bytes"] == 0
            assert await service.delete(user["id"], job_id)
            assert not root.exists()
            assert (await client.get("/api/storage")).json()["allocated_bytes"] == 0


async def test_failed_preparation_retains_source_and_startup_discards_unowned_pcm(tmp_path: Path) -> None:
    settings = Settings(adapter="mock", workbench_data_dir=tmp_path / "data", min_free_disk_mb=0)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        service: WorkbenchService = app.state.workbench
        job_id = await submit(service, "owner", b"invalid media")
        job = await wait_job(service, job_id)
        assert job["status"] == "failed" and job["audio_asset"]["state"] == "failed"
        root = service.job_dir(job_id)
        assert (root / "source").read_bytes() == b"invalid media"
        assert not list(root.rglob("*.wav")) and not list(root.glob("*.partial*"))
        (root / "operation").mkdir()
        (root / "operation/audio.wav").write_bytes(b"orphan")
        (root / "audio.partial.mp3").write_bytes(b"orphan")
    restarted = create_app(settings)
    async with restarted.router.lifespan_context(restarted):
        service = restarted.state.workbench
        assert not (root / "operation").exists() and not (root / "audio.partial.mp3").exists()
        assert (root / "source").exists()
        assert service.store.storage_usage()["temporary_bytes"] == 0


async def test_changed_pcm_identity_refuses_checkpoint_and_allows_explicit_restart(tmp_path: Path) -> None:
    app = create_app(Settings(adapter="mock", workbench_data_dir=tmp_path / "data", min_free_disk_mb=0))
    async with app.router.lifespan_context(app):
        service: WorkbenchService = app.state.workbench
        job_id = await submit(service, "owner", make_wav(2))
        job = await wait_job(service, job_id)
        assert job["status"] == "completed"
        identity = service.job_dir(job_id) / "checkpoints/audio-identity.json"
        identity.write_text('{"retained_sha256":"changed"}')
        job["status"] = "failed"
        service.store.save_job(job)
        await service.retry("owner", job_id)
        mismatch = await wait_job(service, job_id)
        assert mismatch["status"] == "failed" and mismatch["error"]["code"] == "checkpoint_version_mismatch"
        assert not list(service.job_dir(job_id).rglob("*.wav"))
        await service.retry("owner", job_id, from_start=True)
        assert (await wait_job(service, job_id))["status"] == "completed"


async def test_deletion_waits_for_playback_and_failed_removal_retains_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(Settings(adapter="mock", workbench_data_dir=tmp_path / "data", min_free_disk_mb=0))
    async with app.router.lifespan_context(app):
        service: WorkbenchService = app.state.workbench
        job_id = await submit(service, "owner", make_wav(1))
        assert (await wait_job(service, job_id))["status"] == "completed"
        path = await service.acquire_playback("owner", job_id)
        assert not await service.delete("owner", job_id)
        assert path.exists()
        await service.release_playback(job_id)
        assert not path.exists() and service.store.usage() == 0
        job_id = await submit(service, "owner", make_wav(1))
        await wait_job(service, job_id)
        root = service.job_dir(job_id)
        original = shutil.rmtree
        def denied(path: Any, *args: Any, **kwargs: Any) -> None:
            if Path(path) == root:
                raise OSError("test deletion failure")
            original(path, *args, **kwargs)
        monkeypatch.setattr(shutil, "rmtree", denied)
        usage = service.store.usage()
        with pytest.raises(AsrError) as failed:
            await service.delete("owner", job_id)
        assert failed.value.code == "cleanup_failed"
        assert root.exists() and service.store.usage() >= usage
        assert service.store.job(job_id)["delete_requested"]
        monkeypatch.setattr(shutil, "rmtree", original)
        await service.cleanup()
        assert not root.exists() and service.store.usage() == 0


async def test_prefetch_mp3_cancellation_does_not_hold_service_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from asr_server.workbench import service as module
    entered, hold, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def first() -> None:
        hold.set()
        await release.wait()
    original = getattr(module, "prepare_retained_audio")
    async def wait_encoding(*args: Any, **kwargs: Any) -> dict[str, Any]:
        entered.set()
        await asyncio.Event().wait()
        result: dict[str, Any] = await original(*args, **kwargs)
        return result
    monkeypatch.setattr(module, "prepare_retained_audio", wait_encoding)
    app = create_app(Settings(adapter="mock", preprocessing_prefetch=True, workbench_data_dir=tmp_path / "data", min_free_disk_mb=0))
    async with app.router.lifespan_context(app):
        service: WorkbenchService = app.state.workbench
        first_op = await app.state.job_manager.scheduler.submit("hold", first)
        await hold.wait()
        try:
            job_id = await submit(service, "owner", make_wav(1))
            await asyncio.wait_for(entered.wait(), 3)
            await asyncio.wait_for(service.cancel("owner", job_id), 3)
            assert service.store.job(job_id)["status"] == "cancelled"
            assert (service.job_dir(job_id) / "source").exists()
            assert not app.state.workspace_manager.pending_disk_bytes(service.store.root)
        finally:
            release.set()
            await first_op.future
