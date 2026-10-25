from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from asr_server.adapters.base import AudioInput, TimestampItem, TranscriptionResult
from asr_server.adapters.mock import MockAsrAdapter
from asr_server.config import Settings
from asr_server.engines import EngineRegistration
from asr_server.main import create_app
from asr_server.registry import default_models
from asr_server.scheduler import OperationScheduler, ScheduledOperation
from tests.helpers import make_wav


class RecordingAdapter(MockAsrAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.order: list[str] = []
        self.active = 0
        self.maximum_active = 0

    def begin(self, label: str) -> None:
        with self.lock:
            self.order.append(label)
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)

    def end(self) -> None:
        with self.lock:
            self.active -= 1

    async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str,
                         context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
        label = next((value for value in ("FIFO_A", "FIFO_B", "FIFO_C", "FIFO_CANCEL", "FIFO_SEED") if value in context), "unlabelled")
        self.begin(label)
        try:
            if label == "FIFO_A":
                self.entered.set()
                if not self.release.wait(timeout=10):
                    raise RuntimeError("test gate was not released")
            return await super().transcribe(audio, model_id=model_id, backend=backend, language=language,
                                           context=context, timestamps=timestamps, max_new_tokens=max_new_tokens)
        finally:
            self.end()

    async def align_text(self, audio: AudioInput, *, text: str, language: str = "auto",
                         timestamps: str = "auto") -> list[TimestampItem]:
        self.begin(text)
        try:
            return await super().align_text(audio, text=text, language=language, timestamps=timestamps)
        finally:
            self.end()


async def sync_queued(scheduler: OperationScheduler) -> ScheduledOperation:
    async with scheduler.condition:
        await asyncio.wait_for(scheduler.condition.wait_for(lambda: any(
            scheduler.operations[key].kind == "sync" for key in scheduler.queue
        )), timeout=3)
        return next(scheduler.operations[key] for key in scheduler.queue if scheduler.operations[key].kind == "sync")


async def test_real_async_sync_workbench_and_edit_share_fifo_and_cancel_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_TEMP_DIR", str(tmp_path / "temporary"))
    adapter = RecordingAdapter()
    definition = replace(default_models(enable_qwen_alignment=True)["qwen3-asr-0.6b"], id="test-fifo",
                         engine_id="test-fifo", default=True, pipelines=("legacy",), resource_type="cpu")
    engine = EngineRegistration("test-fifo", lambda model, settings: adapter, "test:fifo")
    app = create_app(Settings(adapter="registered", workbench_data_dir=tmp_path / "workbench", min_free_disk_mb=0,
                              enable_qwen_alignment=True), models={definition.id: definition}, engines={engine.engine_id: engine})
    service = app.state.workbench
    service.store.create_user("fifo-user", "fifo-password")
    scheduler: OperationScheduler = app.state.job_manager.scheduler
    source = make_wav(0.2)
    sync_request: asyncio.Task[Any] | None = None
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            assert (await client.post("/api/auth/login", json={"username": "fifo-user", "password": "fifo-password"})).status_code == 200

            async def workbench_job(label: str) -> str:
                upload = (await client.post("/api/uploads", json={"filename": "short.wav", "size_bytes": len(source)})).json()
                assert (await client.patch(f"/api/uploads/{upload['id']}", content=source, headers={"Upload-Offset": "0"})).status_code == 200
                created = await client.post(f"/api/uploads/{upload['id']}/complete", json={"quality_profile": "legacy", "hotwords": label})
                assert created.status_code == 202, created.text
                return str(created.json()["id"])

            # A genuine completed workbench transcript supplies the edit owner.
            seed_id = await workbench_job("FIFO_SEED")
            seed_operation = scheduler.operations.get(seed_id)
            if seed_operation is not None:
                await asyncio.wait_for(asyncio.shield(seed_operation.future), 3)
            seed = (await client.get(f"/api/jobs/{seed_id}")).json()
            assert seed["status"] == "completed", seed
            adapter.order.clear()
            adapter.maximum_active = 0
            try:
                created = await client.post("/v1/audio/transcription-jobs", files={"file": ("a.wav", source)}, data={"context": "FIFO_A"})
                assert created.status_code == 202, created.text
                first_id = created.json()["id"]
                first_operation = scheduler.operations[first_id]
                assert await asyncio.to_thread(adapter.entered.wait, 3)
                sync_request = asyncio.create_task(client.post("/v1/audio/transcriptions", files={"file": ("b.wav", source)}, data={"context": "FIFO_B"}))
                second_operation = await sync_queued(scheduler)
                third_id = await workbench_job("FIFO_C")
                third_operation = scheduler.operations[third_id]
                edit = await client.patch(f"/api/jobs/{seed_id}/segments/{seed['result']['segments'][0]['id']}",
                                          json={"base_version": 1, "text": "FIFO_D"})
                assert edit.status_code == 200, edit.text
                fourth_operation = scheduler.operations["align:" + seed_id]
                cancelled = await client.post("/v1/audio/transcription-jobs", files={"file": ("cancel.wav", source)}, data={"context": "FIFO_CANCEL"})
                assert cancelled.status_code == 202, cancelled.text
                cancelled_id = cancelled.json()["id"]
                cancelled_job = app.state.job_manager._jobs[cancelled_id]
                cancelled_root = cancelled_job.work_dir
                cancelled_operation = scheduler.operations[cancelled_id]
                assert first_operation.sequence < second_operation.sequence < third_operation.sequence < fourth_operation.sequence < cancelled_operation.sequence
                assert adapter.order == ["FIFO_A"] and not sync_request.done()
                cancellation = await client.delete(f"/v1/jobs/{cancelled_id}")
                assert cancellation.status_code == 200 and cancellation.json()["status"] == "cancelled"
                assert cancelled_operation.future.cancelled() and cancelled_id not in scheduler.operations
                assert not cancelled_root.exists()
                adapter.release.set()
                response = await asyncio.wait_for(sync_request, 3)
                assert response.status_code == 200, response.text
                await asyncio.wait_for(asyncio.gather(asyncio.shield(first_operation.future),
                    asyncio.shield(third_operation.future), asyncio.shield(fourth_operation.future)), 3)
                assert (await client.get(f"/v1/jobs/{first_id}")).json()["status"] == "completed"
                assert (await client.get(f"/api/jobs/{third_id}")).json()["status"] == "completed"
                edited = (await client.get(f"/api/jobs/{seed_id}")).json()["result"]
                assert edited["version"] == 2 and edited["alignment_status"] == "aligned"
                assert edited["segments"][0]["text"] == "FIFO_D"
                assert adapter.order == ["FIFO_A", "FIFO_B", "FIFO_C", "FIFO_D"]
                assert adapter.maximum_active == 1 and adapter.active == 0
                assert app.state.workspace_manager._reserved_upload_bytes == 0
            finally:
                adapter.release.set()
                if sync_request is not None:
                    await asyncio.gather(sync_request, return_exceptions=True)
