from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from asr_server.workbench import service as service_module
from asr_server.workbench.service import WorkbenchService
from tests.test_workbench import audio_bytes, completed, login, submit, workbench


async def uploaded(client: AsyncClient) -> str:
    content = audio_bytes()
    upload = (await client.post('/api/uploads', json={'filename': 'review.wav', 'size_bytes': len(content)})).json()
    assert (await client.patch(f"/api/uploads/{upload['id']}", content=content, headers={'Upload-Offset': '0'})).status_code == 200
    return str(upload['id'])


async def test_review_healthy_selected_model_survives_default_model_circuit(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    app, _service, client = workbench
    await login(client)
    upload_id = await uploaded(client)
    app.state.manager.runtime_for(app.state.manager.default_model_id).circuit_until = time.monotonic() + 60
    response = await client.post(f'/api/uploads/{upload_id}/complete', json={'model': 'qwen3-asr-0.6b'})
    assert response.status_code == 202, response.text
    await completed(client, response.json()['id'])


async def test_review_drain_fences_submission_that_was_hashing(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    app, service, client = workbench
    await login(client)
    upload_id = await uploaded(client)
    entered, release = threading.Event(), threading.Event()
    real_hash = service_module.sha256_file
    def held_hash(*args: Any, **kwargs: Any) -> str:
        entered.set()
        assert release.wait(5)
        return real_hash(*args, **kwargs)
    monkeypatch.setattr(service_module, 'sha256_file', held_hash)
    pending = asyncio.create_task(client.post(f'/api/uploads/{upload_id}/complete', json={}))
    assert await asyncio.to_thread(entered.wait, 2)
    app.state.job_manager.scheduler.set_draining(True)
    service.set_draining(True)
    release.set()
    response = await pending
    assert response.status_code == 503, response.text
    assert (await client.get('/api/jobs')).json()['jobs'] == []


async def test_review_drain_keeps_edit_alignment_pending(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    app, service, client = workbench
    await login(client)
    job_id = await submit(client)
    job = await completed(client, job_id)
    entered, release = asyncio.Event(), asyncio.Event()
    real_prepare = service.prepare_audio
    async def held_prepare(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        return await real_prepare(*args, **kwargs)
    monkeypatch.setattr(service, 'prepare_audio', held_prepare)
    response = await client.patch(f"/api/jobs/{job_id}/segments/{job['result']['segments'][0]['id']}", json={'text': '修改后正文', 'base_version': 1})
    assert response.status_code == 200
    await asyncio.wait_for(entered.wait(), 1)
    app.state.job_manager.scheduler.set_draining(True)
    service.set_draining(True)
    release.set()
    alignment = app.state.job_manager.scheduler.operations.get('align:' + job_id)
    if alignment is not None:
        await alignment.future
    final = (await client.get(f'/api/jobs/{job_id}')).json()
    assert final['result']['alignment_status'] == 'pending', final['result']


def test_review_due_retry_joins_after_existing_durable_submissions(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from asr_server.workbench import store as store_module
    from asr_server.workbench.store import WorkbenchStore
    from tests.test_workbench_storage import insert_job
    store = WorkbenchStore(tmp_path)
    store.initialize()
    try:
        insert_job(store, {'id': 'retry', 'owner_id': 'alice', 'status': 'queued', 'created_at': '2000-01-01T00:00:00+00:00'}, 0)
        attempt = store.begin_attempt('retry')
        monkeypatch.setattr(store_module, 'now', lambda: '2000-01-01T00:00:01+00:00')
        store.schedule_infrastructure_retry('retry', attempt['attempt_id'], attempt['generation'], {'code': 'worker_unavailable'}, delay_seconds=0)
        insert_job(store, {'id': 'already_waiting', 'owner_id': 'alice', 'status': 'queued', 'created_at': '2000-01-01T00:00:02+00:00'}, 0)
        monkeypatch.setattr(store_module, 'now', lambda: '2000-01-01T00:00:10+00:00')
        store.claim_due_retry('retry', at='9999-01-01T00:00:00+00:00')
        assert store.pending_jobs() == ['already_waiting', 'retry']
    finally:
        store.close()


async def test_review_direct_repository_write_failure_closes_write_admission(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3
    from contextlib import contextmanager
    app, service, client = workbench
    await login(client)
    job_id = await submit(client)
    job = await completed(client, job_id)
    operation = app.state.job_manager.scheduler.operations.get(job_id)
    if operation is not None:
        await asyncio.shield(operation.future)
    assert job_id not in service.active and job_id not in service.attempts
    original = service.store.connection
    @contextmanager
    def readonly() -> Any:
        with original() as db:
            db.execute('PRAGMA query_only=ON')
            yield db
    with monkeypatch.context() as patch:
        patch.setattr(service.store, 'connection', readonly)
        with pytest.raises(sqlite3.OperationalError):
            await client.patch(f"/api/jobs/{job_id}/segments/{job['result']['segments'][0]['id']}", json={'text': '修改后正文', 'base_version': 1})
        assert not service.readiness()['ready']
        assert 'storage_unavailable' in service.readiness()['reasons']


async def test_review_already_queued_task_waits_out_worker_circuit(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    app, service, client = workbench
    entered, release = asyncio.Event(), asyncio.Event()
    async def block() -> None:
        entered.set()
        await release.wait()
    held = await app.state.job_manager.scheduler.submit('review-gpu-owner', block)
    await entered.wait()
    await login(client)
    job_id = await submit(client)
    assert app.state.job_manager.scheduler.position(job_id) > 0
    app.state.manager.runtime_for(app.state.manager.default_model_id).circuit_until = time.monotonic() + 60
    release.set()
    await held.future
    queued = app.state.job_manager.scheduler.operations.get(job_id)
    if queued is not None:
        await queued.future
    job = (await client.get(f'/api/jobs/{job_id}')).json()
    assert job['status'] == 'queued', job
    assert job['next_attempt_at']
    assert job.get('recovery', {}).get('infrastructure_retries', 0) == 0
