from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from httpx import AsyncClient
from fastapi import FastAPI

from asr_server.errors import AsrError
from asr_server.workbench import service as service_module
from asr_server.workbench.service import WorkbenchService
from tests.test_workbench import workbench, login, submit, completed


async def test_sparse_partial_result_is_visible_exported_and_isolated(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    _app, _service, client = workbench
    async def incomplete(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        await kwargs['stage_callback']('transcribing', {'completed_chunks': 2, 'resolved_chunks': 2, 'total_chunks': 3})
        late = kwargs['checkpoint_dir'] / 'recovery-v1' / 'late-review.json'
        late.parent.mkdir(parents=True, exist_ok=True)
        late.write_text('{"status":"unresolved"}')
        partial = {'text': '前段\n后段', 'duration': 180., 'timestamps': [],
                   'segments': [{'text': '前段', 'start': 0., 'end': 40.}, {'text': '后段', 'start': 120., 'end': 150.}],
                   'quality': {'issues': []}, 'completeness': {'complete': False, 'resolved_chunks': 2, 'total_chunks': 3, 'gap_count': 1},
                   'gaps': [{'chunk_index': 1, 'start': 60., 'end': 120., 'reason': 'empty_transcription'}],
                   'retryability': 'exhausted', 'next_action': 'listen_or_change_parameters'}
        raise AsrError(422, 'quality_incomplete', 'unresolved audio intervals', {'partial_result': partial, 'gaps': partial['gaps']})
    monkeypatch.setattr(service_module, 'run_transcription_path', incomplete)
    await login(client)
    job_id = await submit(client)
    for _ in range(150):
        job = (await client.get(f'/api/jobs/{job_id}')).json()
        if job['status'] == 'failed':
            break
        await asyncio.sleep(.01)
    assert job['status'] == 'failed'
    assert job['result']['text'] == '前段\n后段'
    assert job['completeness']['gap_count'] == 1
    saved = await _service.store.run(_service.store.job_metadata, job_id)
    assert saved['retained_file_bytes']['checkpoints/recovery-v1/late-review.json'] > 0
    assert 'partial_result' not in job['error']['details']
    txt = await client.get(f'/api/jobs/{job_id}/export?format=txt')
    assert txt.status_code == 200 and '不完整转录' in txt.text and '00:01:00.000' in txt.text
    assert (await client.get(f'/api/jobs/{job_id}/export?format=srt')).status_code == 409
    assert (await client.post(f'/api/jobs/{job_id}/retry')).status_code == 409
    summary = (await client.get('/api/jobs?view=summary')).json()['jobs'][0]
    assert '前段' not in str(summary) and summary['completeness']['gap_count'] == 1
    await login(client, 'bob', 'bob-password')
    assert (await client.get(f'/api/jobs/{job_id}/export?format=json')).status_code == 404


async def test_worker_retry_releases_slot_for_next_task(workbench: tuple[FastAPI, WorkbenchService, AsyncClient], monkeypatch: pytest.MonkeyPatch) -> None:
    app, service, client = workbench
    healthy: Callable[..., Awaitable[dict[str, Any]]] = getattr(service_module, "run_transcription_path")
    calls = 0
    async def fail_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise AsrError(503, 'inference_failed', 'worker exited', {'worker_fatal': True, 'worker_reaped': True})
        return await healthy(*args, **kwargs)
    monkeypatch.setattr(service_module, 'run_transcription_path', fail_once)
    await login(client)
    first = await submit(client, key='recover')
    for _ in range(150):
        job = (await client.get(f'/api/jobs/{first}')).json()
        if job['progress']['phase'] == 'retry_wait':
            break
        await asyncio.sleep(.01)
    assert job['progress']['phase'] == 'retry_wait'
    assert job['next_attempt_at'] and job['recovery']['infrastructure_retries'] == 1
    second = await submit(client, key='healthy')
    assert (await completed(client, second))['status'] == 'completed'
    assert (await client.get(f'/api/jobs/{first}')).json()['progress']['phase'] == 'retry_wait'
    await client.post(f'/api/jobs/{first}/cancel')
    assert (await client.get(f'/api/jobs/{first}')).json()['status'] == 'cancelled'
    assert app.state.job_manager.scheduler.readiness()['ready']


async def test_admin_drain_keeps_reads_available_and_blocks_submission(workbench: tuple[FastAPI, WorkbenchService, AsyncClient]) -> None:
    _app, service, client = workbench
    _app.state.started = True  # This fixture starts its components explicitly.
    await login(client)
    job_id = await submit(client)
    await completed(client, job_id)
    assert (await client.post('/api/admin/drain', json={'draining': True})).status_code == 403
    await login(client, 'admin', 'correct-password')
    assert (await client.post('/api/admin/drain', json={'draining': True})).status_code == 200
    assert (await client.get('/readyz')).status_code == 503
    await login(client)
    assert (await client.get(f'/api/jobs/{job_id}')).status_code == 200
    assert (await client.get(f'/api/jobs/{job_id}/audio', headers={'Range': 'bytes=0-43'})).status_code == 206
    upload = (await client.post('/api/uploads', json={'filename': 'new.wav', 'size_bytes': 8})).json()
    assert (await client.post(f"/api/uploads/{upload['id']}/complete", json={})).status_code == 503
    await login(client, 'admin', 'correct-password')
    await client.post('/api/admin/drain', json={'draining': False})
    assert (await client.get('/readyz')).status_code == 200
    assert not service.draining
