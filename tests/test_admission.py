from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncByteStream, AsyncClient

from asr_server.config import Settings
from asr_server.main import create_app


@pytest.mark.parametrize('endpoint', ['/v1/audio/transcriptions', '/v1/audio/transcription-jobs'])
async def test_stream_limit_precedes_upload_copy_and_reclaims_partial_parser(
    endpoint: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('ASR_TEMP_DIR', str(tmp_path))
    app = create_app(Settings(max_upload_mb=1, min_free_disk_mb=0, workbench_enabled=False))
    reads = 0
    class Body(AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            nonlocal reads
            yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="x.wav"\r\n\r\n'
            for _ in range(16):
                reads += 1
                yield b'x' * (256 * 1024)
            yield b'\r\n--boundary--\r\n'
    async def should_not_copy(*args: Any, **kwargs: Any) -> None:
        raise AssertionError('oversized multipart reached workspace copy')
    monkeypatch.setattr(app.state.workspace_manager, 'store_upload', should_not_copy)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post(endpoint, content=Body(), headers={'Content-Type': 'multipart/form-data; boundary=boundary'})
    assert response.status_code == 413
    assert response.json()['error']['code'] == 'audio_too_large'
    assert reads < 16
    assert app.state.workspace_manager._reserved_upload_bytes == 0
    assert not list(tmp_path.iterdir())
    assert response.headers['X-Request-ID'].startswith('req_')


async def test_receive_slots_auth_timeout_and_cancel_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ASR_TEMP_DIR', str(tmp_path))
    app = create_app(Settings(receive_slots=1, receive_timeout_seconds=0.1, api_token='test-token',
                              min_free_disk_mb=0, workbench_enabled=False))
    entered, release = asyncio.Event(), asyncio.Event()
    reads = 0
    class Body(AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            nonlocal reads
            reads += 1
            entered.set()
            await release.wait()
            yield b'{}'
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        unauthorized = await client.post('/v1/models/qwen3-asr-1.7b/load', content=Body())
        assert unauthorized.status_code == 401 and reads == 0
        headers = {'Authorization': 'Bearer test-token', 'Content-Type': 'application/json'}
        pending = asyncio.create_task(client.post('/v1/models/qwen3-asr-1.7b/load', content=Body(), headers=headers))
        await entered.wait()
        busy = await client.post('/v1/models/qwen3-asr-1.7b/load', content=Body(), headers=headers)
        assert busy.status_code == 429 and reads == 1
        timed_out = await pending
        assert timed_out.status_code == 408
        entered.clear()
        pending = asyncio.create_task(client.post('/v1/models/qwen3-asr-1.7b/load', content=Body(), headers=headers))
        await entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()
        response = await client.post('/v1/models/qwen3-asr-1.7b/load', json={}, headers=headers)
        assert response.status_code == 200
    await app.state.job_manager.shutdown()
    await app.state.manager.shutdown()
    assert app.state.workspace_manager._reserved_upload_bytes == 0
