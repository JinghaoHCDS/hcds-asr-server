from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.main import create_app
from asr_server.workbench.service import WorkbenchService
from tests.helpers import make_wav
from tests.test_retained_video import make_video


async def wait(service: WorkbenchService, key: str) -> None:
    operation = service.app.state.job_manager.scheduler.operations.get(key)
    if operation:
        await asyncio.wait_for(asyncio.shield(operation.future), 30)
    job_id = key.removeprefix('video:')
    task = service.video_tasks.get(job_id)
    if task:
        await asyncio.wait_for(asyncio.shield(task), 30)


async def upload(client: AsyncClient, content: bytes, *, video: bool = False) -> str:
    response = await client.post('/api/uploads', json={'filename': 'synthetic.mp4' if video else 'synthetic.wav', 'size_bytes': len(content)})
    upload_id = response.json()['id']
    await client.patch(f'/api/uploads/{upload_id}', content=content, headers={'Upload-Offset': '0'})
    response = await client.post(f'/api/uploads/{upload_id}/complete', json={'model': 'moss-transcribe-diarize-0.9b', 'quality_profile': 'legacy', 'expected_speakers': 3})
    assert response.status_code == 202, response.text
    return str(response.json()['id'])


async def mock_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return {'text': '甲乙丙', 'duration': 1.5, 'segments': [
        {'start': 0, 'end': .4, 'speaker': 'speaker-0001', 'speaker_scope': 'global', 'text': '甲'},
        {'start': .5, 'end': .9, 'speaker': 'speaker-0002', 'speaker_scope': 'global', 'text': '乙'},
        {'start': 1., 'end': 1.4, 'speaker': 'speaker-0003', 'speaker_scope': 'global', 'text': '丙'}],
        'diarization': {'status': 'complete', 'speaker_scope': 'global'}}


async def test_moss_edit_versions_names_and_owner_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('asr_server.workbench.service.run_transcription_path', mock_result)
    app = create_app(Settings(adapter='mock', workbench_data_dir=tmp_path / 'data', enable_moss=True, min_free_disk_mb=0))
    service = app.state.workbench
    service.store.create_user('alice', 'synthetic-password')
    service.store.create_user('bob', 'synthetic-password')
    async with app.router.lifespan_context(app), AsyncClient(transport=ASGITransport(app), base_url='http://testserver') as client:
        await client.post('/api/auth/login', json={'username': 'alice', 'password': 'synthetic-password'})
        key = await upload(client, make_wav(1.5))
        await wait(service, key)
        job = (await client.get(f'/api/jobs/{key}')).json()
        assert job['status'] == 'completed', job
        request = service.store.job(key)['request']
        assert request['speaker_resolution'] == 'auto' and request['max_new_tokens'] == 65536
        assert len(job['result']['speakers']) == 3
        assert job['result']['alignment_status'] == 'segment_timed'
        expires = job['expires_at']
        response = await client.patch(f'/api/jobs/{key}/speakers/P1', json={'name': '我', 'base_version': 1})
        assert response.status_code == 200
        response = await client.patch(f'/api/jobs/{key}/segments/segment-1', json={'speaker_id': 'P1', 'text': '乙修正', 'base_version': 2})
        assert response.status_code == 200, response.text
        result = response.json()['result']
        assert result['version'] == 3 and result['alignment_status'] == 'segment_timed'
        assert result['segments'][1]['speaker_id'] == 'P1'
        assert result['segments'][1]['model_speaker_id'] == 'P2'
        assert result['segments'][1]['raw_text'] == '乙'
        assert response.json()['expires_at'] == expires
        assert (await client.patch(f'/api/jobs/{key}/speakers/P1', json={'name': '过时', 'base_version': 1})).status_code == 409
        assert (await client.patch(f'/api/jobs/{key}/segments/segment-1', json={'speaker_id': 'P999', 'base_version': 3})).status_code == 404
        assert (await client.post(f'/api/jobs/{key}/speakers', json={'name': '第四人', 'base_version': 3})).status_code == 200
        assert 'P1 我' in (await client.get(f'/api/jobs/{key}/export?format=md&version=3')).text
        assert 'P1 说话人 1' in (await client.get(f'/api/jobs/{key}/export?format=md&version=1')).text
        assert (await client.get(f'/api/jobs/{key}/export?format=srt&version=3')).status_code == 200
        await client.post('/api/auth/login', json={'username': 'bob', 'password': 'synthetic-password'})
        assert (await client.patch(f'/api/jobs/{key}/speakers/P1', json={'name': '禁止', 'base_version': 4})).status_code == 404
        assert (await client.get(f'/api/jobs/{key}/video')).status_code == 404


async def test_video_failure_retry_playback_and_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from asr_server.workbench import service as module
    original = getattr(module, "prepare_video")
    async def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AsrError(400, 'video_encode_failed', 'synthetic failure')
    monkeypatch.setattr(module, 'prepare_video', fail)
    monkeypatch.setattr(module, 'run_transcription_path', mock_result)
    source = tmp_path / 'synthetic.mp4'
    make_video(source)
    app = create_app(Settings(adapter='mock', workbench_data_dir=tmp_path / 'data', enable_moss=True, min_free_disk_mb=0))
    service = app.state.workbench
    user = service.store.create_user('alice', 'synthetic-password')
    async with app.router.lifespan_context(app), AsyncClient(transport=ASGITransport(app), base_url='http://testserver') as client:
        await client.post('/api/auth/login', json={'username': 'alice', 'password': 'synthetic-password'})
        key = await upload(client, source.read_bytes(), video=True)
        await wait(service, key)
        job = service.store.job(key)
        assert job['status'] == 'completed' and job['video_asset']['state'] == 'failed'
        root = service.job_dir(key)
        assert (root / 'source').exists()
        audio = (root / 'audio.mp3').read_bytes()
        expires = job['expires_at']
        monkeypatch.setattr(module, 'prepare_video', original)
        response = await client.post(f'/api/jobs/{key}/video/retry')
        assert response.status_code == 202, response.text
        await wait(service, 'video:' + key)
        job = service.store.job(key)
        assert job['video_asset']['state'] == 'ready', job['video_asset']
        assert job['expires_at'] == expires and job['result']['version'] == 1
        assert not (root / 'source').exists() and (root / 'audio.mp3').read_bytes() == audio
        response = await client.get(f'/api/jobs/{key}/video', headers={'Range': 'bytes=0-99'})
        assert response.status_code == 206 and response.headers['content-type'] == 'video/mp4'
        usage = (await client.get('/api/storage')).json()
        assert usage['temporary_bytes'] == usage['reserved_bytes'] == 0
        assert usage['saved_bytes'] == (root / 'audio.mp3').stat().st_size + (root / 'video.mp4').stat().st_size
        await service.acquire_playback(user['id'], key, video=True)
        assert not await service.delete(user['id'], key)
        assert root.exists()
        await service.release_playback(key)
        assert not root.exists()
        assert (await client.get('/api/storage')).json()['allocated_bytes'] == 0

async def test_video_restart_migration_and_expiry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from asr_server.workbench import service as module
    from asr_server.workbench.store import WorkbenchStore
    monkeypatch.setattr(module, 'run_transcription_path', mock_result)
    source = tmp_path / 'synthetic.mp4'
    make_video(source)
    settings = Settings(adapter='mock', workbench_data_dir=tmp_path / 'data', enable_moss=True, min_free_disk_mb=0)
    app = create_app(settings)
    service = app.state.workbench
    service.store.create_user('alice', 'synthetic-password')
    async with app.router.lifespan_context(app), AsyncClient(transport=ASGITransport(app), base_url='http://testserver') as client:
        await client.post('/api/auth/login', json={'username': 'alice', 'password': 'synthetic-password'})
        key = await upload(client, source.read_bytes(), video=True)
        await wait(service, key)
        job = service.store.job(key)
        assert job['video_asset']['state'] == 'ready'
        old_result = job['result']
        old_expiry = job['expires_at']
        # Simulate a schema-5 store with a retained source interrupted before video commit.
        root = service.job_dir(key)
        (root / 'source').write_bytes(source.read_bytes())
        (root / 'video.mp4').unlink()
        (root / 'video.partial.mp4').write_bytes(b'partial')
        job['video_asset'] = {'state': 'preparing'}
        service.store.save_job(job)
        with service.store.connection() as db:
            db.execute('PRAGMA user_version=5')
    migrated = WorkbenchStore(settings.workbench_data_dir)
    migrated.initialize()
    assert migrated.job(key)['result'] == old_result
    assert migrated.job(key)['expires_at'] == old_expiry
    with migrated.connection() as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 6
    migrated.close()
    restarted = create_app(settings)
    async with restarted.router.lifespan_context(restarted):
        restored = restarted.state.workbench
        await restored.fill_queue()
        await wait(restored, 'video:' + key)
        job = restored.store.job(key)
        assert job['video_asset']['state'] == 'ready', job
        assert job['result'] == old_result and job['expires_at'] == old_expiry
        assert not list(root.glob('*.partial*')) and not (root / 'source').exists()
        job['expires_at'] = '2000-01-01T00:00:00+00:00'
        restored.store.save_job(job)
        await restored.cleanup()
        assert not root.exists()


@pytest.mark.parametrize('action', ['cancel', 'delete'])
async def test_video_retry_cancel_delete_no_late_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    from asr_server.workbench import service as module
    monkeypatch.setattr(module, 'run_transcription_path', mock_result)
    async def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AsrError(400, 'video_encode_failed', 'synthetic failure')
    monkeypatch.setattr(module, 'prepare_video', fail)
    source = tmp_path / 'synthetic.mp4'
    make_video(source)
    app = create_app(Settings(adapter='mock', workbench_data_dir=tmp_path / 'data', enable_moss=True, min_free_disk_mb=0))
    service = app.state.workbench
    user = service.store.create_user('alice', 'synthetic-password')
    async with app.router.lifespan_context(app), AsyncClient(transport=ASGITransport(app), base_url='http://testserver') as client:
        await client.post('/api/auth/login', json={'username': 'alice', 'password': 'synthetic-password'})
        key = await upload(client, source.read_bytes(), video=True)
        await wait(service, key)
        started, finish = asyncio.Event(), asyncio.Event()
        async def blocked(*args: Any, **kwargs: Any) -> dict[str, Any]:
            await kwargs['progress']({'phase': 'encoding_video', 'percent': 25.0, 'processed_seconds': 5, 'duration_seconds': 20})
            started.set()
            while not kwargs['context'].cancel_event.is_set():
                await asyncio.sleep(.01)
            await finish.wait()
            # Simulate a late encoder return after its final atomic rename.
            (service.job_dir(key) / 'video.mp4').write_bytes(b'uncommitted')
            return {'state': 'ready', 'size_bytes': 11}
        monkeypatch.setattr(module, 'prepare_video', blocked)
        assert (await client.post(f'/api/jobs/{key}/video/retry')).status_code == 202
        await asyncio.wait_for(started.wait(), 10)
        summary = (await client.get('/api/jobs?view=summary')).json()['jobs'][0]
        assert summary['video_asset']['progress']['percent'] == 25
        assert summary['status'] == 'completed'  # Video progress must not rewrite transcription completion.
        detail = (await client.get('/api/jobs/' + key)).json()
        assert detail['video_asset']['progress']['phase'] == 'encoding_video'
        if action == 'delete':
            await service.acquire_playback(user['id'], key)
            assert not await service.delete(user['id'], key)
            await service.release_playback(key)
            assert service.job_dir(key).exists()  # Encoder still owns its files.
        else:
            await service.cancel(user['id'], key)
        finish.set()
        await wait(service, 'video:' + key)
        if action == 'delete':
            assert not service.job_dir(key).exists()
        else:
            job = service.store.job(key)
            assert job['status'] == 'completed' and job['video_asset']['state'] == 'failed'
            assert job['result']['version'] == 1
            assert (service.job_dir(key) / 'source').exists()
            assert not (service.job_dir(key) / 'video.mp4').exists()

async def test_video_does_not_block_transcript_and_keeps_separate_reservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from asr_server.workbench import service as module
    original = getattr(module, "prepare_video")
    started, release = asyncio.Event(), asyncio.Event()
    async def delayed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        pending = Path(args[1]).with_name('video.unreported.partial.mp4')
        pending.write_bytes(b'x' * (2 * 1024 * 1024))
        started.set()
        await release.wait()
        pending.unlink()
        return dict(await original(*args, **kwargs))
    monkeypatch.setattr(module, 'prepare_video', delayed)
    monkeypatch.setattr(module, 'run_transcription_path', mock_result)
    source = tmp_path / 'synthetic.mp4';make_video(source)
    app = create_app(Settings(adapter='mock', video_encoder='cpu', workbench_data_dir=tmp_path / 'data', enable_moss=True, min_free_disk_mb=0))
    service = app.state.workbench
    service.store.create_user('alice', 'synthetic-password')
    async with app.router.lifespan_context(app), AsyncClient(transport=ASGITransport(app), base_url='http://testserver') as client:
        await client.post('/api/auth/login', json={'username':'alice', 'password':'synthetic-password'})
        key = await upload(client, source.read_bytes(), video=True)
        await asyncio.wait_for(started.wait(), 10)
        operation = service.app.state.job_manager.scheduler.operations.get(key)
        if operation:
            await asyncio.wait_for(asyncio.shield(operation.future), 10)
        job = service.store.job(key)
        assert job['status'] == 'completed' and job['result']['text']
        assert job['video_asset']['state'] == 'preparing'
        declarations = service.store.future_disk_reservations()
        assert declarations['video:' + key] > 0 and declarations.get('job:' + key, 0) == 0
        expires = job['expires_at']
        release.set()
        await wait(service, 'video:' + key)
        job = service.store.job(key)
        assert job['video_asset']['state'] == 'ready' and job['expires_at'] == expires
        assert not service.store.future_disk_reservations()
        root = service.job_dir(key)
        assert not (root / 'source').exists() and not list(root.glob('*.partial*'))
        storage = (await client.get('/api/storage')).json()
        assert storage['reserved_bytes'] == 0
        assert storage['allocated_bytes'] == storage['saved_bytes'] + storage['text_bytes']

async def test_gpu_grace_is_bounded_and_cancelable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from asr_server.workbench.service import WorkbenchInterrupted
    app = create_app(Settings(adapter='mock', workbench_data_dir=tmp_path / 'data', video_inference_grace_seconds=.05))
    service = app.state.workbench
    calls: list[str] = []
    async def metadata(*args: Any) -> dict[str, Any]:
        return {'video_asset': {'state':'preparing', 'encoder':'h264_nvenc', 'progress':{'phase':'encoding_video'}}}
    async def announce(job_id: str, phase: str) -> None: calls.append(phase)
    monkeypatch.setattr(service.store, 'run', metadata)
    monkeypatch.setattr(service, 'audio_phase', announce)
    sleeping = asyncio.create_task(asyncio.sleep(60))
    service.video_tasks['synthetic'] = sleeping
    try:
        await asyncio.wait_for(service.wait_for_video_capacity('synthetic'), .5)
        assert calls == ['waiting_for_video_capacity']
        assert not sleeping.done()  # Transcription proceeds even while encode is unfinished.
        service.cancel_intents.add('synthetic')
        with pytest.raises(WorkbenchInterrupted):
            await service.wait_for_video_capacity('synthetic')
    finally:
        sleeping.cancel()
        await asyncio.gather(sleeping, return_exceptions=True)
        service.close()
