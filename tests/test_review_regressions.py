"""Regressions reproduced in the 2026-10-25 code/production review."""
from __future__ import annotations

import asyncio
import copy
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from asr_server.audio.assets import NormalizedAudioAsset
from asr_server.audio.preprocess import MediaPreprocessor
from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.main import create_app
from asr_server.workbench.service import WorkbenchService, export_result
from tests.helpers import make_wav


@asynccontextmanager
async def running(tmp_path: Path, **settings: Any) -> AsyncIterator[WorkbenchService]:
    app = create_app(Settings(adapter='mock', workbench_data_dir=tmp_path / 'workbench',
                              min_free_disk_mb=0, **settings))
    async with app.router.lifespan_context(app):
        yield app.state.workbench


async def submit(service: WorkbenchService, content: bytes | None = None, *, owner: str = "review-user",
                 model: str | None = None) -> str:
    content = content or make_wav(.1)
    upload = await service.create_upload(owner, 'fixture.wav', len(content))
    await service.append_upload(owner, upload['id'], 0, content)
    job = await service.complete_upload(owner, upload['id'], {'quality_profile': 'legacy', **({'model': model} if model else {})})
    return str(job['id'])


@pytest.mark.parametrize('action', ['cancel', 'delete'])
@pytest.mark.parametrize('prefetch', [False, True])
async def test_preparation_cancellation_finishes_after_decoder_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str, prefetch: bool,
) -> None:
    entered, release_decoder = asyncio.Event(), asyncio.Event()

    async def decode(source: Path, output: Path, **kwargs: Any) -> None:
        output.write_bytes(b'partial')
        entered.set()
        while not kwargs['cancel_event'].is_set():
            await asyncio.sleep(0)
        await release_decoder.wait()
        output.unlink()
        raise asyncio.CancelledError

    monkeypatch.setattr('asr_server.workbench.service.normalize_audio_path_to_wav_async', decode)
    async with running(tmp_path, preprocessing_prefetch=prefetch) as service:
        scheduler = service.app.state.job_manager.scheduler
        job_id = await submit(service)
        operation = scheduler.operations[job_id]
        await asyncio.wait_for(entered.wait(), 2)
        await getattr(service, action)('review-user', job_id)
        assert (await service.store.run(service.store.job_metadata, job_id))['status'] == 'cancel_requested'
        release_decoder.set()
        await asyncio.wait_for(asyncio.shield(operation.future), 2)
        if action == 'delete':
            with pytest.raises(AsrError, match='not found'):
                await service.store.run(service.store.job, job_id)
            assert not service.job_dir(job_id).exists()
        else:
            job = await service.store.run(service.store.job, job_id)
            assert job['status'] == 'cancelled' and job['expires_at']
            assert not (service.job_dir(job_id) / 'audio.partial.wav').exists()
            assert (service.job_dir(job_id) / 'audio.mp3').exists()  # retry reconstructs PCM
        assert not service.active and not service.prepared_assets and not service.preparation_errors
        assert service.app.state.workspace_manager._reserved_upload_bytes == 0
        assert service.app.state.workspace_manager.pending_disk_bytes(service.store.root) == 0


async def test_durable_backlog_refills_before_new_admission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release, older_started, older_release = (asyncio.Event() for _ in range(4))
    order: list[str] = []
    async def first() -> None:
        entered.set()
        await release.wait()
    async def transcribe(*args: Any, **kwargs: Any) -> dict[str, Any]:
        order.append('older')
        older_started.set()
        await older_release.wait()
        return {'text': 'ok', 'duration': .1, 'segments': [], 'timestamps': []}
    async def newer() -> None:
        order.append('newer')
    monkeypatch.setattr('asr_server.workbench.service.run_transcription_path', transcribe)
    async with running(tmp_path, max_queued_jobs=1) as service:
        scheduler = service.app.state.job_manager.scheduler
        first_op = await scheduler.submit('first', first)
        await entered.wait()
        job_id = await submit(service)
        assert job_id not in scheduler.operations
        release.set()
        await first_op.future
        # This may race with completion refill. Either way the old owner wins.
        with pytest.raises(AsrError) as full:
            await scheduler.submit('newer', newer)
        assert full.value.code == 'job_queue_full'
        await asyncio.wait_for(older_started.wait(), 2)
        older = scheduler.operations[job_id]
        older_release.set()
        await older.future
        await scheduler.wait('newer', newer)
        assert order == ['older', 'newer']
        assert (await service.store.run(service.store.job, job_id))['status'] == 'completed'


async def test_vbr_estimate_does_not_limit_actual_pcm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg required')
    parts = []
    for index, (seconds, rate) in enumerate([(10, '320k'), (60, '32k')]):
        path = tmp_path / f'{index}.mp3'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'sine=frequency=440:duration={seconds}',
                        '-c:a', 'libmp3lame', '-b:a', rate, '-write_xing', '0', str(path)], check=True)
        parts.append(path.read_bytes())
    # Decode the concatenated stream once into a valid, tag-consistent VBR asset.
    joined = tmp_path / 'joined.mp3'
    joined.write_bytes(b''.join(parts))
    source = tmp_path / 'vbr.mp3'
    subprocess.run(['ffmpeg', '-v', 'quiet', '-i', str(joined), '-c:a', 'libmp3lame', '-q:a', '5', str(source)], check=True)
    content = source.read_bytes()
    estimate = await MediaPreprocessor().probe(source)
    assert estimate is not None
    async def underestimate(*args: Any, **kwargs: Any) -> float:
        return 16.0
    monkeypatch.setattr('asr_server.workbench.service.probe_audio_duration_path_async', underestimate)
    async with running(tmp_path) as service:
        job_id = await submit(service, content)
        operation = service.app.state.job_manager.scheduler.operations[job_id]
        await asyncio.wait_for(asyncio.shield(operation.future), 5)
        job = await service.store.run(service.store.job, job_id)
        assert job['status'] == 'completed', job.get('error')
        assert 69 < job['normalized_asset']['frame_count'] / 16000 < 71
        assert service.app.state.workspace_manager._reserved_upload_bytes == 0
        assert service.app.state.workspace_manager.pending_disk_bytes(service.store.root) == 0


async def test_pcm_growth_obeys_persistent_quota(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def underestimate(*args: Any, **kwargs: Any) -> float:
        return .01
    monkeypatch.setattr('asr_server.workbench.service.probe_audio_duration_path_async', underestimate)
    async with running(tmp_path, workbench_user_quota_bytes=4_700_000) as service:
        # Saving the MP3 fits; adding PCM plus the worker slice exceeds the quota.
        job_id = await submit(service, make_wav(45))
        operation = service.app.state.job_manager.scheduler.operations[job_id]
        await asyncio.wait_for(asyncio.shield(operation.future), 5)
        job = await service.store.run(service.store.job, job_id)
        assert job['status'] == 'failed'
        assert job['error']['code'] == 'storage_quota_exceeded'
        assert not (service.job_dir(job_id) / 'audio.partial.wav').exists()
        assert service.app.state.workspace_manager._reserved_upload_bytes == 0
        assert service.app.state.workspace_manager.pending_disk_bytes(service.store.root) == 0


async def test_fresh_asset_is_borrowed_without_pcm_rescan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scans: list[Path] = []
    original = NormalizedAudioAsset.from_path
    def record(path: Path, **kwargs: Any) -> NormalizedAudioAsset:
        scans.append(path)
        return original(path, **kwargs)
    monkeypatch.setattr(NormalizedAudioAsset, 'from_path', record)
    async with running(tmp_path) as service:
        job_id = await submit(service)
        await asyncio.wait_for(service.app.state.job_manager.scheduler.operations[job_id].future, 3)
        assert not scans
        job = await service.store.run(service.store.job, job_id)
        assert job['status'] == 'completed'
        assert not (service.job_dir(job_id) / 'operation').exists()
        asset = await service.prepare_audio(job_id)
        path = asset.path
        asset = NormalizedAudioAsset.from_metadata(path, job['normalized_asset'])
        assert scans == [path]  # recovery still verifies PCM contents
        asset.validate()
        assert scans == [path]
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        with pytest.raises(AsrError) as invalid:
            asset.validate()
        assert invalid.value.code == 'normalized_asset_invalid'
        await service.cleanup_operation(job_id)


@pytest.mark.parametrize('format_', ['srt', 'vtt'])
def test_model_segments_export_without_claiming_word_alignment(format_: str) -> None:
    result: dict[str, Any] = {'text': 'hello\nworld', 'duration': 3., 'alignment_status': 'unavailable',
              'segments': [{'text': 'hello', 'start': .2, 'end': 1.0}, {'text': 'world', 'start': 1.5, 'end': 2.8}]}
    output, _ = export_result(result, format_, segment_timestamps=True)
    assert 'hello' in output and 'world' in output
    assert ('00:00:00.200 --> 00:00:01.000' if format_ == 'vtt' else '00:00:00,200 --> 00:00:01,000') in output
    assert result['alignment_status'] == 'unavailable'
    with pytest.raises(AsrError):
        export_result(result, format_)  # chunk boundaries aren't model timing
    for state in ['pending', 'failed', 'incomplete']:
        with pytest.raises(AsrError):
            export_result({**result, 'alignment_status': state}, format_, segment_timestamps=True)
    for end in [float('nan'), float('inf'), -.1, 5.]:
        malformed = copy.deepcopy(result)
        malformed['segments'][0]['end'] = end
        with pytest.raises(AsrError):
            export_result(malformed, format_, segment_timestamps=True)
    with pytest.raises(AsrError):
        export_result({**result, 'text': 'hello world missing words'}, format_, segment_timestamps=True)
    fallback = copy.deepcopy(result)
    fallback['segments'][0]['timestamp_source'] = 'vad_chunk_window'
    with pytest.raises(AsrError):
        export_result(fallback, format_, segment_timestamps=True)


@pytest.mark.parametrize('action', ['cancel', 'delete'])
async def test_removed_prefetched_owner_releases_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str,
) -> None:
    first_started, first_release, preparing, draining, decoder_release = (asyncio.Event() for _ in range(5))
    async def first() -> None:
        first_started.set()
        await first_release.wait()
    async def decode(source: Path, output: Path, **kwargs: Any) -> None:
        output.write_bytes(b'partial')
        preparing.set()
        try:
            await asyncio.Event().wait()
        finally:
            draining.set()
            await decoder_release.wait()
            output.unlink()
    monkeypatch.setattr('asr_server.workbench.service.normalize_audio_path_to_wav_async', decode)
    async with running(tmp_path, preprocessing_prefetch=True) as service:
        scheduler = service.app.state.job_manager.scheduler
        first_op = await scheduler.submit('first', first)
        await first_started.wait()
        try:
            job_id = await submit(service)
            await asyncio.wait_for(preparing.wait(), 2)
            cancellation = asyncio.create_task(getattr(service, action)('review-user', job_id))
            await asyncio.wait_for(draining.wait(), 2)
            assert not cancellation.done()
            decoder_release.set()
            await asyncio.wait_for(cancellation, 2)
            assert job_id not in scheduler.operations
            assert not service.active and not service.prepared_assets
            assert service.app.state.workspace_manager._reserved_upload_bytes == 0
            assert service.app.state.workspace_manager.pending_disk_bytes(service.store.root) == 0
            if action == 'delete':
                assert not service.job_dir(job_id).exists()
            else:
                assert (await service.store.run(service.store.job, job_id))['status'] == 'cancelled'
        finally:
            decoder_release.set()
            first_release.set()
            await first_op.future


async def test_moss_subtitle_route_supports_new_and_saved_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def transcribe(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {'text': 'hello', 'duration': 1., 'timestamps': [],
                'segments': [{'text': 'hello', 'start': .1, 'end': .9, 'speaker': 'S01'}]}
    monkeypatch.setattr('asr_server.workbench.service.run_transcription_path', transcribe)
    async with running(tmp_path, enable_moss=True) as service:
        user = await service.store.run(service.store.create_user, 'subtitle-user', 'subtitle-password')
        job_id = await submit(service, owner=user['id'], model='moss-transcribe-diarize-0.9b')
        await asyncio.wait_for(service.app.state.job_manager.scheduler.operations[job_id].future, 3)
        job = await service.store.run(service.store.job, job_id)
        assert job['result']['alignment_status'] == 'segment_timed'
        assert job['result']['timestamp_source'] == 'model_segments'
        assert job['result']['timestamps'] == []
        async with AsyncClient(transport=ASGITransport(app=service.app), base_url='http://test') as client:
            assert (await client.post('/api/auth/login', json={'username': 'subtitle-user', 'password': 'subtitle-password'})).status_code == 200
            for saved_status in ['segment_timed', 'unavailable']:
                job['result']['alignment_status'] = saved_status
                await service.store.run(service.store.save_version, job)
                for format_ in ['srt', 'vtt']:
                    response = await client.get(f'/api/jobs/{job_id}/export?format={format_}&version=1')
                    assert response.status_code == 200, response.text
                    assert 'hello' in response.text
