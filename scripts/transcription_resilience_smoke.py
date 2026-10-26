"""Isolated worker-exit recovery exercise; never kills an online service worker.

Uses one new data directory and a public fixture. --real-gpu requires the
existing WSL runtime and an idle GPU; no dependencies or models are installed.
Reports behavior and timing only, not transcript accuracy or long-run stability.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient

from asr_server.config import Settings
from asr_server.main import create_app
from asr_server.workbench.service import WorkbenchService


async def wait_for_job(service: WorkbenchService, job_id: str, *, waiting: bool = False) -> dict[str, Any]:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        job = await service.store.run(service.store.job, job_id)
        phase = job.get('progress', {}).get('phase')
        if waiting and phase == 'retry_wait' and job_id not in service.active:
            return job
        if job['status'] in {'completed', 'failed', 'cancelled'} and job_id not in service.active:
            return job
        await asyncio.sleep(.02)
    raise RuntimeError('isolated task deadline exceeded')


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.data_dir.exists():
        raise RuntimeError('choose a new isolated data directory')
    args.data_dir.mkdir(parents=True, mode=0o700)
    settings = replace(Settings(), adapter='qwen' if args.real_gpu else 'mock',
        workbench_data_dir=args.data_dir / 'workbench', enable_qwen_alignment=True,
        min_free_disk_mb=0, idle_unload_seconds=0, log_dir=args.data_dir / 'logs')
    app = create_app(settings)
    service: WorkbenchService = app.state.workbench
    password = secrets.token_urlsafe(24)
    owner = service.store.create_user('resilience-smoke', password, True)['id']
    report: dict[str, Any] = {'mode': 'real_gpu' if args.real_gpu else 'mock', 'started_at': time.time(),
        'scope': 'isolated data and owned worker only', 'quality_acceptance': False}
    async def submit(key: str) -> str:
        audio = args.audio.read_bytes()
        upload = await service.create_upload(owner, 'public-fixture.wav', len(audio))
        await service.append_upload(owner, upload['id'], 0, audio)
        return str((await service.complete_upload(owner, upload['id'], {'idempotency_key': key}))['id'])
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url='http://resilience.local') as client:
            await client.post('/api/auth/login', json={'username': 'resilience-smoke', 'password': password})
            assert (await client.get('/readyz')).status_code == 200
            if args.real_gpu:
                # Preload only in our app. The victim PID belongs to this
                # manager and cannot refer to an online service process.
                await app.state.manager.load_model(settings.default_model, backend='auto')
                victim = await submit('victim')
                runtime = app.state.manager.runtime_for(settings.default_model)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    transport = runtime.adapter._transport
                    if runtime.active_requests and transport.is_alive:
                        process = transport._process
                        assert process is not None and process.pid == transport.pid
                        report['killed_owned_pid'] = process.pid
                        process.kill()
                        break
                    await asyncio.sleep(.001)
                else:
                    raise RuntimeError('could not inject exit during an active owned call')
                waiting = await wait_for_job(service, victim, waiting=True)
                assert waiting['progress']['phase'] == 'retry_wait'
                assert waiting['recovery']['infrastructure_retries'] == 1
                report['worker_exit'] = {'error_code': waiting['error']['code'],
                    'worker_reaped': waiting['error']['details'].get('worker_reaped'),
                    'debits': waiting['recovery']['infrastructure_retries']}
                healthy = await submit('healthy')
                healthy_job = await wait_for_job(service, healthy)
                assert healthy_job['status'] == 'completed'
                recovered = await wait_for_job(service, victim)
                assert recovered['status'] == 'completed'
                report['recovery'] = {'victim': recovered['status'], 'healthy': healthy_job['status'],
                    'infrastructure_retries': recovered['recovery']['infrastructure_retries']}
                job_id = victim
            else:
                job_id = await submit('healthy')
                assert (await wait_for_job(service, job_id))['status'] == 'completed'
            for format_ in ('txt', 'json', 'srt', 'vtt'):
                response = await client.get(f'/api/jobs/{job_id}/export?format={format_}')
                allowed = {200} if args.real_gpu or format_ in {'txt', 'json'} else {200, 409}
                assert response.status_code in allowed, (format_, response.status_code)
            assert (await client.get(f'/api/jobs/{job_id}/audio', headers={'Range': 'bytes=0-43'})).status_code == 206
            await client.post('/api/admin/drain', json={'draining': True})
            assert (await client.get('/readyz')).status_code == 503
            assert (await client.get(f'/api/jobs/{job_id}/export?format=txt')).status_code == 200
            await client.post('/api/admin/drain', json={'draining': False})
            assert (await client.get('/readyz')).status_code == 200
            usage = (await client.get('/api/storage')).json()
            assert usage['temporary_bytes'] == usage['reserved_bytes'] == 0
            assert not service.operation_leases
            report.update({'storage': usage, 'readyz_drain_reads': 'passed', 'exports_range': 'passed'})
    assert not list((args.data_dir / 'workbench').rglob('*.wav'))
    report.update({'status': 'passed', 'elapsed_seconds': time.time() - report['started_at']})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--audio', type=Path, default=Path('test-fixtures/audio/test_short.wav'))
    parser.add_argument('--real-gpu', action='store_true')
    args = parser.parse_args()
    os.environ['ASR_TEMP_DIR'] = str(args.data_dir / 'temporary')
    report = asyncio.run(run(args))
    (args.data_dir / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'mode': report['mode'], 'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
