from __future__ import annotations

import asyncio
from pathlib import Path

from asr_server.config import Settings
from asr_server.jobs import JobManager
from asr_server.lifecycle import ModelLifecycleManager
from asr_server.adapters.mock import MockAsrAdapter
from asr_server.registry import default_models
from asr_server.scheduler import OperationScheduler


async def test_mixed_operation_fifo_and_pending_cancellation() -> None:
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: MockAsrAdapter(), idle_unload_seconds=0)
    queue = JobManager(manager, Settings())
    entered, release = asyncio.Event(), asyncio.Event()
    order: list[str] = []
    active = peak = 0
    async def infer(name: str) -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        order.append(name)
        if name == 'legacy':
            entered.set()
            await release.wait()
        await asyncio.sleep(0)
        active -= 1
    await queue.enqueue_external('job_legacy', lambda: infer('legacy'))
    await entered.wait()
    sync = asyncio.create_task(queue.scheduler.wait('sync', lambda: infer('sync')))
    # Wait for the sync operation to be submitted, without relying on wall time.
    async with queue.scheduler.condition:
        await queue.scheduler.condition.wait_for(lambda: 'sync' in queue.scheduler.operations)
    await queue.enqueue_external('wb_job', lambda: infer('workbench'))
    await queue.enqueue_external('align:wb_job', lambda: infer('alignment'))
    cancelled = asyncio.create_task(queue.scheduler.wait('abandoned', lambda: infer('abandoned')))
    async with queue.scheduler.condition:
        await queue.scheduler.condition.wait_for(lambda: 'abandoned' in queue.scheduler.operations)
    cancelled.cancel()
    try:
        await cancelled
    except asyncio.CancelledError:
        pass
    release.set()
    await sync
    alignment = queue.scheduler.operations.get('align:wb_job')
    if alignment:
        await alignment.future
    await queue.shutdown()
    await manager.shutdown()
    assert order == ['legacy', 'sync', 'workbench', 'alignment']
    assert peak == 1


async def test_prefetch_is_one_cpu_successor_and_cancel_disposes(tmp_path: Path) -> None:
    scheduler = OperationScheduler(5, prefetch=True)
    first_running, finish_first = asyncio.Event(), asyncio.Event()
    preparing, hold_preparation = asyncio.Event(), asyncio.Event()
    touched: list[str] = []
    asset = tmp_path / 'prepared'
    async def first() -> None:
        first_running.set()
        await finish_first.wait()
    async def prepare() -> None:
        asset.write_bytes(b'bounded reservation fixture')
        preparing.set()
        await hold_preparation.wait()
    async def dispose() -> None:
        asset.unlink(missing_ok=True)
    async def second() -> None:
        touched.append('second')
    async def third_prepare() -> None:
        touched.append('third-prepared')
    a = await scheduler.submit('a', first)
    await first_running.wait()
    await scheduler.submit('b', second, prepare=prepare, dispose=dispose)
    await preparing.wait()
    await scheduler.submit('c', second, prepare=third_prepare)
    assert touched == [] and asset.exists()
    assert await scheduler.cancel('b')
    assert not asset.exists()
    finish_first.set()
    await a.future
    c = scheduler.operations.get('c')
    if c:
        await c.future
    async def abort() -> None:
        raise AssertionError('idle scheduler should not abort workers')
    await scheduler.shutdown(1, abort)
    assert touched == ['third-prepared', 'second']


async def test_business_cancellation_keeps_worker_serving_already_queued_operations() -> None:
    scheduler = OperationScheduler(3)
    entered, release, next_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def cancelled() -> None:
        entered.set()
        await release.wait()
        raise asyncio.CancelledError
    async def following() -> None:
        next_finished.set()
    a = await scheduler.submit("cancelled", cancelled)
    await entered.wait()
    b = await scheduler.submit("following", following)
    release.set()
    await asyncio.wait_for(next_finished.wait(), 1)
    await b.future
    assert a.future.cancelled()
    async def abort() -> None:
        raise AssertionError("unexpected abort")
    await scheduler.shutdown(1, abort)


async def test_cancelled_prefetch_retains_preparation_slot_through_decoder_and_lease_cleanup() -> None:
    scheduler = OperationScheduler(5, prefetch=True)
    running, finish = asyncio.Event(), asyncio.Event()
    b_preparing, b_draining, allow_decoder_stop = asyncio.Event(), asyncio.Event(), asyncio.Event()
    disposing, allow_lease_release, c_preparing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    disposals: list[str] = []
    async def first() -> None:
        running.set()
        await finish.wait()
    async def prepare_b() -> None:
        b_preparing.set()
        try:
            await asyncio.Event().wait()
        finally:
            b_draining.set()
            await allow_decoder_stop.wait()
    async def dispose_b() -> None:
        disposals.append("b")
        disposing.set()
        await allow_lease_release.wait()
    async def prepare_c() -> None:
        c_preparing.set()
    async def noop() -> None:
        pass
    a = await scheduler.submit("a", first)
    await running.wait()
    await scheduler.submit("b", noop, prepare=prepare_b, dispose=dispose_b)
    await b_preparing.wait()
    c = await scheduler.submit("c", noop, prepare=prepare_c)
    cancellation = asyncio.create_task(scheduler.cancel("b"))
    await b_draining.wait()
    await scheduler.submit("d", noop)
    # The new head has been offered prefetch, but cannot acquire resources yet.
    await asyncio.sleep(0)
    assert not c_preparing.is_set()
    allow_decoder_stop.set()
    await disposing.wait()
    assert not c_preparing.is_set()
    allow_lease_release.set()
    assert await cancellation
    await asyncio.wait_for(c_preparing.wait(), 1)
    finish.set()
    await a.future
    await c.future
    await scheduler.shutdown(1, noop)
    assert disposals == ["b"]
