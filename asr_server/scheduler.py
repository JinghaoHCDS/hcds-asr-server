"""One bounded FIFO for all inference owners; no payload or permission storage."""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from asr_server.errors import AsrError
from asr_server.observability import ExecutionContext, bind_context, current_context, emit, safe_exception

_owns_operation: ContextVar[bool] = ContextVar("owns_scheduled_operation", default=False)


@dataclass
class ScheduledOperation:
    key: str
    sequence: int
    kind: str
    run: Callable[[], Awaitable[Any]]
    future: asyncio.Future[Any]
    context: ExecutionContext
    submitted_at: float
    prepare: Callable[[], Awaitable[None]] | None = None
    dispose: Callable[[], Awaitable[None]] | None = None
    preparation: asyncio.Task[None] | None = None
    disposal: asyncio.Task[None] | None = None
    cancel_requested: bool = False


class OperationScheduler:
    def __init__(self, capacity: int, *, prefetch: bool = False) -> None:
        self.capacity = capacity
        self.queue: deque[str] = deque()
        self.operations: dict[str, ScheduledOperation] = {}
        self.running: str | None = None
        self.worker: asyncio.Task[None] | None = None
        self.condition = asyncio.Condition()
        self.sequence = 0
        self.closed = False
        self.prefetch = prefetch
        self._preparation_slot = asyncio.Semaphore(1)

    async def start(self) -> None:
        if self.worker is None or self.worker.done():
            self.closed = False
            self.worker = asyncio.create_task(self._loop())

    async def submit(self, key: str, run: Callable[[], Awaitable[Any]], *, kind: str = "job",
                     context: ExecutionContext | None = None,
                     prepare: Callable[[], Awaitable[None]] | None = None,
                     dispose: Callable[[], Awaitable[None]] | None = None) -> ScheduledOperation:
        await self.start()
        async with self.condition:
            if self.closed:
                raise AsrError(503, "server_stopping", "inference scheduler is stopping")
            if key in self.operations:
                return self.operations[key]
            if len(self.operations) >= self.capacity:
                raise AsrError(429, "job_queue_full", "too many queued or running inference operations")
            self.sequence += 1
            future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            # Durable owners may only inspect their own store, never this Future.
            future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
            operation = ScheduledOperation(key, self.sequence, kind, run, future,
                                           context or current_context(), perf_counter(), prepare, dispose)
            self.operations[key] = operation
            self.queue.append(key)
            with bind_context(operation.context):
                emit("job.queued", sequence=operation.sequence, operation=kind)
            self.condition.notify_all()
            current = self.operations.get(self.running) if self.running else None
            if current is not None and (current.preparation is None or current.preparation.done()):
                self._prefetch_next()
            return operation

    async def wait(self, key: str, run: Callable[[], Awaitable[Any]], *, kind: str = "sync",
                   on_cancel: Callable[[], None] | None = None) -> Any:
        if _owns_operation.get():
            return await run()
        operation = await self.submit(key, run, kind=kind)
        try:
            return await asyncio.shield(operation.future)
        except asyncio.CancelledError:
            if on_cancel:
                on_cancel()
            removed = await self.cancel(key)
            if not removed:
                # Running model calls retain ownership until their safe boundary.
                while not operation.future.done():
                    try:
                        await asyncio.shield(operation.future)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
            raise

    async def cancel(self, key: str) -> bool:
        async with self.condition:
            if key not in self.queue:
                return False
            self.queue.remove(key)
            operation = self.operations.pop(key)
            operation.cancel_requested = True
            operation.future.cancel()
            self.condition.notify_all()
        if operation.preparation is not None:
            operation.preparation.cancel()
            await asyncio.gather(operation.preparation, return_exceptions=True)
        await self._dispose(operation)
        return True

    async def _dispose(self, operation: ScheduledOperation) -> None:
        callback = operation.dispose
        if callback is None:
            return
        if operation.disposal is None:
            async def dispose() -> None:
                with bind_context(operation.context):
                    await callback()
            operation.disposal = asyncio.create_task(dispose())
        interrupted = False
        while True:
            try:
                await asyncio.shield(operation.disposal)
                break
            except asyncio.CancelledError:
                if operation.disposal.cancelled():
                    raise
                interrupted = True
        if interrupted:
            raise asyncio.CancelledError

    async def _prepare(self, operation: ScheduledOperation) -> None:
        if operation.prepare is not None:
            # A removed successor still owns this slot until its decoder and
            # reservations finish cancellation cleanup. A new queue head must
            # not acquire another lease during that interval.
            async with self._preparation_slot:
                failed = False
                with bind_context(operation.context):
                    try:
                        await operation.prepare()
                    except BaseException:
                        failed = True
                        raise
                    finally:
                        task = asyncio.current_task()
                        if failed or operation.cancel_requested or (task is not None and task.cancelling()):
                            await self._dispose(operation)

    def _prefetch_next(self) -> None:
        if self.prefetch and self.queue:
            following = self.operations[self.queue[0]]
            if following.prepare is not None and following.preparation is None:
                following.preparation = asyncio.create_task(self._prepare(following))

    def position(self, key: str) -> int:
        return list(self.queue).index(key) + 1 if key in self.queue else 0

    async def shutdown(self, timeout: float, abort: Callable[[], Awaitable[None]]) -> None:
        self.closed = True
        for key in list(self.queue):
            await self.cancel(key)
        async with self.condition:
            self.condition.notify_all()
        if self.worker:
            done, _ = await asyncio.wait({self.worker}, timeout=timeout)
            if not done:
                await abort()
                self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass
            self.worker = None

    async def _loop(self) -> None:
        while True:
            async with self.condition:
                await self.condition.wait_for(lambda: self.closed or bool(self.queue))
                if self.closed and not self.queue:
                    return
                key = self.queue.popleft()
                operation = self.operations[key]
                self.running = key
            started = perf_counter()
            token = _owns_operation.set(True)
            with bind_context(operation.context):
                emit("job.started", sequence=operation.sequence,
                     queue_wait_ms=(started - operation.submitted_at) * 1000, operation=operation.kind)
                try:
                    if operation.prepare:
                        if operation.preparation is None:
                            operation.preparation = asyncio.create_task(self._prepare(operation))
                        await operation.preparation
                    # Only one successor is prepared; all inference remains FIFO.
                    self._prefetch_next()
                    value = await operation.run()
                    if not operation.future.done():
                        operation.future.set_result(value)
                except asyncio.CancelledError:
                    operation.future.cancel()
                    emit("job.cancelled")
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise
                except Exception as exc:
                    emit("job.failed", error_code=getattr(exc, "code", "operation_failed"),
                         diagnostics=safe_exception(exc))
                    if not operation.future.done():
                        operation.future.set_exception(exc)
                finally:
                    emit("operation.finished", duration_ms=(perf_counter() - started) * 1000)
                    self.operations.pop(key, None)
                    self.running = None
                    _owns_operation.reset(token)
