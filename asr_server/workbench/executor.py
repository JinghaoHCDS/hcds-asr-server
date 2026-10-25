"""Small bounded executors; a cancelled waiter never releases a running slot."""
from __future__ import annotations

import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, ParamSpec, TypeVar

from asr_server.errors import AsrError


P = ParamSpec("P")
T = TypeVar("T")


class BoundedExecutor:
    def __init__(self, workers: int, waiting: int, name: str, *, drain_on_cancel: bool = False) -> None:
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)
        self.slots = threading.BoundedSemaphore(workers + waiting)
        self.name = name
        self.drain_on_cancel = drain_on_cancel

    async def run(self, operation: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        if not self.slots.acquire(blocking=False):
            raise AsrError(429, "workbench_busy", "workbench is busy; try again shortly")

        context = contextvars.copy_context()
        def execute() -> T:
            try:
                return context.run(operation, *args, **kwargs)
            finally:
                self.slots.release()

        try:
            future = self.pool.submit(execute)
        except BaseException:
            self.slots.release()
            raise
        # The work owns its slot until it ends, even if the HTTP client leaves.
        wrapped = asyncio.wrap_future(future)
        # A disconnected waiter may never inspect the result. Consume failures
        # here as well; awaiting the future still raises them to live callers.
        wrapped.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            if self.drain_on_cancel:
                # Repository mutations must finish before the caller releases
                # an upload/state lock; a late thread must not revive old state.
                while not wrapped.done():
                    try:
                        await asyncio.shield(wrapped)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
            raise

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=False)
