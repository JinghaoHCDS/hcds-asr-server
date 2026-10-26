from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from time import perf_counter
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import UploadFile

from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.observability import emit, safe_exception


UPLOAD_BLOCK_BYTES = 1024 * 1024


def service_temporary_root() -> Path:
    suffix = str(os.getuid()) if hasattr(os, "getuid") else str(os.getpid())
    root = Path(os.getenv("ASR_TEMP_DIR", str(Path(tempfile.gettempdir()) / f"asr-server-{suffix}")))
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return root


@dataclass
class ResourceLease:
    """One shared logical reservation, released exactly once.

    ``materialize`` converts outstanding disk reservations into existing bytes:
    disk_usage already sees those bytes, so they must not be deducted twice.
    A parser may retain its memory spool as pending until it actually rolls.
    """

    _manager: WorkspaceManager
    root: Path
    size_bytes: int = 0
    pending_bytes: int = 0
    _released: bool = False
    _device: int = field(init=False, repr=False)
    _persistent_key: str | None = field(default=None, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self._device = self.root.stat().st_dev

    async def grow(self, size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError("reservation growth must be non-negative")
        async with self._lock:
            if self._released:
                raise RuntimeError("resource lease is already released")
            await self._manager._reserve_resource_bytes(size_bytes, self.root)
            self.size_bytes += size_bytes
            self.pending_bytes += size_bytes

    async def materialize(self, size_bytes: int) -> None:
        async with self._lock:
            if self._released:
                return
            amount = min(max(size_bytes, 0), self.pending_bytes)
            await self._manager._release_pending(amount, self._device)
            self.pending_bytes -= amount

    async def grow_persistent(self, size_bytes: int) -> None:
        """Transfer additional, already admitted persistent bytes to this lease."""
        if size_bytes < 0:
            raise ValueError("reservation growth must be non-negative")
        async with self._lock:
            if self._released or self._persistent_key is None:
                raise RuntimeError("persistent resource lease is unavailable")
            await self._manager.grow_handoff(self, size_bytes)
            self.size_bytes += size_bytes
            self.pending_bytes += size_bytes

    async def release_unused(self) -> None:
        """After EOF, retain actual bytes and return the unused reservation."""
        async with self._lock:
            if self._released or not self.pending_bytes:
                return
            await self._manager._release_resource(self.pending_bytes, self.pending_bytes, self._device)
            self.size_bytes -= self.pending_bytes
            self.pending_bytes = 0

    async def release(self) -> None:
        async with self._lock:
            if self._released:
                return
            await self._manager._release_resource(self.size_bytes, self.pending_bytes, self._device,
                                                  persistent_key=self._persistent_key)
            self.size_bytes = self.pending_bytes = 0
            self._released = True

    async def __aenter__(self) -> ResourceLease:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.release()


@dataclass
class AudioWorkspace:
    root: Path
    upload_path: Path
    size_bytes: int
    _manager: WorkspaceManager
    _cleaned: bool = False
    resource_leases: list[ResourceLease] = field(default_factory=list, repr=False)
    _cleanup_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    async def cleanup(self) -> None:
        if self._cleaned:
            return
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup())
        cleanup_task = self._cleanup_task
        # All callers join the same deletion and budget release, including
        # cancelled callers; terminal job states must not precede file cleanup.
        cancelled = False
        while True:
            try:
                await asyncio.shield(cleanup_task)
                break
            except asyncio.CancelledError:
                if cleanup_task.cancelled():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _cleanup(self) -> None:
        started = perf_counter()
        released = self.size_bytes + sum(lease.size_bytes for lease in self.resource_leases)
        try:
            await asyncio.to_thread(shutil.rmtree, self.root, True)
            await self._manager.release(self.size_bytes)
            for lease in self.resource_leases:
                await lease.release()
            self.resource_leases.clear()
            self._cleaned = True
        except BaseException as exc:
            emit("cleanup.failed", diagnostics=safe_exception(exc))
            raise
        emit("cleanup.completed", released_bytes=released, duration_ms=(perf_counter() - started) * 1000)


class WorkspaceManager:
    """Own request/job workspaces and a process-wide upload spool budget."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._reserved_upload_bytes = 0
        self._pending_disk_bytes: dict[int, int] = {}
        # Persistent promises affect physical free space, never the transient
        # spool quota. Workbench source/PCM bytes already on disk are excluded.
        self._persistent_disk_bytes: dict[int, dict[str, int]] = {}
        self._persistent_handoffs: dict[int, dict[str, tuple[int, int]]] = {}
        self._persistent_totals: dict[int, int] = {}

    def _persistent_total(self, device: int, declarations: dict[str, int]) -> int:
        handoffs = self._persistent_handoffs.get(device, {})
        total = 0
        for key, promised in declarations.items():
            handoff = handoffs.get(key)
            if handoff is None:
                total += promised
            else:
                original, transferred = handoff
                # A DB decrement can mean the decoder's bytes have become a
                # retained asset. Stop excluding that materialized portion.
                credit = max(0, transferred - max(original - promised, 0))
                total += max(0, promised - credit)
        return total

    def pending_disk_bytes(self, root: Path) -> int:
        """Cached snapshot; reservations themselves always recheck under lock."""
        device = root.stat().st_dev
        return self._pending_disk_bytes.get(device, 0) + self._persistent_totals.get(device, 0)

    async def set_persistent_reservations(self, root: Path, declarations: dict[str, int], *,
                                         check_growth: bool = True) -> None:
        if any(value < 0 for value in declarations.values()):
            raise ValueError("persistent disk reservations must be non-negative")
        async with self._lock:
            device = root.stat().st_dev
            pending = self._persistent_total(device, declarations)
            previous = self._persistent_totals.get(device, 0)
            # A decrease must remain possible on an already full filesystem.
            if check_growth and pending > previous:
                free = shutil.disk_usage(root).free
                if free - self._pending_disk_bytes.get(device, 0) - pending < self._settings.min_free_disk_mb * 1024 * 1024:
                    raise AsrError(503, "storage_unavailable", "insufficient shared disk space")
            self._persistent_disk_bytes[device] = dict(declarations)
            self._persistent_totals[device] = pending

    async def handoff_persistent(self, root: Path, key: str, size_bytes: int) -> ResourceLease:
        """Atomically transfer a persistent promise into a decoder lease."""
        if size_bytes < 0:
            raise ValueError("reservation must be non-negative")
        lease = ResourceLease(self, root, _persistent_key=key)
        async with self._lock:
            device = root.stat().st_dev
            declarations = self._persistent_disk_bytes.get(device, {})
            promised = declarations.get(key, 0)
            handoffs = self._persistent_handoffs.setdefault(device, {})
            if key in handoffs or promised < size_bytes:
                raise RuntimeError("persistent decoder budget is unavailable or already transferred")
            handoffs[key] = (promised, size_bytes)
            previous = self._persistent_totals.get(device, 0)
            self._persistent_totals[device] = self._persistent_total(device, declarations)
            try:
                self._check_shared_budget(size_bytes, root)
            except BaseException:
                handoffs.pop(key)
                self._persistent_totals[device] = previous
                raise
            self._pending_disk_bytes[device] = self._pending_disk_bytes.get(device, 0) + size_bytes
            self._reserved_upload_bytes += size_bytes
            lease.size_bytes = lease.pending_bytes = size_bytes
        return lease

    async def store_upload(self, upload: UploadFile) -> AudioWorkspace:
        root = Path(tempfile.mkdtemp(prefix="asr_job_", dir=service_temporary_root()))
        (root / "owner.pid").write_text(str(os.getpid()), encoding="ascii")
        upload_path = root / "upload"
        total = 0
        try:
            with upload_path.open("wb") as output:
                while True:
                    block = await upload.read(UPLOAD_BLOCK_BYTES)
                    if not block:
                        break
                    await self._reserve(len(block), root)
                    total += len(block)
                    writing = asyncio.create_task(asyncio.to_thread(output.write, block))
                    try:
                        await asyncio.shield(writing)
                    except asyncio.CancelledError:
                        # Do not close the file while a write still owns it.
                        await writing
                        raise
            return AudioWorkspace(root=root, upload_path=upload_path, size_bytes=total, _manager=self)
        except BaseException:
            await AudioWorkspace(root, upload_path, total, self).cleanup()
            raise

    async def grow_handoff(self, lease: ResourceLease, size_bytes: int) -> None:
        async with self._lock:
            device, key = lease._device, lease._persistent_key
            assert key is not None
            handoffs = self._persistent_handoffs[device]
            original, transferred = handoffs[key]
            declarations = self._persistent_disk_bytes[device]
            promised = declarations.get(key, 0)
            # Materialized bytes have already left the database promise. The
            # next transfer needs only the lease's still-unwritten bytes and
            # its growth, rather than the original lease's full allocation.
            if promised < lease.pending_bytes + size_bytes:
                raise RuntimeError("persistent growth was not admitted")
            previous = self._persistent_totals[device]
            materialized = lease.size_bytes - lease.pending_bytes
            # Rebase on the current promise: growth may consume an existing
            # spare reservation or newly admitted bytes. Both must leave the
            # cached physical reservation equal to the outstanding promise.
            handoffs[key] = (promised + materialized, lease.size_bytes + size_bytes)
            self._persistent_totals[device] = self._persistent_total(device, declarations)
            try:
                self._check_shared_budget(size_bytes, lease.root)
            except BaseException:
                handoffs[key] = (original, transferred)
                self._persistent_totals[device] = previous
                raise
            self._pending_disk_bytes[device] = self._pending_disk_bytes.get(device, 0) + size_bytes
            self._reserved_upload_bytes += size_bytes

    async def reserve_resource(self, size_bytes: int, root: Path) -> ResourceLease:
        lease = ResourceLease(self, root)
        await lease.grow(size_bytes)
        return lease

    async def _reserve_resource_bytes(self, size_bytes: int, root: Path) -> None:
        async with self._lock:
            self._check_shared_budget(size_bytes, root)
            device = root.stat().st_dev
            self._pending_disk_bytes[device] = self._pending_disk_bytes.get(device, 0) + size_bytes
            self._reserved_upload_bytes += size_bytes

    async def _release_pending(self, size_bytes: int, device: int) -> None:
        async with self._lock:
            self._pending_disk_bytes[device] = max(self._pending_disk_bytes.get(device, 0) - size_bytes, 0)

    async def _release_resource(self, size_bytes: int, pending_bytes: int, device: int,
                                *, persistent_key: str | None = None) -> None:
        async with self._lock:
            self._pending_disk_bytes[device] = max(self._pending_disk_bytes.get(device, 0) - pending_bytes, 0)
            self._reserved_upload_bytes = max(self._reserved_upload_bytes - size_bytes, 0)
            if persistent_key is not None:
                self._persistent_handoffs.get(device, {}).pop(persistent_key, None)
                self._persistent_totals[device] = self._persistent_total(device, self._persistent_disk_bytes.get(device, {}))

    def _check_shared_budget(self, size_bytes: int, root: Path) -> None:
        max_spool_bytes = self._settings.max_spool_mb * 1024 * 1024
        if self._reserved_upload_bytes + size_bytes > max_spool_bytes:
            raise AsrError(429, "job_queue_full", "audio spool byte budget is exhausted",
                           {"max_spool_bytes": max_spool_bytes})
        free_bytes = shutil.disk_usage(root).free
        min_free_bytes = self._settings.min_free_disk_mb * 1024 * 1024
        device = root.stat().st_dev
        pending = self._pending_disk_bytes.get(device, 0) + self._persistent_totals.get(device, 0)
        if free_bytes - pending - size_bytes < min_free_bytes:
            raise AsrError(503, "storage_unavailable", "insufficient temporary disk space",
                           {"free_bytes": free_bytes, "min_free_disk_bytes": min_free_bytes})

    async def release(self, size_bytes: int) -> None:
        async with self._lock:
            self._reserved_upload_bytes = max(self._reserved_upload_bytes - size_bytes, 0)

    async def _reserve(self, size_bytes: int, root: Path) -> None:
        max_upload_bytes = self._settings.max_upload_mb * 1024 * 1024
        async with self._lock:
            projected = self._reserved_upload_bytes + size_bytes
            current_workspace_bytes = 0
            upload_path = root / "upload"
            if upload_path.exists():
                current_workspace_bytes = upload_path.stat().st_size
            if current_workspace_bytes + size_bytes > max_upload_bytes:
                raise AsrError(
                    413,
                    "audio_too_large",
                    "audio upload exceeds the server size limit",
                    {"max_upload_bytes": max_upload_bytes, "max_upload_mb": self._settings.max_upload_mb},
                )
            self._check_shared_budget(size_bytes, root)
            self._reserved_upload_bytes = projected


def validate_workspace_limits(root: Path, settings: Settings) -> None:
    files = [path for path in root.rglob("*") if path.is_file()]
    if len(files) > settings.max_workspace_files:
        raise AsrError(
            503,
            "storage_unavailable",
            "audio workspace file count limit exceeded",
            {"file_count": len(files), "max_workspace_files": settings.max_workspace_files},
        )
    unique: dict[tuple[int, int], int] = {}
    for path in files:
        stat = path.stat()
        unique[(stat.st_dev, stat.st_ino)] = stat.st_size
    size_bytes = sum(unique.values())
    max_bytes = settings.max_workspace_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise AsrError(
            503,
            "storage_unavailable",
            "audio workspace byte limit exceeded",
            {"workspace_bytes": size_bytes, "max_workspace_bytes": max_bytes},
        )


def remaining_workspace_output_bytes(root: Path, settings: Settings, *,
                                     exclude: tuple[Path, ...] = (),
                                     output_path: Path | None = None) -> int:
    """Remaining physical WAV budget; hard links to one inode count once."""
    seen: set[tuple[int, int]] = set()
    excluded = tuple(path.resolve() for path in exclude)
    replaced = output_path.resolve() if output_path is not None else None
    used = 0
    file_count = 0
    for path in root.rglob("*"):
        absolute = path.resolve()
        if absolute == replaced or any(absolute.is_relative_to(directory) for directory in excluded):
            continue
        if not path.is_file():
            continue
        file_count += 1
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino)
        if key not in seen:
            seen.add(key)
            used += stat.st_size
    if file_count >= settings.max_workspace_files:
        raise AsrError(503, "storage_unavailable", "audio workspace file count limit exceeded")
    return max(settings.max_workspace_mb * 1024 * 1024 - used, 0)
