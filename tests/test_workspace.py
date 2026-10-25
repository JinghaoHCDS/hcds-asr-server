from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path

import pytest

from asr_server.audio.workspace import AudioWorkspace, WorkspaceManager, validate_workspace_limits
from asr_server.config import Settings
from asr_server.errors import AsrError


def test_workspace_file_count_limit_is_enforced(tmp_path: Path) -> None:
    for index in range(3):
        (tmp_path / f"file-{index}").write_bytes(b"x")

    with pytest.raises(AsrError) as exc_info:
        validate_workspace_limits(tmp_path, Settings(max_workspace_files=2))

    assert exc_info.value.code == "storage_unavailable"
    assert exc_info.value.details["file_count"] == 3


def test_workspace_byte_limit_is_enforced(tmp_path: Path) -> None:
    (tmp_path / "large").write_bytes(b"x" * (1024 * 1024 + 1))

    with pytest.raises(AsrError) as exc_info:
        validate_workspace_limits(tmp_path, Settings(max_workspace_mb=1))

    assert exc_info.value.details["max_workspace_bytes"] == 1024 * 1024


async def test_cancelled_cleanup_waits_for_deletion_and_shared_budget_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    manager = WorkspaceManager(Settings())
    await manager._reserve(7, root)
    upload_path = root / "upload"
    upload_path.write_bytes(b"sample!")
    workspace = AudioWorkspace(root=root, upload_path=upload_path, size_bytes=7, _manager=manager)
    deletion_entered = asyncio.Event()
    second_caller_entered = asyncio.Event()
    allow_deletion = threading.Event()
    loop = asyncio.get_running_loop()
    original_rmtree = shutil.rmtree
    original_release = manager.release
    release_sizes: list[int] = []

    def controlled_rmtree(path: Path, ignore_errors: bool) -> None:
        loop.call_soon_threadsafe(deletion_entered.set)
        assert allow_deletion.wait(timeout=5), "test did not release controlled deletion"
        original_rmtree(path, ignore_errors)

    async def record_release(size_bytes: int) -> None:
        release_sizes.append(size_bytes)
        await original_release(size_bytes)

    async def second_cleanup() -> None:
        second_caller_entered.set()
        await workspace.cleanup()

    monkeypatch.setattr("asr_server.audio.workspace.shutil.rmtree", controlled_rmtree)
    monkeypatch.setattr(manager, "release", record_release)
    first = asyncio.create_task(workspace.cleanup())
    second: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(deletion_entered.wait(), timeout=5)
        first.cancel()
        second = asyncio.create_task(second_cleanup())
        await asyncio.wait_for(second_caller_entered.wait(), timeout=5)

        assert not first.done()
        assert not second.done()
        assert root.exists()
        assert manager._reserved_upload_bytes == 7
        allow_deletion.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, timeout=5)
        await asyncio.wait_for(second, timeout=5)

        assert not root.exists()
        assert workspace._cleaned
        assert manager._reserved_upload_bytes == 0
        assert release_sizes == [7]
        await workspace.cleanup()
        assert release_sizes == [7]
    finally:
        allow_deletion.set()
        await asyncio.gather(first, *([second] if second is not None else []), return_exceptions=True)


async def test_resource_lease_growth_counts_both_copies_and_releases_once_after_directory_removal(tmp_path: Path) -> None:
    manager = WorkspaceManager(Settings(max_spool_mb=1, min_free_disk_mb=0))
    root = tmp_path / "media"
    root.mkdir()
    parser = await manager.reserve_resource(512 * 1024, root)
    await parser.materialize(512 * 1024)
    workspace = await manager.reserve_resource(512 * 1024, root)
    assert manager._reserved_upload_bytes == 1024 * 1024
    with pytest.raises(AsrError) as error:
        await workspace.grow(1)
    assert error.value.code == "job_queue_full"
    assert manager._reserved_upload_bytes == 1024 * 1024
    root.rmdir()
    await parser.release()
    await workspace.release()
    await parser.release()
    assert manager._reserved_upload_bytes == 0
    assert not any(manager._pending_disk_bytes.values())


async def test_decode_materializes_lease_incrementally_and_returns_unused_reservation(tmp_path: Path) -> None:
    from asr_server.audio.preprocess import normalize_audio_path_to_wav_async
    from tests.helpers import make_wav

    manager = WorkspaceManager(Settings(min_free_disk_mb=0, max_spool_mb=1))
    source, output = tmp_path / "source.wav", tmp_path / "normalized.wav"
    source.write_bytes(make_wav(0.1))
    lease = await manager.reserve_resource(1024 * 1024, tmp_path)
    result = await normalize_audio_path_to_wav_async(source, output, max_output_bytes=1024 * 1024,
                                                    resource_lease=lease)
    actual = result.path.stat().st_size
    assert lease.pending_bytes == 1024 * 1024 - actual
    assert manager._pending_disk_bytes[tmp_path.stat().st_dev] == lease.pending_bytes
    await lease.release_unused()
    assert lease.size_bytes == actual
    assert lease.pending_bytes == 0
    assert manager._reserved_upload_bytes == actual
    await lease.release()
    assert manager._reserved_upload_bytes == 0
