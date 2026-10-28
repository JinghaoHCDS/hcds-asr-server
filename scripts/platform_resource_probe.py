"""Reproducible POSIX-only, CPU media resource probe. No models or network.

Creates sparse WAV inputs, actually decodes every frame through the production
preprocessor, splits file descriptors, and deletes each case's large files.
Each case is an independent process; the supervisor samples gateway-helper and
FFmpeg RSS separately with ps. Synthetic silence proves resources/coverage only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import resource
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.workbench_release import ReleaseError, require_generated_output


SAMPLE_RATE = 16_000
MAX_SECONDS = 21_600
OUTPUT_BUDGET = MAX_SECONDS * SAMPLE_RATE * 2 + 44
RSS_BUDGET = 256 * 1024**2
CASES = ("600", "3600", "21600", "overflow", "cancel")


def rss_bytes(value: float) -> int:
    return int(value) if sys.platform == "darwin" else int(value * 1024)


def sparse_wav(path: Path, frames: int) -> None:
    pcm_bytes = frames * 2
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + pcm_bytes, b"WAVE", b"fmt ",
                         16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, b"data", pcm_bytes)
    with path.open("wb") as audio:
        audio.write(header)
        audio.truncate(44 + pcm_bytes)


def sample_processes(gateway_pid: int) -> tuple[int | None, dict[int, int]]:
    try:
        completed = subprocess.run(["ps", "-axo", "pid=,ppid=,rss=,comm="], capture_output=True,
                                   text=True, check=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None, {}
    gateway = None
    decoders: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        pid, parent, rss = (int(value) for value in fields[:3])
        if pid == gateway_pid:
            gateway = rss * 1024
        if parent == gateway_pid and Path(fields[3]).name == "ffmpeg":
            decoders[pid] = rss * 1024
    return gateway, decoders


async def child_case(case: str, workspace: Path) -> dict[str, Any]:
    from asr_server.audio.preprocess import normalize_audio_path_to_wav_async
    from asr_server.audio.splitter import split_audio_path
    from asr_server.audio.workspace import WorkspaceManager
    from asr_server.config import Settings
    from asr_server.errors import AsrError

    root = Path(tempfile.mkdtemp(prefix=f"case_{case}_", dir=workspace))
    source, output = root / "sparse.wav", root / "normalized.wav"
    frames = (int(case) if case.isdecimal() else MAX_SECONDS) * SAMPLE_RATE
    if case == "overflow":
        frames += 1
    sparse_wav(source, frames)
    input_stat = source.stat()
    baseline = rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    baseline_rss, _ = sample_processes(os.getpid())
    manager = WorkspaceManager(Settings(max_spool_mb=1024, min_free_disk_mb=0))
    lease = await manager.reserve_resource(OUTPUT_BUDGET, root)
    cancelled = threading.Event()
    decoder_processes: list[asyncio.subprocess.Process] = []
    real_create = asyncio.create_subprocess_exec
    real_unlink = Path.unlink
    removed_output_sizes: list[int] = []

    async def create(*arguments: Any, **options: Any) -> asyncio.subprocess.Process:
        process = await real_create(*arguments, **options)
        if arguments and arguments[0] == "ffmpeg":
            decoder_processes.append(process)
        return process
    def unlink(path: Path, missing_ok: bool = False) -> None:
        if path == output and path.exists():
            removed_output_sizes.append(path.stat().st_size)
        real_unlink(path, missing_ok=missing_ok)
    async def cancel_after_output() -> None:
        while not output.exists() or output.stat().st_size <= 44:
            await asyncio.sleep(0.001)
        cancelled.set()

    asyncio.create_subprocess_exec = create
    setattr(Path, "unlink", unlink)
    cancellation = asyncio.create_task(cancel_after_output()) if case == "cancel" else None
    started = time.perf_counter()
    result: dict[str, Any] = {"case": case, "input_audio_seconds": frames / SAMPLE_RATE,
                              "input_logical_bytes": input_stat.st_size,
                              "input_allocated_bytes": input_stat.st_blocks * 512,
                              "output_budget_bytes": OUTPUT_BUDGET,
                              "gateway_baseline_ru_maxrss_bytes": baseline,
                              "gateway_baseline_sampled_rss_bytes": baseline_rss}
    try:
        decoded = await normalize_audio_path_to_wav_async(source, output, max_output_bytes=OUTPUT_BUDGET,
                                                          max_duration_seconds=MAX_SECONDS, timeout_seconds=90,
                                                          cancel_event=cancelled, resource_lease=lease)
        normalized_at = time.perf_counter()
        await lease.release_unused()
        split = await asyncio.to_thread(split_audio_path, decoded.path, split_strategy="fixed",
                                         max_chunk_seconds=120.0, overlap_seconds=2.0)
        intervals = [(chunk.start, chunk.end) for chunk in split.chunks]
        cursor = 0.0
        coverage = bool(intervals)
        for start, end in intervals:
            coverage = coverage and start <= cursor + 1 / SAMPLE_RATE and end > start
            cursor = max(cursor, end)
        coverage = coverage and abs(cursor - frames / SAMPLE_RATE) < 1 / SAMPLE_RATE
        actual_bytes = output.stat().st_size
        result.update(status="completed", normalized_frames=decoded.asset.frame_count if decoded.asset else None,
                      output_bytes=actual_bytes, decode_seconds=normalized_at - started,
                      split_seconds=time.perf_counter() - normalized_at, chunks=len(intervals),
                      coverage_start=intervals[0][0], coverage_end=intervals[-1][1], full_coverage=coverage,
                      exact_output_size=actual_bytes == frames * 2 + 44,
                      descriptor_only=all(not hasattr(chunk, "audio") for chunk in split.chunks),
                      logical_bytes_after_unused_release=lease.size_bytes)
        if not coverage or actual_bytes != frames * 2 + 44 or actual_bytes > OUTPUT_BUDGET:
            raise RuntimeError("normalization lost coverage or violated the exact output boundary")
        if case in {"overflow", "cancel"}:
            raise RuntimeError("boundary input was incorrectly accepted")
    except asyncio.CancelledError:
        if case != "cancel":
            raise
        result.update(status="cancelled", partial_removed=not output.exists())
    except AsrError as error:
        if case != "overflow" or error.code != "duration_limit_exceeded":
            raise
        result.update(status="rejected", error_code=error.code, partial_removed=not output.exists(),
                      largest_partial_before_delete_bytes=max(removed_output_sizes, default=0))
    finally:
        if cancellation is not None:
            cancellation.cancel()
            await asyncio.gather(cancellation, return_exceptions=True)
        asyncio.create_subprocess_exec = real_create
        setattr(Path, "unlink", real_unlink)
        await lease.release()
        cleanup_started = time.perf_counter()
        shutil.rmtree(root)
        result.update(cleanup_seconds=time.perf_counter() - cleanup_started,
                      operation_seconds=time.perf_counter() - started,
                      gateway_ru_maxrss_bytes=rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                      children_ru_maxrss_bytes=rss_bytes(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
                      decoder_exit_codes=[process.returncode for process in decoder_processes],
                      all_decoders_reaped=all(process.returncode is not None for process in decoder_processes),
                      logical_reserved_after_cleanup=manager._reserved_upload_bytes,
                      physical_pending_after_cleanup=sum(manager._pending_disk_bytes.values()),
                      workspace_removed=not root.exists())
    if (not result["workspace_removed"] or not result["all_decoders_reaped"]
            or result["logical_reserved_after_cleanup"] or result["physical_pending_after_cleanup"]
            or (case in {"overflow", "cancel"} and not result["partial_removed"])):
        raise RuntimeError("resource cleanup failed")
    return result


def supervise(case: str, workspace: Path, sample_ms: float) -> dict[str, Any]:
    process = subprocess.Popen([sys.executable, "-m", "scripts.platform_resource_probe", "--case", case,
                                "--workspace", str(workspace)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    started = time.perf_counter()
    gateway_peak = 0
    decoder_peaks: dict[int, int] = {}
    samples = 0
    while process.poll() is None:
        gateway, decoders = sample_processes(process.pid)
        gateway_peak = max(gateway_peak, gateway or 0)
        for pid, value in decoders.items():
            decoder_peaks[pid] = max(decoder_peaks.get(pid, 0), value)
        samples += 1
        if time.perf_counter() - started > 120:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            raise RuntimeError(f"resource case {case} exceeded 120 seconds")
        time.sleep(sample_ms / 1000)
    stdout, stderr = process.communicate(timeout=5)
    if process.returncode:
        raise RuntimeError(f"resource case {case} failed (exit {process.returncode}): {stderr[-2000:]}")
    result: dict[str, Any] = json.loads(stdout)
    baseline = result["gateway_baseline_sampled_rss_bytes"] or result["gateway_baseline_ru_maxrss_bytes"]
    gateway_peak = max(gateway_peak, result["gateway_ru_maxrss_bytes"])
    increment = max(0, gateway_peak - baseline)
    result.update(supervisor_wall_seconds=time.perf_counter() - started,
                  gateway_peak_rss_bytes=gateway_peak, gateway_peak_increment_bytes=increment,
                  gateway_increment_within_256_mib=increment <= RSS_BUDGET,
                  decoder_sampled_peak_rss_bytes=max(decoder_peaks.values(), default=None),
                  decoder_sampled_by_pid={str(pid): value for pid, value in decoder_peaks.items()},
                  rss_samples=samples, decoder_measurement="ps samples; sampling can miss a shorter-lived peak")
    if not decoder_peaks:
        result["decoder_measurement_missing"] = "decoder finished between samples; children ru_maxrss is separate, unattributed evidence"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("data/platform-validation-20261025/resource"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/platform/resource-probe.json"))
    parser.add_argument("--sample-ms", type=float, default=25.0)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("this resource probe requires POSIX sparse files and ps")
    if args.sample_ms <= 0 or args.sample_ms > 1000:
        parser.error("sample interval must be in (0, 1000] ms")
    try:
        require_generated_output(args.workspace)
        require_generated_output(args.output)
    except ReleaseError as exc:
        parser.error(str(exc))
    args.workspace.mkdir(parents=True, exist_ok=True)
    if args.case:
        print(json.dumps(asyncio.run(child_case(args.case, args.workspace)), sort_keys=True))
        return
    ffmpeg = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    results = [supervise(case, args.workspace, args.sample_ms) for case in CASES]
    report = {"schema_version": 1, "utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "python": platform.python_version(), "ffmpeg": ffmpeg,
              "source": "sparse zero PCM16 WAV; no real speech or model inference",
              "scope": "gateway audio helper control process; not the complete HTTP/SQLite server RSS",
              "memory_method": "independent child per case; self ru_maxrss plus supervisor ps samples; FFmpeg sampled separately",
              "sampling_interval_ms": args.sample_ms, "rss_increment_budget_bytes": RSS_BUDGET,
              "exact_six_hour_output_budget_bytes": OUTPUT_BUDGET, "cases": results,
              "not_validated": ["ASR text accuracy", "GPU/CUDA or model memory", "HTTP mixed-load latency",
                                "MOSS native long-form full-input memory", "WSL FFmpeg build behavior"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        os.chmod(args.output, 0o600)
        target.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(results),
                      "max_gateway_increment_mib": max(case["gateway_peak_increment_bytes"] for case in results) / 1024**2,
                      "all_within_budget": all(case["gateway_increment_within_256_mib"] for case in results)}, sort_keys=True))


if __name__ == "__main__":
    main()
