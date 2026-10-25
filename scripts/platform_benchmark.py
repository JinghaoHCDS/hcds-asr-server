"""Repeatable CPU-only platform measurements against an explicit source tree.

Run baseline and current in separate processes with the same Python interpreter.
Outputs contain metrics and fingerprints only; synthetic fixtures are removed.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import importlib
import json
import logging
import math
import os
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def statistics_ms(samples: list[float]) -> dict[str, object]:
    return {"samples_ms": samples, "median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples)}


def query_benchmark(scratch: Path, repeats: int) -> list[dict[str, object]]:
    store_module = importlib.import_module("asr_server.workbench.store")
    service_module = importlib.import_module("asr_server.workbench.service")
    store = store_module.WorkbenchStore(scratch / "query-database")
    store.initialize()
    service = service_module.WorkbenchService.__new__(service_module.WorkbenchService)
    service.app = SimpleNamespace(state=SimpleNamespace(job_manager=SimpleNamespace(external_queue_position=lambda job_id: 0)))
    summaries = callable(getattr(store, "job_summaries", None))
    projection = "stored summary, page size 50" if summaries else "all full payloads, then public projection without result"
    fixture = {
        "owner_id": "benchmark-owner", "filename": "synthetic.wav", "status": "completed",
        "created_at": "2026-10-25T00:00:00Z", "started_at": "2026-10-25T00:00:00Z",
        "completed_at": "2026-10-25T00:00:01Z", "expires_at": "2026-11-01T00:00:01Z",
        "progress": {"phase": "completed", "percent": 100}, "error": None,
        "request": {"model": "qwen3-asr-1.7b"}, "model_revision": "synthetic-revision",
        "result": {"text": "x" * 16_384, "version": 1, "alignment_status": "aligned"},
        "source_sha256": "synthetic", "delete_requested": False, "cancel_requested": False,
    }
    results: list[dict[str, object]] = []
    previous = 0

    def request() -> tuple[int, int]:
        if summaries:
            body = store.job_summaries("benchmark-owner", limit=50)
            for job in body["jobs"]:
                job["queue_position"] = 0
        else:
            body = {"jobs": [service.public_job(job, include_result=False) for job in store.jobs("benchmark-owner")]}
        return len(body["jobs"]), len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode())

    try:
        for count in (100, 1000, 10_000):
            with store.connection() as db:
                db.executemany("INSERT INTO jobs(id,owner_id,status,reserved_bytes,payload) VALUES (?,?,?,?,?)", (
                    (f"wb_benchmark_{index:05d}", "benchmark-owner", "completed", 0,
                     json.dumps({**fixture, "id": f"wb_benchmark_{index:05d}"}, ensure_ascii=False))
                    for index in range(previous, count)
                ))
            previous = count
            request()  # Warm DB and interpreter cache; never label this cold I/O.
            samples: list[float] = []
            returned, response_bytes = 0, 0
            for _ in range(repeats):
                gc.collect()
                started = time.perf_counter()
                returned, response_bytes = request()
                samples.append((time.perf_counter() - started) * 1000)
            results.append({"records": count, "synthetic_text_bytes_per_record": 16_384,
                            "query": projection, "rows_returned": returned, "response_bytes": response_bytes,
                            "cache": "warm after one unmeasured query", **statistics_ms(samples)})
    finally:
        if callable(getattr(store, "close", None)):
            store.close()
    return results


def _wave_fixture(path: Path, seconds: int) -> None:
    # One deterministic second, repeated incrementally: no full long PCM buffer.
    block = b"".join(struct.pack("<hh", sample, sample) for index in range(44_100)
                     for sample in (round(3000 * math.sin(2 * math.pi * 440 * index / 44_100)),))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(44_100)
        for _ in range(seconds):
            output.writeframesraw(block)


def _digest_pcm(path: Path) -> str:
    digest = hashlib.sha256()
    with wave.open(str(path), "rb") as stream:
        while block := stream.readframes(32_768):
            digest.update(block)
    return digest.hexdigest()


async def decode_benchmark(scratch: Path, repeats: int) -> list[dict[str, object]]:
    normalize = importlib.import_module("asr_server.audio.preprocess").normalize_audio_path_to_wav_async
    results: list[dict[str, object]] = []
    for seconds in (30, 180):
        source = scratch / f"synthetic-{seconds}s.wav"
        target = scratch / "normalized.wav"
        _wave_fixture(source, seconds)
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        await normalize(source, target, timeout_seconds=120)
        target.unlink()
        samples: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            await normalize(source, target, timeout_seconds=120)
            samples.append((time.perf_counter() - started) * 1000)
            with wave.open(str(target), "rb") as normalized:
                sample_rate, channels, frames = normalized.getframerate(), normalized.getnchannels(), normalized.getnframes()
            output_bytes = target.stat().st_size
            pcm_sha256 = _digest_pcm(target)
            target.unlink()
        results.append({"audio_seconds": seconds, "input_sample_rate": 44_100, "input_channels": 2,
                        "source_sha256": source_sha256, "sample_rate": sample_rate, "channels": channels,
                        "frames": frames, "output_bytes": output_bytes, "pcm_sha256": pcm_sha256,
                        "cache": "warm after one unmeasured decode", **statistics_ms(samples)})
        source.unlink()
    return results


def log_benchmark(scratch: Path, repeats: int) -> dict[str, object]:
    try:
        observation = importlib.import_module("asr_server.observability")
    except ModuleNotFoundError:
        return {"available": False, "reason": "application JSONL events absent from baseline"}
    count = 10_000
    results: dict[str, Any] = {}
    for level in ("CRITICAL", "INFO"):
        samples: list[float] = []
        flush_samples: list[float] = []
        dropped: list[int] = []
        for index in range(repeats):
            runtime = observation.configure_logging(level=level, log_dir=scratch / f"log-{level}-{index}")
            context = observation.ExecutionContext.new(job_id="job_benchmark", model_id="synthetic")
            with observation.bind_context(context):
                started = time.perf_counter()
                for chunk in range(count):
                    observation.emit("chunk.completed", chunk_index=chunk, duration_ms=10)
                samples.append((time.perf_counter() - started) * 1000)
                dropped.append(observation.log_stats()["dropped_count"])
                runtime.close()
                flush_samples.append((time.perf_counter() - started) * 1000)
        results[level] = {**statistics_ms(samples), "emit_and_flush_ms": flush_samples, "dropped_count": dropped}
    disabled = results["CRITICAL"]["median_ms"]
    info = results["INFO"]["median_ms"]
    return {"available": True, "event_count": count, "queue_capacity": 4096,
            "workload": "event-only saturated producer microbenchmark; not transcription wall-clock overhead",
            "modes": results, "producer_overhead_percent": (info / disabled - 1) * 100,
            "extra_microseconds_per_event": (info - disabled) * 1000 / count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True, help="new temporary directory, removed after measurement")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scenario", choices=("all", "query", "decode", "logs"), default="all")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three repeats are required")
    source = args.source.resolve()
    if not (source / "asr_server").is_dir():
        parser.error("source must contain asr_server")
    if args.output.exists() or args.scratch.exists():
        parser.error("output and scratch must be new paths")
    if args.output.resolve().is_relative_to(args.scratch.resolve()):
        parser.error("output must be outside temporary scratch")
    sys.path.insert(0, str(source))
    args.scratch.mkdir(parents=True)
    os.environ["ASR_TEMP_DIR"] = str(args.scratch / "temporary")
    metrics: dict[str, object] = {
        "schema_version": 1, "source_label": source.name, "python": platform.python_version(),
        "machine": {"os": platform.system(), "release": platform.release(), "architecture": platform.machine(),
                    "cpu_count": os.cpu_count()},
        "repeats": args.repeats, "scope": "synthetic CPU/I/O only; no GPU, no quality measurement",
        "query_semantics": "baseline returns all rows; current returns one summary page of 50; these are different protocols",
    }
    try:
        if args.scenario in {"all", "query"}:
            metrics["query"] = query_benchmark(args.scratch, args.repeats)
        if args.scenario in {"all", "decode"}:
            metrics["ffmpeg"] = subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
            metrics["decode"] = asyncio.run(decode_benchmark(args.scratch, args.repeats))
        if args.scenario in {"all", "logs"}:
            metrics["logs"] = log_benchmark(args.scratch, args.repeats)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(metrics, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(args.scratch)


if __name__ == "__main__":
    main()
