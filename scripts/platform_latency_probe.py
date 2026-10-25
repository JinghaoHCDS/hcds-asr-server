"""CPU-only ASGI workload probe: polling, two uploads and one delayed mock job.

This measures in-process server behavior, not network, GPU or ASR quality.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import io
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any


def distribution(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    return {"count": len(values), "p95_ms": ordered[max(0, math.ceil(len(values) * 0.95) - 1)] if values else None,
            "median_ms": statistics.median(values) if values else None,
            "min_ms": min(values) if values else None, "max_ms": max(values) if values else None}


def audio(seconds: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        for _ in range(seconds):
            output.writeframesraw(b"\0\0" * 16000)
    return buffer.getvalue()


async def round_probe(root: Path, index: int) -> dict[str, object]:
    import httpx

    Settings = importlib.import_module("asr_server.config").Settings
    create_app = importlib.import_module("asr_server.main").create_app
    options: dict[str, Any] = {"workbench_data_dir": root / f"round-{index}", "min_free_disk_mb": 0}
    if "log_dir" in Settings.__dataclass_fields__:
        options["log_dir"] = root / f"logs-{index}"
    app = create_app(Settings(**options), adapter_delay_seconds=8.0)
    service = app.state.workbench
    user = service.store.create_user("benchmark-user", "benchmark-only-password")
    fixture = {
        "owner_id": user["id"], "filename": "synthetic.wav", "status": "completed",
        "created_at": "2026-10-25T00:00:00Z", "started_at": "2026-10-25T00:00:00Z",
        "completed_at": "2026-10-25T00:00:01Z", "expires_at": "2099-01-01T00:00:00Z",
        "progress": {"phase": "completed", "percent": 100}, "error": None,
        "request": {"model": "qwen3-asr-1.7b"}, "model_revision": "synthetic-revision",
        "result": {"text": "x" * 16384, "version": 1, "alignment_status": "aligned"},
        "source_sha256": "synthetic", "delete_requested": False, "cancel_requested": False,
    }
    with service.store.connection() as db:
        db.executemany("INSERT INTO jobs(id,owner_id,status,reserved_bytes,payload) VALUES (?,?,?,?,?)", (
            (f"wb_benchmark_{value:05d}", user["id"], "completed", 0,
             json.dumps({**fixture, "id": f"wb_benchmark_{value:05d}"})) for value in range(1000)
        ))
    summary = callable(getattr(service.store, "job_summaries", None))
    list_url = "/api/jobs?view=summary&limit=50" if summary else "/api/jobs"
    measurements: dict[str, list[float]] = {"health": [], "jobs": []}
    response_bytes = {"health": 0, "jobs": 0}
    schedule_lateness: list[float] = []
    loop_lags: list[float] = []
    upload_results: list[dict[str, object]] = []
    stop = asyncio.Event()
    transport = httpx.ASGITransport(app=app)
    wall_started = time.perf_counter()
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://benchmark") as control:
            logged_in = await control.post("/api/auth/login", json={"username": "benchmark-user", "password": "benchmark-only-password"})
            if logged_in.status_code != 200:
                raise RuntimeError("synthetic account login failed")
            clients = [httpx.AsyncClient(transport=transport, base_url="http://benchmark", cookies=control.cookies) for _ in range(10)]
            try:
                async def warm(client: httpx.AsyncClient) -> None:
                    for url in ("/health", list_url):
                        response = await client.get(url)
                        if response.status_code != 200:
                            raise RuntimeError("warmup request failed")
                await asyncio.gather(*(warm(client) for client in clients))
                created = await control.post("/v1/audio/transcription-jobs", files={"file": ("short.wav", audio(1))})
                if created.status_code != 202:
                    raise RuntimeError("mock job admission failed")
                job_id = created.json()["id"]
                job: dict[str, Any] = {}
                for _ in range(200):
                    job = (await control.get(f"/v1/jobs/{job_id}")).json()
                    if job["status"] == "transcribing":
                        break
                    if job["status"] in {"failed", "cancelled"}:
                        raise RuntimeError("mock job failed before load probe")
                    await asyncio.sleep(0.01)
                inference_observed = job.get("status") == "transcribing"
                started = time.perf_counter()

                async def lag_probe() -> None:
                    while not stop.is_set():
                        expected = time.perf_counter() + 0.02
                        await asyncio.sleep(0.02)
                        loop_lags.append(max(0, time.perf_counter() - expected) * 1000)

                async def poll(client: httpx.AsyncClient) -> None:
                    for tick in range(3):
                        target = started + tick * 3.5
                        await asyncio.sleep(max(0, target - time.perf_counter()))
                        schedule_lateness.append(max(0, time.perf_counter() - target) * 1000)
                        for kind, url in (("health", "/health"), ("jobs", list_url)):
                            request_started = time.perf_counter()
                            response = await client.get(url)
                            measurements[kind].append((time.perf_counter() - request_started) * 1000)
                            if response.status_code != 200:
                                raise RuntimeError("polling request failed")
                            response_bytes[kind] += len(response.content)

                async def upload() -> None:
                    content = audio(60)  # 1,920,044 bytes, bounded and valid WAV.
                    upload_started = time.perf_counter()
                    record = await control.post("/api/uploads", json={"filename": "upload.wav", "size_bytes": len(content)})
                    if record.status_code != 201:
                        raise RuntimeError("synthetic upload admission failed")
                    upload_id = record.json()["id"]
                    block = 256 * 1024
                    for offset in range(0, len(content), block):
                        response = await control.patch(f"/api/uploads/{upload_id}", content=content[offset:offset + block], headers={"Upload-Offset": str(offset)})
                        if response.status_code != 200:
                            raise RuntimeError("synthetic upload append failed")
                        await asyncio.sleep(0.1)
                    upload_results.append({"bytes": len(content), "duration_ms": (time.perf_counter() - upload_started) * 1000})

                monitor = asyncio.create_task(lag_probe())
                try:
                    await asyncio.gather(*(poll(client) for client in clients), upload(), upload())
                    await asyncio.sleep(max(0, started + 7.1 - time.perf_counter()))
                finally:
                    stop.set()
                    await monitor
                measured_wall_ms = (time.perf_counter() - started) * 1000
                for _ in range(200):
                    job = (await control.get(f"/v1/jobs/{job_id}")).json()
                    if job["status"] in {"completed", "failed", "cancelled"}:
                        break
                    await asyncio.sleep(0.01)
                if job["status"] != "completed":
                    raise RuntimeError("mock job did not finish safely")
            finally:
                await asyncio.gather(*(client.aclose() for client in clients))
    return {"round": index, "records": 1000, "clients": 10, "poll_interval_seconds": 3.5,
            "polls_per_client": 3, "list_protocol": "summary page of 50" if summary else "legacy all rows",
            "measured_wall_ms": measured_wall_ms, "round_wall_ms": (time.perf_counter() - wall_started) * 1000,
            "health": distribution(measurements["health"]), "jobs": distribution(measurements["jobs"]),
            "response_bytes": response_bytes, "loop_lag": distribution(loop_lags),
            "schedule_lateness": distribution(schedule_lateness), "uploads": upload_results,
            "mock_inference_delay_seconds": 8, "inference_observed_transcribing": inference_observed,
            "mock_terminal_status": job["status"]}


async def run(root: Path, repeats: int) -> list[dict[str, object]]:
    results = []
    for index in range(repeats):
        result = await round_probe(root, index)
        results.append(result)
        print(json.dumps({"completed_round": index + 1, "health": result["health"], "jobs": result["jobs"], "loop_lag": result["loop_lag"]}), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three rounds are required")
    if args.scratch.exists() or args.output.exists():
        parser.error("scratch and output must be new paths")
    if args.output.resolve().is_relative_to(args.scratch.resolve()):
        parser.error("output must be outside temporary scratch")
    sys.path.insert(0, str(args.source.resolve()))
    args.scratch.mkdir(parents=True)
    os.environ["ASR_TEMP_DIR"] = str(args.scratch / "temporary")
    try:
        rounds = asyncio.run(run(args.scratch, args.repeats))
        metrics = {"schema_version": 1, "source_label": args.source.resolve().name,
                   "python": platform.python_version(), "platform": platform.system(), "architecture": platform.machine(),
                   "scope": "in-process ASGI CPU/I/O with delayed mock inference; no network, GPU or quality measurement",
                   "warmup": "one health and one list request per client before the measured window",
                   "polling_latency_ms": "elapsed ASGI request time; scheduling lateness and 20ms loop lag recorded separately",
                   "rounds": rounds}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(metrics, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print("Wrote measured ASGI latency results", flush=True)
    finally:
        shutil.rmtree(args.scratch)


if __name__ == "__main__":
    main()
