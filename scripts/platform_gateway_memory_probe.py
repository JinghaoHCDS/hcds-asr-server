"""Measure the complete mock gateway's memory on real multipart/queued media.

No GPU, real speech, remote node or listening port is used. Each independent
case starts FastAPI, SQLite workbench state, shared FIFO, HTTP multipart parser,
real FFmpeg decode, VAD/file splitting and the Qwen mock path adapter. The ASGI
HTTPX harness lives in the measured process, so its overhead is included.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.platform_resource_probe import RSS_BUDGET, rss_bytes, sample_processes
from scripts.workbench_release import ReleaseError, require_generated_output


CASES = (600, 3600, 21600)
PARAMETERS = {"model": "qwen3-asr-1.7b", "backend": "transformers", "quality_profile": "legacy",
              "split_strategy": "auto", "max_chunk_seconds": "120", "overlap_seconds": "2",
              "language": "auto", "timestamps": "none", "response_format": "verbose_json",
              "preserve_segments": "true"}


def code_hashes() -> dict[str, str]:
    names = ["main.py", "jobs.py", "scheduler.py", "transcription.py", "admission.py",
             "audio/preprocess.py", "audio/splitter.py", "audio/workspace.py", "audio/assets.py"]
    return {"asr_server/" + name: hashlib.sha256((Path("asr_server") / name).read_bytes()).hexdigest() for name in names}


async def gateway_case(seconds: int, root: Path) -> dict[str, Any]:
    # Keep the module entry point's unused default app within this case too.
    os.environ["ASR_WORKBENCH_DATA_DIR"] = str(root / "workbench")
    os.environ["ASR_TEMP_DIR"] = str(root / "spool")
    os.environ["ASR_ADAPTER"] = "mock"
    from httpx import ASGITransport, AsyncClient
    from asr_server.config import Settings
    from asr_server.main import create_app

    source = root / "silence.flac"
    with source.open("rb") as stream:
        source_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    settings = Settings(adapter="mock", workbench_enabled=True, workbench_data_dir=root / "workbench",
                        min_free_disk_mb=0, idle_unload_seconds=0, log_level="INFO", log_format="json",
                        log_dir=root / "logs", preprocessing_prefetch=False)
    app = create_app(settings)
    manager = app.state.workspace_manager
    phases: list[str] = []
    poll_codes: list[int] = []
    result: dict[str, Any] = {"case_seconds": seconds, "source_bytes": source.stat().st_size,
                              "source_sha256": source_digest, "source_format": "FLAC mono 16000 Hz digital silence",
                              "parameters": PARAMETERS, "workbench_initialized": True,
                              "prefetch": False, "api_max_upload_bytes": settings.max_upload_mb * 1024**2}
    process_baseline = rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    started = time.perf_counter()
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://memory-probe") as client:
            health = await client.get("/health")
            models = await client.get("/v1/models")
            baseline_rss, _ = sample_processes(os.getpid())
            result.update(health_http_status=health.status_code, models_http_status=models.status_code,
                          gateway_baseline_ru_maxrss_bytes=rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                          gateway_baseline_sampled_rss_bytes=baseline_rss,
                          startup_seconds=time.perf_counter() - started,
                          gateway_before_startup_ru_maxrss_bytes=process_baseline)
            operation_started = time.perf_counter()
            with source.open("rb") as stream:
                created = await client.post("/v1/audio/transcription-jobs",
                    files={"file": (source.name, stream, "audio/flac")}, data=PARAMETERS)
            result["create_http_status"] = created.status_code
            if created.status_code != 202:
                raise RuntimeError(f"job creation failed: HTTP {created.status_code}, {created.text[:500]}")
            job_id = created.json()["id"]
            result["job_id"] = job_id
            while True:
                response = await client.get(f"/v1/jobs/{job_id}", params={"detail": "compact"})
                poll_codes.append(response.status_code)
                response.raise_for_status()
                status = response.json()
                phase = status["status"]
                if not phases or phase != phases[-1]:
                    phases.append(phase)
                if phase in {"completed", "failed", "cancelled"}:
                    break
                if time.perf_counter() - operation_started > 90:
                    raise RuntimeError("gateway job exceeded 90 seconds")
                await asyncio.sleep(0.05)
            full = await client.get(f"/v1/jobs/{job_id}", params={"detail": "full"})
            full.raise_for_status()
            body = full.json()
            if body["status"] != "completed":
                raise RuntimeError(f"gateway job failed: {body.get('error')}")
            output = body["result"]
            chunks = output["chunks"]
            cursor = 0.0
            coverage = bool(chunks)
            for chunk in chunks:
                coverage = coverage and float(chunk["start"]) <= cursor + 1 / 16000 and float(chunk["end"]) > float(chunk["start"])
                cursor = max(cursor, float(chunk["end"]))
            coverage = coverage and abs(cursor - seconds) <= 1 / 16000
            result.update(job_status=body["status"], observed_job_phases=phases,
                          poll_count=len(poll_codes), polling_http_statuses=sorted(set(poll_codes)),
                          final_http_status=full.status_code, operation_seconds=time.perf_counter() - operation_started,
                          result_duration_seconds=output["duration"], chunks=len(chunks),
                          coverage_start=chunks[0]["start"] if chunks else None,
                          coverage_end=cursor, full_coverage=coverage, split=output["split"],
                          gateway_peak_before_shutdown_ru_maxrss_bytes=rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                          logical_spool_after_completed=manager._reserved_upload_bytes,
                          temporary_pending_after_completed=sum(manager._pending_disk_bytes.values()),
                          persistent_pending_after_completed=sum(manager._persistent_totals.values()),
                          temporary_files_after_completed=[path.name for path in (root / "spool").iterdir()])
            if not coverage or abs(float(output["duration"]) - seconds) > 1 / 16000:
                raise RuntimeError("gateway result does not cover the complete input")
    result.update(gateway_ru_maxrss_bytes=rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  children_ru_maxrss_bytes=rss_bytes(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
                  total_seconds=time.perf_counter() - started,
                  logical_spool_after_shutdown=manager._reserved_upload_bytes,
                  temporary_pending_after_shutdown=sum(manager._pending_disk_bytes.values()),
                  persistent_pending_after_shutdown=sum(manager._persistent_totals.values()))
    events: list[dict[str, Any]] = []
    for log in (root / "logs").glob("*.jsonl*"):
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    normalized = [event for event in events if event.get("event") == "media.normalized"]
    result["normalized_events"] = [{key: event.get(key) for key in
        ("event", "output_bytes", "audio_seconds", "frames", "sample_rate", "duration_ms")} for event in normalized]
    if len(normalized) != 1 or normalized[0].get("output_bytes") != seconds * 32000 + 44:
        raise RuntimeError("expected one exact, full-length normalized asset")
    if (result["logical_spool_after_shutdown"] or result["temporary_pending_after_shutdown"]
            or result["persistent_pending_after_shutdown"] or result["temporary_files_after_completed"]):
        raise RuntimeError("gateway retained temporary resources after completion")
    return result


def supervise(seconds: int, workspace: Path, sample_ms: float) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix=f"gateway_{seconds}_", dir=workspace))
    source = root / "silence.flac"
    generation_started = time.perf_counter()
    try:
        # Generator is outside the measured gateway process; decoder samples
        # therefore cannot be confused with compression of this fixture.
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "anullsrc=r=16000:cl=mono", "-t", str(seconds), "-c:a", "flac", "-compression_level", "5",
                        str(source)], check=True, capture_output=True, timeout=60)
        generation_seconds = time.perf_counter() - generation_started
        if source.stat().st_size >= 512 * 1024**2:
            raise RuntimeError("compressed fixture unexpectedly exceeds the legacy upload limit")
        process = subprocess.Popen([sys.executable, "-m", "scripts.platform_gateway_memory_probe", "--case", str(seconds),
                                    "--case-root", str(root)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        started = time.perf_counter()
        sampled_peak = 0
        decoder_peaks: dict[int, int] = {}
        samples = 0
        while process.poll() is None:
            gateway, decoders = sample_processes(process.pid)
            sampled_peak = max(sampled_peak, gateway or 0)
            for pid, value in decoders.items():
                decoder_peaks[pid] = max(decoder_peaks.get(pid, 0), value)
            samples += 1
            if time.perf_counter() - started > 120:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                process.communicate(timeout=5)
                raise RuntimeError("gateway case exceeded 120 seconds")
            time.sleep(sample_ms / 1000)
        stdout, stderr = process.communicate(timeout=5)
        if process.returncode:
            raise RuntimeError(f"gateway case {seconds} failed: {stderr[-2500:]}")
        result: dict[str, Any] = json.loads(stdout)
        baseline = result["gateway_baseline_sampled_rss_bytes"] or result["gateway_baseline_ru_maxrss_bytes"]
        peak = max(sampled_peak, result["gateway_ru_maxrss_bytes"])
        increment = max(0, peak - baseline)
        result.update(generator_seconds=generation_seconds, supervisor_wall_seconds=time.perf_counter() - started,
                      gateway_sampled_peak_rss_bytes=sampled_peak, gateway_peak_rss_bytes=peak,
                      gateway_peak_increment_bytes=increment, gateway_increment_within_256_mib=increment <= RSS_BUDGET,
                      decoder_sampled_peak_rss_bytes=max(decoder_peaks.values(), default=None),
                      decoder_sampled_by_pid={str(pid): value for pid, value in decoder_peaks.items()},
                      rss_samples=samples, decoder_measurement="ps samples of gateway child FFmpeg only; generator excluded")
        if not decoder_peaks:
            result["decoder_measurement_missing"] = "normalization finished between samples"
        return result
    finally:
        shutil.rmtree(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("data/platform-validation-20261025/gateway-memory"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/platform/gateway-memory.json"))
    parser.add_argument("--sample-ms", type=float, default=25)
    parser.add_argument("--cases", type=int, nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--case", type=int, choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--case-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("this resource probe requires POSIX and ps")
    if not 0 < args.sample_ms <= 1000:
        parser.error("sample interval must be in (0, 1000] ms")
    try:
        require_generated_output(args.workspace)
        require_generated_output(args.output)
    except ReleaseError as exc:
        parser.error(str(exc))
    if args.case:
        if args.case_root is None:
            parser.error("internal case requires --case-root")
        try:
            require_generated_output(args.case_root)
        except ReleaseError as exc:
            parser.error(str(exc))
        print(json.dumps(asyncio.run(gateway_case(args.case, args.case_root)), sort_keys=True))
        return
    args.workspace.mkdir(parents=True, exist_ok=True)
    results = []
    for seconds in args.cases:
        result = supervise(seconds, args.workspace, args.sample_ms)
        result["case_files_removed"] = True
        results.append(result)
        print(json.dumps({"case_seconds": seconds, "job_status": result["job_status"],
                          "gateway_peak_increment_mib": result["gateway_peak_increment_bytes"] / 1024**2,
                          "full_coverage": result["full_coverage"]}), flush=True)
    report = {"schema_version": 1, "utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "python": platform.python_version(), "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
              "scope": "complete FastAPI gateway + initialized SQLite workbench + HTTPX ASGI harness in each independent child",
              "adapter": "Qwen mock; real parser/upload/probe/decode/VAD/split/FIFO/status/cleanup",
              "rss_method": "self ru_maxrss and supervisor ps; baseline after startup and health/models warmup",
              "sampling_interval_ms": args.sample_ms, "rss_increment_budget_bytes": RSS_BUDGET,
              "source_code_sha256": code_hashes(), "cases": results,
              "not_validated": ["GPU/CUDA", "ASR quality", "model worker memory", "real network latency", "WSL FFmpeg build"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as target:
        os.chmod(args.output, 0o600)
        target.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "all_within_budget": all(case["gateway_increment_within_256_mib"] for case in results)}))


if __name__ == "__main__":
    main()
