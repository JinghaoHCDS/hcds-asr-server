"""Alternate prefetch off/on against one frozen 30 s audio, offline and isolated.

Each repetition starts a fresh gateway/model worker, submits three real legacy
HTTP jobs, and unloads before exit. RPC envelope gaps include IPC/scheduling and
are only a proxy for time between model calls, never GPU kernel idle time.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.platform_gpu_compare import (
    child_environment, digest_file, require_untracked_output, source_identity, stop_process, write_json,
)


MODEL = "qwen3-asr-0.6b"
CHILD_BOOTSTRAP = r'''
import asyncio, importlib, json, sys
from pathlib import Path
configuration = json.loads(sys.argv[1])
source = Path(configuration["source"]).resolve()
sys.path.insert(0, str(source))
probe = importlib.import_module("scripts.platform_prefetch_probe")
assert Path(probe.__file__).resolve() == source / "scripts" / "platform_prefetch_probe.py"
metrics = asyncio.run(probe.run_child(configuration))
probe.write_json(Path(configuration["metrics"]), metrics)
'''


def require(value: object, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def warm_existing_cache(definition: Any, audio: Path, *, mock: bool) -> dict[str, Any]:
    files = [audio]
    if not mock:
        hub = importlib.import_module("huggingface_hub")
        config = hub.try_to_load_from_cache(definition.repository, "config.json", revision=definition.revision)
        require(isinstance(config, str), "fixed model revision is absent from the existing cache")
        assert isinstance(config, str)
        snapshot = Path(config).parent
        weights = sorted(snapshot.glob("*.safetensors"))
        require(weights, "cached model safetensors are unavailable; no download attempted")
        files.extend(weights)
        files.extend(path for path in sorted(snapshot.iterdir()) if path.suffix in {".json", ".txt", ".model"} and path.is_file())
    total = 0
    started = time.perf_counter()
    for path in files:
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                total += len(block)
    return {"files_read": len(files), "bytes_read": total, "wall_seconds": time.perf_counter() - started,
            "scope": "audio only" if mock else "existing fixed-revision weights/tokenizer plus audio; no download"}


def read_events(root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(root.glob("events.jsonl*")):
        with path.open("rb") as stream:
            for line in stream:
                require(len(line) <= 8193, "oversized application event")
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    result.append(parsed)
    return sorted(result, key=lambda event: str(event.get("ts", "")))


async def run_child(configuration: dict[str, Any]) -> dict[str, Any]:
    import httpx
    from asr_server import observability
    from asr_server.adapters.base import AudioInput, TranscriptionResult
    from asr_server.adapters.mock import MockAsrAdapter
    from asr_server.config import Settings
    from asr_server.main import create_app

    class MeasuredMockAdapter(MockAsrAdapter):
        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str,
                             context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            started = time.perf_counter()
            result = await super().transcribe(audio, model_id=model_id, backend=backend, language=language,
                                             context=context, timestamps=timestamps, max_new_tokens=max_new_tokens)
            observability.emit("probe.mock_inference_completed", level=logging.DEBUG, operation="transcribe",
                               duration_ms=(time.perf_counter() - started) * 1000)
            return result

    work, audio, raw = (Path(configuration[key]) for key in ("work", "audio", "raw"))
    mock = bool(configuration["mock"])
    enabled = bool(configuration["prefetch"])
    timeout = float(configuration["timeout"])
    app = create_app(Settings(
        adapter="mock" if mock else "registered", default_model=MODEL, workbench_enabled=False,
        workbench_data_dir=work / "unused-workbench", api_token=None, enable_qwen_alignment=False,
        preprocessing_prefetch=enabled, idle_unload_seconds=0, log_level="DEBUG", log_dir=raw / "events",
        min_free_disk_mb=0 if mock else int(configuration["min_free_disk_mb"]),
        worker_load_timeout_seconds=timeout, worker_inference_timeout_seconds=timeout,
        worker_shutdown_timeout_seconds=30, job_shutdown_grace_seconds=60,
    ))
    definition = app.state.manager.runtime_for(MODEL).definition
    if mock:
        app.state.manager.runtime_for(MODEL).adapter = MeasuredMockAdapter(delay_seconds=0.05)
    cold = app.state.manager.runtime_for(MODEL)
    require(cold.status == "unloaded" and cold.active_requests == 0, "model was not initially cold")
    cache = warm_existing_cache(definition, audio, mock=mock)
    identifiers: list[str] = []
    full_jobs: list[dict[str, Any]] = []
    phases: dict[str, list[str]] = {}
    parameters = {"model": MODEL, "language": "zh", "quality_profile": "legacy", "timestamps": "none",
                  "response_format": "verbose_json", "max_new_tokens": "4096", "split_strategy": "auto"}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://prefetch-validation",
                                     timeout=timeout) as client:
            started = time.perf_counter()
            for index in range(3):
                with audio.open("rb") as stream:
                    response = await client.post("/v1/audio/transcription-jobs", files={"file": ("frozen.wav", stream)}, data=parameters)
                write_json(raw / f"submit-{index + 1}.json", {"status": response.status_code, "body": response.json()})
                require(response.status_code == 202, "job submission failed; inspect its private response")
                identifiers.append(str(response.json()["id"]))

            async def completed(job_id: str) -> dict[str, Any]:
                phases[job_id] = []
                deadline = time.perf_counter() + timeout
                while time.perf_counter() < deadline:
                    response = await client.get("/v1/jobs/" + job_id, params={"detail": "compact"})
                    require(response.status_code == 200, "job polling failed")
                    job = response.json()
                    state = str(job["status"])
                    if not phases[job_id] or phases[job_id][-1] != state:
                        phases[job_id].append(state)
                    if state in {"completed", "failed", "cancelled", "expired"}:
                        full = (await client.get("/v1/jobs/" + job_id, params={"detail": "full"})).json()
                        write_json(raw / (job_id + ".json"), full)
                        require(state == "completed", "job did not complete; inspect its private result")
                        require(full["result"].get("text", "").strip(), "job returned empty text")
                        require(not (full["result"].get("generation") or {}).get("truncated", False), "job result was truncated")
                        return dict(full)
                    await asyncio.sleep(0.01 if mock else 0.05)
                raise RuntimeError("job polling deadline exceeded")

            full_jobs = await asyncio.gather(*(completed(job_id) for job_id in identifiers))
            batch_wall = time.perf_counter() - started
            runtime = app.state.manager.runtime_for(MODEL)
            transport = getattr(runtime.adapter, "_transport", None)
            worker_pid = transport.pid if transport is not None else None
            require(runtime.active_requests == 0 and runtime.status == "loaded", "completed jobs left model ownership active")
            require(app.state.workspace_manager._reserved_upload_bytes == 0, "completed jobs leaked temporary quota")
            if not mock:
                require(transport is not None and transport.is_alive and worker_pid is not None, "real model worker was lost")
            response = await client.delete("/v1/models/" + MODEL)
            require(response.status_code == 200 and runtime.status == "unloaded", "final unload failed")
            require(transport is None or not transport.is_alive, "worker was not reaped")
            worker_reaped = transport is None or not transport.is_alive
            quota_after = app.state.workspace_manager._reserved_upload_bytes
    require(app.state.workspace_manager._reserved_upload_bytes == 0, "shutdown leaked temporary quota")
    events = read_events(raw / "events")
    queued = [event for event in events if event["event"] == "job.queued" and event.get("job_id") in identifiers]
    job_started = [event for event in events if event["event"] == "job.started" and event.get("job_id") in identifiers]
    require([event["job_id"] for event in queued] == identifiers, "job admission order differs from submission FIFO")
    require([event["job_id"] for event in job_started] == identifiers, "scheduler reordered submitted jobs")
    normalization = {job_id: sum(event["event"] == "media.normalized" and event.get("job_id") == job_id for event in events) for job_id in identifiers}
    require(all(count == 1 for count in normalization.values()), "a job normalized zero or multiple times")
    loads = [event for event in events if event["event"] == "model.loaded" and event.get("model_id") == MODEL]
    require(len(loads) == 1, "same-model jobs did not reuse one model load")
    event_name = "probe.mock_inference_completed" if mock else "worker.rpc_completed"
    calls = [event for event in events if event["event"] == event_name and event.get("job_id") in identifiers
             and event.get("operation") in {"transcribe", "transcribe_batch"}]
    require([event["job_id"] for event in calls] == identifiers, "expected exactly one ordered transcription call per short job")
    intervals = []
    for event in calls:
        end = datetime.fromisoformat(event["ts"].replace("Z", "+00:00")).timestamp()
        duration = float(event["duration_ms"]) / 1000
        intervals.append({"job_id": event["job_id"], "start_utc_seconds": end - duration,
                          "end_utc_seconds": end, "rpc_duration_seconds": duration,
                          "worker_pid": event.get("worker_pid"), "rpc_request_id": event.get("rpc_request_id")})
    gaps = [following["start_utc_seconds"] - previous["end_utc_seconds"] for previous, following in zip(intervals, intervals[1:])]
    require(all(gap >= -0.002 for gap in gaps), "overlapping inference RPC envelopes observed")
    if not mock:
        require({item["worker_pid"] for item in intervals} == {worker_pid}, "same-model requests used different workers")
    texts = [str(job["result"]["text"]) for job in full_jobs]
    job_metrics = []
    for index, job in enumerate(full_jobs):
        created_at, started_at, completed_at = (str(job[key]) for key in ("created_at", "started_at", "completed_at"))
        created_seconds, started_seconds, completed_seconds = (
            datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            for value in (created_at, started_at, completed_at)
        )
        timings = dict(job["result"].get("timings") or {})
        job_loads = [event for event in loads if event.get("job_id") == identifiers[index]]
        require(bool(job_loads) == (index == 0), "cold load was not attributable only to the first job")
        job_metrics.append({
            "job_id": identifiers[index], "batch_index": index + 1,
            "created_at": created_at, "started_at": started_at, "completed_at": completed_at,
            "queue_and_prepare_before_started_seconds": started_seconds - created_seconds,
            "processing_after_started_seconds": completed_seconds - started_seconds,
            "end_to_end_since_created_seconds": completed_seconds - created_seconds,
            "rtf_processing_after_started": (completed_seconds - started_seconds) / float(configuration["duration"]),
            "rtf_end_to_end_since_created": (completed_seconds - created_seconds) / float(configuration["duration"]),
            "result_timings": timings,
            "reported_pipeline_rtf": (float(timings["total_ms"]) / 1000 / float(configuration["duration"])) if "total_ms" in timings else None,
            "cold_model_load": bool(job_loads),
            "model_loaded_events": [{key: event[key] for key in ("ts", "job_id", "model_id", "duration_ms") if key in event} for event in job_loads],
        })
    metrics: dict[str, Any] = {
        "mode": "mock" if mock else "real-offline", "prefetch": enabled, "batch_jobs": 3,
        "batch_wall_seconds": batch_wall, "batch_rtf_end_to_end": batch_wall / (3 * float(configuration["duration"])),
        "batch_throughput_audio_seconds_per_wall_second": 3 * float(configuration["duration"]) / batch_wall,
        "model_state": "fresh model and gateway process; disk inputs explicitly warmed",
        "cache_warm": cache, "parameters": parameters, "model_identity": definition.identity(),
        "submitted_order": identifiers, "started_order": [event["job_id"] for event in job_started],
        "sequences": [event["sequence"] for event in queued], "phases": phases,
        "normalization_calls_by_job": normalization, "model_loads": len(loads),
        "worker_pid": worker_pid, "worker_reaped": worker_reaped, "temporary_quota_after": quota_after,
        "jobs": job_metrics,
        "job_timing_interpretation": "created is after upload/hash; started is _run_job entry. With prefetch on, normalization precedes started, so processing-after-started and pipeline total exclude prefetched decoding. Batch wall and end-to-end include it; overlapping stages must not be summed.",
        "model_loaded_events": [{key: event[key] for key in ("ts", "job_id", "model_id", "duration_ms") if key in event} for event in loads],
        "inference_interval_source": "mock adapter elapsed time; no RPC or GPU" if mock else "actual parent-worker transcription RPC envelope elapsed time",
        "interval_interpretation": "UTC completion timestamp minus monotonic RPC duration; contains IPC and scheduling, not CUDA kernel timing",
        "inference_intervals": intervals, "inter_call_gap_proxy_seconds": gaps,
        "inter_call_gap_proxy_total_seconds": sum(max(0, value) for value in gaps),
        "inter_call_gap_proxy_max_seconds": max(gaps),
        "texts": [{"sha256": hashlib.sha256(text.encode()).hexdigest(), "characters": len(text)} for text in texts],
        "dropped_log_events": max((int(event.get("log_queue_dropped_total", 0)) for event in events), default=0),
        "versions": {}, "python": platform.python_version(),
    }
    require(metrics["dropped_log_events"] == 0, "event loss makes interval or normalization evidence incomplete")
    for name in ("torch", "transformers", "fastapi", "starlette", "httpx"):
        try:
            metrics["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            metrics["versions"][name] = None
    return metrics


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    import statistics

    fields = ("batch_wall_seconds", "batch_rtf_end_to_end", "inter_call_gap_proxy_total_seconds")
    return {field: {"median": statistics.median(values), "min": min(values), "max": max(values)}
            for field in fields for values in ([float(run[field]) for run in runs],)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio30s", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--operation-timeout", type=float, default=600)
    parser.add_argument("--min-free-disk-mb", type=int, default=5120)
    args = parser.parse_args()
    source, audio, output = (path.resolve() for path in (args.source, args.audio30s, args.output_dir))
    if args.repeats < 3 or not math.isfinite(args.operation_timeout) or args.operation_timeout <= 0 or args.min_free_disk_mb < 0:
        parser.error("at least three repeats, positive timeout and nonnegative free-disk minimum required")
    if output.exists():
        parser.error("output-dir must be a new ignored or external directory")
    try:
        require_untracked_output(output, [source])
        with wave.open(str(audio), "rb") as track:
            duration = track.getnframes() / track.getframerate()
        if duration <= 0 or (not args.mock and abs(duration - 30) > 0.1):
            parser.error("real mode requires a 30-second WAV; mock accepts a short fixture")
    except (ValueError, OSError, wave.Error) as exc:
        parser.error(str(exc))
    identity = source_identity(source)
    output.mkdir(parents=True, mode=0o700)
    scratch = output / "scratch"
    scratch.mkdir(mode=0o700)
    private = output / "private"
    private.mkdir(mode=0o700)
    frozen = scratch / "frozen.wav"
    shutil.copyfile(audio, frozen)
    frozen.chmod(0o600)
    summary: dict[str, Any] = {
        "schema_version": 1, "status": "running", "mock": args.mock, "model": MODEL, "source": identity,
        "probe_sha256": digest_file(source / "scripts" / "platform_prefetch_probe.py"),
        "audio": {"sha256": digest_file(frozen), "bytes": frozen.stat().st_size, "duration_seconds": duration},
        "repeats_per_mode": args.repeats, "order": "off1,on1,off2,on2,...", "runs": [],
        "scope": "same frozen audio/parameters, three FIFO legacy jobs per fresh app; no sockets or deployment",
        "gap_interpretation": "transcription RPC envelope gap proxy includes IPC and scheduler; it is not GPU kernel idle",
    }
    try:
        for iteration in range(1, args.repeats + 1):
            for enabled in (False, True):
                name = f"{'on' if enabled else 'off'}-{iteration:02d}"
                work, raw = scratch / name, private / name
                work.mkdir(mode=0o700)
                raw.mkdir(mode=0o700)
                metrics = output / (name + ".metrics.json")
                configuration = {"source": str(source), "work": str(work), "raw": str(raw), "metrics": str(metrics),
                                 "audio": str(frozen), "duration": duration, "mock": args.mock, "prefetch": enabled,
                                 "timeout": args.operation_timeout, "min_free_disk_mb": args.min_free_disk_mb}
                with (raw / "stdout.log").open("xb") as stdout, (raw / "stderr.log").open("xb") as stderr:
                    os.chmod(raw / "stdout.log", 0o600)
                    os.chmod(raw / "stderr.log", 0o600)
                    process = subprocess.Popen([sys.executable, "-I", "-c", CHILD_BOOTSTRAP, json.dumps(configuration)],
                        cwd=work, env=child_environment(work, None), stdout=stdout, stderr=stderr, start_new_session=os.name == "posix")
                    try:
                        process.wait(timeout=args.operation_timeout + 180)
                    except BaseException:
                        stop_process(process)
                        raise
                require(process.returncode == 0, name + " failed; inspect private stderr/results")
                record = json.loads(metrics.read_text())
                summary["runs"].append({"run": name, **record})
                write_json(output / f"progress-{name}.json", {"status": "passed", "run": name})
                print(json.dumps({"run": name, "status": "passed", "batch_wall_seconds": record["batch_wall_seconds"],
                                  "rpc_gap_proxy_seconds": record["inter_call_gap_proxy_total_seconds"]}), flush=True)
        off = [run for run in summary["runs"] if not run["prefetch"]]
        on = [run for run in summary["runs"] if run["prefetch"]]
        summary["aggregates"] = {"off": aggregate(off), "on": aggregate(on)}
        med_off = summary["aggregates"]["off"]["batch_wall_seconds"]["median"]
        med_on = summary["aggregates"]["on"]["batch_wall_seconds"]["median"]
        summary["median_batch_change_percent"] = (med_on / med_off - 1) * 100
        summary["text_digests"] = sorted({item["sha256"] for run in summary["runs"] for item in run["texts"]})
        summary["text_equivalent_across_all_jobs"] = len(summary["text_digests"]) == 1
        summary["source_unchanged"] = source_identity(source) == identity
        require(summary["source_unchanged"], "runtime source changed during comparison")
        summary["status"] = "passed"
        summary["interpretation"] = ("mock plumbing only; no GPU performance conclusion" if args.mock else
            "three samples per mode; compare median and ranges, source/text identity and gap proxy before attributing a benefit")
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error_type"] = type(exc).__name__
        raise
    finally:
        write_json(output / "summary.json", summary)
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
