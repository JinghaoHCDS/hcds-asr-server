"""Read-only, bounded ASR event traces and deliberately content-free bundles."""
from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import platform
import re
import sqlite3
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from asr_server.observability import MAX_RECORD_BYTES, event_record


MAX_INPUT_BYTES = 100 * 1024 * 1024
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")


def _events(log_dir: Path) -> Iterator[dict[str, Any]]:
    consumed = 0
    files = sorted(log_dir.glob("events.jsonl*"), key=lambda path: path.stat().st_mtime)
    for path in files[-10:]:
        if not path.is_file() or path.is_symlink():
            continue
        with path.open("rb") as stream:
            while consumed < MAX_INPUT_BYTES:
                raw = stream.readline(MAX_RECORD_BYTES + 2)
                if not raw:
                    break
                consumed += len(raw)
                if not raw.endswith(b"\n") or len(raw) > MAX_RECORD_BYTES + 1:
                    # Ignore a corrupted/foreign overlong line without allocating it.
                    while raw and not raw.endswith(b"\n") and consumed < MAX_INPUT_BYTES:
                        raw = stream.readline(MAX_RECORD_BYTES + 2)
                        consumed += len(raw)
                    continue
                try:
                    event = json.loads(raw)
                except (ValueError, UnicodeError):
                    continue
                if isinstance(event, dict) and event.get("schema_version") == 1:
                    yield event


def _public_event(raw: dict[str, Any]) -> dict[str, object]:
    # Reapply the producer whitelist: a bundle must not trust arbitrary old logs.
    event = str(raw.get("event", "event.invalid"))
    filtered = event_record(event, **{key: value for key, value in raw.items() if key not in {"event", "level"}})
    for key in ("ts", "level", "service_instance_id", "pid"):
        value = raw.get(key)
        if key == "ts" and isinstance(value, str) and re.fullmatch(r"[0-9T:.+Z-]{1,40}", value):
            filtered[key] = value
        elif key in {"level", "service_instance_id"} and isinstance(value, str) and _ID.fullmatch(value):
            filtered[key] = value
        elif key == "pid" and isinstance(value, int):
            filtered[key] = value
    if "error_chain" in raw:
        diagnostics = event_record(event, diagnostics=raw)
        if "error_chain" in diagnostics:
            filtered["error_chain"] = diagnostics["error_chain"]
    return filtered


def trace_job(job_id: str, log_dir: Path) -> dict[str, object]:
    if not _ID.fullmatch(job_id):
        raise ValueError("invalid job id")
    traces = {event.get("trace_id") for event in _events(log_dir) if event.get("job_id") == job_id}
    timeline: list[dict[str, object]] = []
    size = 0
    truncated = False
    for raw in _events(log_dir):
        if raw.get("job_id") != job_id and (not traces or raw.get("trace_id") not in traces):
            continue
        event = _public_event(raw)
        event_size = len(json.dumps(event).encode("utf-8"))
        if size + event_size > MAX_OUTPUT_BYTES:
            truncated = True
            break
        size += event_size
        timeline.append(event)
    timeline.sort(key=lambda event: str(event.get("ts", "")))
    errors = [event for event in timeline if event.get("level") in {"ERROR", "CRITICAL"} or "error_code" in event]
    successes = [event for event in timeline if str(event.get("event", "")).endswith((".completed", ".loaded", ".normalized", ".reused"))]
    return {
        "job_id": job_id, "timeline": timeline, "last_successful_stage": successes[-1] if successes else None,
        "errors": errors, "truncated": truncated, "events": len(timeline),
        "input_limit_bytes": MAX_INPUT_BYTES, "output_limit_bytes": MAX_OUTPUT_BYTES,
    }


def _task_summary(job_id: str, data_dir: Path) -> dict[str, object]:
    database = data_dir / "workbench.sqlite3"
    if not database.is_file():
        return {"available": False}
    # mode=ro prevents accidental creation or writes; do not instantiate Store.
    fields = ("id", "status", "created_at", "started_at", "completed_at", "attempt")
    parameters = ("model", "quality_profile", "language", "max_new_tokens", "max_chunk_seconds", "overlap_seconds", "timestamps", "split_strategy")
    # Project inside SQLite: task results may be large and must never be loaded
    # merely to create a content-free diagnostic summary.
    paths = ["$." + field for field in fields] + ["$.request." + field for field in parameters]
    projection = ",".join("substr(json_extract(payload,?),1,160)" for _ in paths)
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute("SELECT " + projection + " FROM jobs WHERE id=?", (*paths, job_id)).fetchone()
    if row is None:
        return {"available": False}
    summary: dict[str, object] = {"available": True}
    for name, value in zip((*fields, *parameters), row):
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:+Z-]{1,160}", value):
            summary[name] = value
    return summary


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "platform": platform.system()}
    for name in ("asr-server", "fastapi", "starlette", "httpx", "transformers", "torch"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def create_bundle(job_id: str, log_dir: Path, data_dir: Path, output: Path, *, max_bytes: int = 16 * 1024 * 1024) -> None:
    if output.exists():
        raise ValueError("output already exists")
    trace = trace_job(job_id, log_dir)
    summary = _task_summary(job_id, data_dir)
    # Checkpoint contents contain text. Include only bounded, relative names and
    # sizes from the selected job directory, never checkpoint payloads.
    checkpoints: list[dict[str, object]] = []
    root = data_dir / "jobs" / job_id / "checkpoints"
    if root.is_dir() and not root.is_symlink():
        for path in root.glob("*.json"):
            if len(checkpoints) >= 4096:
                break
            if path.is_file() and not path.is_symlink() and _ID.fullmatch(path.name):
                checkpoints.append({"name": path.name, "bytes": path.stat().st_size})
    entries = {
        "trace.json": trace, "task-summary.json": summary,
        "runtime-versions.json": _versions(), "checkpoints.json": checkpoints,
        "manifest.json": {"schema_version": 1, "job_id": job_id, "content": "content-free diagnostics", "max_bytes": max_bytes},
    }
    # Bound both uncompressed content and final archive before publishing output.
    total = 0
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in entries.items():
            content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            total += len(content)
            if total > max_bytes:
                raise ValueError("diagnostic bundle exceeds size limit")
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(content), 0o600
            archive.addfile(info, io.BytesIO(content))
    if buffer.tell() > max_bytes:
        raise ValueError("diagnostic archive exceeds size limit")
    with output.open("xb") as stream:
        output.chmod(0o600)
        stream.write(buffer.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("trace", "bundle"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--job", required=True)
        sub.add_argument("--log-dir", type=Path, required=True)
        if action == "trace":
            sub.add_argument("--format", choices=("json", "text"), default="json")
        else:
            sub.add_argument("--data-dir", type=Path, required=True)
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "bundle":
            create_bundle(args.job, args.log_dir, args.data_dir, args.output)
            print("Created content-free diagnostic bundle")
        else:
            trace = trace_job(args.job, args.log_dir)
            if args.format == "json":
                print(json.dumps(trace, ensure_ascii=False, indent=2))
            else:
                print(f"Job {args.job}: {trace['events']} events; truncated={trace['truncated']}")
                for event in cast(list[dict[str, object]], trace["timeline"]):
                    print(f"{event.get('ts', '')} {event.get('event', '')} duration_ms={event.get('duration_ms', '')} error={event.get('error_code', '')}")
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(f"diagnostics unavailable: {type(exc).__name__}")


if __name__ == "__main__":
    main()
