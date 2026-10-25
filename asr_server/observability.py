"""Bounded, content-free application events; only the gateway writes log files.

Contexts cross background queues explicitly. ContextVars are a local convenience,
not a substitute for attaching an ExecutionContext to a scheduled operation.
"""
from __future__ import annotations

import contextvars
import io
import json
import logging
import logging.handlers
import math
import os
import queue
import re
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast


SCHEMA_VERSION = 1
PROTOCOL_VERSION = 1
MAX_RECORD_BYTES = 8192
SERVICE_INSTANCE_ID = "svc_" + uuid.uuid4().hex
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_EVENT = re.compile(r"^[a-z][a-z0-9_.]{0,95}$")
_context: contextvars.ContextVar[ExecutionContext | None] = contextvars.ContextVar("asr_execution", default=None)
_logger = logging.getLogger("asr.events")
_logger.propagate = False
_logger.addHandler(logging.NullHandler())
_runtime: LoggingRuntime | None = None


def _identifier(value: object) -> str | None:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


@dataclass(frozen=True)
class ExecutionContext:
    trace_id: str
    request_id: str | None = None
    job_id: str | None = None
    attempt: int = 1
    model_id: str | None = None
    engine_id: str | None = None
    pipeline_id: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    @classmethod
    def new(cls, **fields: Any) -> ExecutionContext:
        return cls(trace_id="tr_" + uuid.uuid4().hex, request_id="req_" + uuid.uuid4().hex).with_fields(**fields)

    def with_fields(self, **fields: Any) -> ExecutionContext:
        allowed = {name: value for name, value in fields.items() if name in self.__dataclass_fields__}
        return replace(self, **allowed)

    def to_wire(self) -> dict[str, object]:
        fields: dict[str, object] = {"attempt": max(1, self.attempt)}
        for name in ("trace_id", "request_id", "job_id", "model_id", "engine_id", "pipeline_id"):
            value = _identifier(getattr(self, name))
            if value is not None:
                fields[name] = value
        return fields

    @classmethod
    def from_wire(cls, value: object) -> ExecutionContext:
        fields = value if isinstance(value, dict) else {}
        trace_id = _identifier(fields.get("trace_id")) or "tr_" + uuid.uuid4().hex
        attempt = fields.get("attempt", 1)
        return cls(
            trace_id=trace_id, request_id=_identifier(fields.get("request_id")),
            job_id=_identifier(fields.get("job_id")), attempt=attempt if isinstance(attempt, int) and attempt > 0 else 1,
            model_id=_identifier(fields.get("model_id")), engine_id=_identifier(fields.get("engine_id")),
            pipeline_id=_identifier(fields.get("pipeline_id")),
        )


def current_context() -> ExecutionContext:
    return _context.get() or ExecutionContext.new()


@contextmanager
def bind_context(context: ExecutionContext) -> Iterator[ExecutionContext]:
    token = _context.set(context)
    try:
        yield context
    finally:
        _context.reset(token)


def safe_exception(exc: BaseException) -> dict[str, object]:
    """Stack locations without messages, source lines, locals or full paths."""
    chain: list[dict[str, object]] = []
    seen: set[int] = set()
    item: BaseException | None = exc
    while item is not None and id(item) not in seen and len(chain) < 4:
        seen.add(id(item))
        frames = [
            {"file": Path(frame.filename).name[:80], "line": frame.lineno, "function": frame.name[:80]}
            for frame in traceback.extract_tb(item.__traceback__, limit=12)
        ]
        chain.append({"error_type": type(item).__name__, "frames": frames})
        item = item.__cause__ or (None if item.__suppress_context__ else item.__context__)
    return {"error_type": type(exc).__name__, "error_chain": chain}


# Free text (including messages, paths, prompts and request bodies) is deliberately
# absent. New call sites must choose a typed/small diagnostic field here.
_NUMERIC_FIELDS = frozenset({
    "status_code", "read_bytes", "input_bytes", "output_bytes", "reserved_bytes", "released_bytes",
    "remaining_bytes", "budget_bytes", "duration_ms", "queue_wait_ms", "duration_s", "audio_seconds",
    "chunk_index", "chunk_count", "completed_chunks", "total_chunks", "start_s", "end_s", "attempt",
    "sequence", "worker_pid", "worker_generation", "rpc_request_id", "deadline_seconds", "exit_code",
    "generated_tokens", "prompt_tokens", "max_new_tokens", "sample_rate", "channels", "frames",
    "queue_depth", "dropped_count", "log_queue_dropped_total", "review_budget_seconds", "version",
})
_ID_FIELDS = frozenset({
    "trace_id", "request_id", "job_id", "model_id", "engine_id", "pipeline_id", "lease_id", "asset_id",
    "reason", "error_code", "error_type", "phase", "operation", "strategy", "requested_strategy",
    "decoder_exit", "normalization_version", "model_revision", "runtime_profile_id", "asset_type",
    "method", "fingerprint", "backend", "dtype", "device", "status", "cancelled", "owner_type",
})
_CRITICAL_EVENTS = frozenset({"job.queued", "job.started", "job.completed", "job.cancelled", "job.failed", "request.rejected"})


def _diagnostics(value: object) -> dict[str, object]:
    """Validate untrusted worker diagnostics a second time in the parent."""
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, object] = {}
    error_type = _identifier(value.get("error_type"))
    if error_type:
        result["error_type"] = error_type
    chain_value = value.get("error_chain")
    chain: list[dict[str, object]] = []
    if isinstance(chain_value, list):
        for item in chain_value[:4]:
            if not isinstance(item, Mapping):
                continue
            frames: list[dict[str, object]] = []
            frame_values = item.get("frames")
            if isinstance(frame_values, list):
                for frame in frame_values[-12:]:
                    if not isinstance(frame, Mapping):
                        continue
                    filename, function, line = frame.get("file"), frame.get("function"), frame.get("line")
                    if isinstance(filename, str) and isinstance(function, str) and isinstance(line, int):
                        frames.append({"file": Path(filename.replace("\\", "/")).name[:80], "function": re.sub(r"[^A-Za-z0-9_<>.]", "_", function)[:80], "line": line})
            chain.append({"error_type": _identifier(item.get("error_type")) or "Exception", "frames": frames})
    if chain:
        result["error_chain"] = chain
    return result


def event_record(event: str, *, level: int = logging.INFO, **fields: object) -> dict[str, object]:
    context = current_context()
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION, "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "level": logging.getLevelName(level), "event": event if _EVENT.fullmatch(event) else "event.invalid",
        "service_instance_id": SERVICE_INSTANCE_ID, "pid": os.getpid(), **context.to_wire(),
    }
    for name, value in fields.items():
        if name in _NUMERIC_FIELDS and isinstance(value, (int, float)) and math.isfinite(value):
            record[name] = value
        elif name in _ID_FIELDS:
            if isinstance(value, bool):
                record[name] = value
            elif (identifier := _identifier(value)) is not None:
                record[name] = identifier
        elif name == "route" and isinstance(value, str) and re.fullmatch(r"/[A-Za-z0-9_/{}/.:-]{0,180}", value):
            record[name] = value
        elif name == "diagnostics":
            record.update(_diagnostics(value))
    return record


def emit(event: str, *, level: int = logging.INFO, **fields: object) -> None:
    if _logger.isEnabledFor(level):
        _logger.log(level, "", extra={"asr_event": event_record(event, level=level, **fields)})


@contextmanager
def stage(event: str, **fields: object) -> Iterator[None]:
    started = time.perf_counter()
    emit(event + ".started", level=logging.INFO, **fields)
    try:
        yield
    except BaseException as exc:
        emit(event + ".failed", level=logging.ERROR, duration_ms=(time.perf_counter() - started) * 1000,
             diagnostics=safe_exception(exc), **fields)
        raise
    else:
        emit(event + ".completed", level=logging.INFO, duration_ms=(time.perf_counter() - started) * 1000, **fields)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = dict(getattr(record, "asr_event", {}))
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_RECORD_BYTES:
            payload.pop("error_chain", None)
            payload["record_truncated"] = True
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_RECORD_BYTES:
            payload = {name: payload[name] for name in ("schema_version", "ts", "level", "event", "trace_id") if name in payload}
            payload["record_truncated"] = True
            encoded = json.dumps(payload, separators=(",", ":"))
        return encoded


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        return rendered.encode("utf-8")[:MAX_RECORD_BYTES].decode("utf-8", errors="ignore")


class BoundedQueueHandler(logging.handlers.QueueHandler):
    def __init__(self, target: queue.Queue[logging.LogRecord], *, reserve: int = 128) -> None:
        super().__init__(target)
        self.pending = target
        self.reserve = min(max(0, reserve), max(0, target.maxsize - 1))
        self.dropped_count = 0

    def enqueue(self, record: logging.LogRecord) -> None:
        payload = getattr(record, "asr_event", {})
        critical = record.levelno >= logging.WARNING or payload.get("event") in _CRITICAL_EVENTS
        if not critical and self.pending.qsize() >= self.pending.maxsize - self.reserve:
            self.dropped_count += 1
            return
        payload["log_queue_dropped_total"] = self.dropped_count
        try:
            self.pending.put_nowait(record)
        except queue.Full:
            self.dropped_count += 1


class _PrivateRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """One writer, a daily/size boundary and conservative whole-file expiry."""

    def __init__(self, filename: str | Path, *, maxBytes: int, backupCount: int,
                 encoding: str, clock: Callable[[], float] = time.time) -> None:
        super().__init__(filename, maxBytes=maxBytes, backupCount=backupCount, encoding=encoding)
        self._clock = clock
        self._bucket_started = self._first_record_time(Path(self.baseFilename))
        self._next_maintenance = 0.0
        self.maintain()

    def _first_record_time(self, path: Path) -> float:
        try:
            with path.open("rb") as stream:
                line = stream.readline(MAX_RECORD_BYTES + 1)
            if line:
                parsed = json.loads(line)
                value = parsed.get("ts") if isinstance(parsed, dict) else None
                if isinstance(value, str):
                    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            return path.stat().st_mtime
        except (OSError, ValueError, TypeError):
            # Unknown/corrupt historical data is never granted a fresh lifetime.
            return 0.0

    def _open(self) -> io.TextIOWrapper:
        fd = os.open(self.baseFilename, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        os.chmod(self.baseFilename, 0o600)
        return os.fdopen(fd, "a", encoding="utf-8")

    def doRollover(self) -> None:
        super().doRollover()
        self._bucket_started = self._clock()
        self._purge()

    def _purge(self) -> None:
        # The idle listener wakes within 60 s. Expire 120 s early so even idle
        # files cannot cross the seven-day bound between maintenance ticks.
        cutoff = self._clock() - (7 * 86400 - 120)
        for index in range(1, self.backupCount + 1):
            path = Path(self.baseFilename + "." + str(index))
            if path.is_file() and self._first_record_time(path) < cutoff:
                path.unlink(missing_ok=True)

    def maintain(self) -> None:
        now = self._clock()
        if now < self._next_maintenance:
            return
        self._next_maintenance = now + 60
        if now >= self._bucket_started + 86400:
            if Path(self.baseFilename).stat().st_size:
                self.doRollover()
            else:
                self._bucket_started = now
        self._purge()

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        self.maintain()
        if self.stream is None:
            self.stream = self._open()
        self.stream.seek(0, 2)
        # RotatingFileHandler counts characters. Our limit is physical UTF-8
        # bytes, so a Chinese event cannot grow a file past its granted budget.
        encoded_size = len((self.format(record) + self.terminator).encode("utf-8"))
        return self.maxBytes > 0 and self.stream.tell() + encoded_size > self.maxBytes


class _Listener(logging.handlers.QueueListener):
    _sentinel: None = None

    def __init__(self, pending: queue.Queue[logging.LogRecord], *handlers: logging.Handler,
                 respect_handler_level: bool = False) -> None:
        super().__init__(pending, *handlers, respect_handler_level=respect_handler_level)
        self.pending = cast(queue.Queue[Any], pending)

    def dequeue(self, block: bool) -> logging.LogRecord:
        while True:
            try:
                return cast(logging.LogRecord, self.pending.get(block, timeout=60 if block else None))
            except queue.Empty:
                for handler in self.handlers:
                    if isinstance(handler, _PrivateRotatingFileHandler):
                        handler.maintain()
                if not block:
                    raise

    def enqueue_sentinel(self) -> None:
        # Stop is only called at gateway shutdown, outside any model operation.
        # Ensure a saturated queue cannot prevent the listener from terminating.
        try:
            self.pending.put(self._sentinel, timeout=1)
        except queue.Full:
            try:
                self.pending.get_nowait()
                self.pending.task_done()
            except queue.Empty:
                pass
            self.pending.put_nowait(self._sentinel)


class LoggingRuntime:
    def __init__(self, handler: BoundedQueueHandler, listener: _Listener, sink: logging.Handler) -> None:
        self.handler, self.listener, self.sink = handler, listener, sink
        self.closed = False

    def close(self) -> None:
        global _runtime
        if self.closed:
            return
        self.closed = True
        _logger.removeHandler(self.handler)
        self.listener.stop()
        self.sink.close()
        self.handler.close()
        if _runtime is self:
            _runtime = None


def configure_logging(*, level: str = "INFO", log_format: str = "json", log_dir: str | Path | None = None,
                      queue_size: int = 4096) -> LoggingRuntime:
    global _runtime
    if _runtime is not None:
        _runtime.close()
    sink: logging.Handler
    if log_dir:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        sink = _PrivateRotatingFileHandler(directory / "events.jsonl", maxBytes=10 * 1024 * 1024, backupCount=9, encoding="utf-8")
    else:
        sink = logging.StreamHandler(sys.stderr)
    # Text output stays structured and content-free, too.
    sink.setFormatter(JsonFormatter() if log_format == "json" else _TextFormatter("%(asr_event)s"))
    pending: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=max(2, queue_size))
    handler = BoundedQueueHandler(pending, reserve=min(128, max(1, queue_size // 8)))
    listener = _Listener(pending, sink, respect_handler_level=True)
    runtime = LoggingRuntime(handler, listener, sink)
    _logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    _logger.addHandler(handler)
    listener.start()
    _runtime = runtime
    return runtime


def log_stats() -> dict[str, int]:
    runtime = _runtime
    return {"dropped_count": runtime.handler.dropped_count if runtime else 0,
            "queue_depth": runtime.handler.pending.qsize() if runtime else 0}
