from __future__ import annotations

import logging
import multiprocessing
import threading
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from asr_server.errors import AsrError
from asr_server.observability import PROTOCOL_VERSION, current_context, emit
from asr_server.workers.codec import request_envelope


logger = logging.getLogger(__name__)


class ProcessRpcTransport:
    """Small, fail-closed RPC transport for one spawned model worker.

    A transport generation owns exactly one Process and one Connection. Any
    timeout, EOF, broken pipe, protocol mismatch, or CUDA availability error
    poisons that generation and synchronously reaps it before returning.
    """

    # A surviving generation may still own CUDA. Retain its Process handle
    # across all adapters until exit has been proved, including other models.
    _quarantined: dict[int, Any] = {}
    _quarantine_lock = threading.RLock()

    @classmethod
    def quarantine_status(cls) -> dict[str, object]:
        with cls._quarantine_lock:
            for pid, process in list(cls._quarantined.items()):
                if not process.is_alive():
                    process.join(timeout=0)
                    del cls._quarantined[pid]
            return {"isolated": bool(cls._quarantined), "unreaped_workers": len(cls._quarantined)}

    @classmethod
    def assert_gpu_safe(cls) -> None:
        if cls.quarantine_status()["isolated"]:
            raise AsrError(503, "worker_isolated", "previous worker has not exited; GPU inference is isolated",
                           {"worker_fatal": True, "worker_reaped": False, "retryable": False})

    def configure_deadlines(self, *, chunk: float, alignment: float, native: float,
                            observation: float = 300.0) -> None:
        self._phase_deadlines = {"chunk": chunk, "alignment": alignment, "native": native}
        self._observation_seconds = observation

    def _phase_deadline(self, op: str, timeout: float, payload: dict[str, object]) -> tuple[str, float]:
        phase = "load" if op == "load" else "control"
        if op == "align_text":
            phase = "alignment"
        elif op in {"transcribe", "transcribe_batch"}:
            phase = "native"
            # Qwen short windows and native MOSS calls have different runtime
            # envelopes. Unknown or unbounded input keeps the native deadline.
            audio = payload.get("audio")
            start, end = getattr(audio, "start", None), getattr(audio, "end", None)
            if self.name.startswith("Qwen") and isinstance(start, (int, float)) and isinstance(end, (int, float)):
                if 0 <= end - start <= 180:
                    phase = "chunk"
        return phase, self._phase_deadlines.get(phase, timeout)

    def __init__(
        self,
        *,
        name: str,
        target: Callable[..., None],
        target_args: tuple[object, ...],
        startup_timeout_seconds: float = 30.0,
        terminate_timeout_seconds: float = 5.0,
        kill_timeout_seconds: float = 5.0,
    ) -> None:
        self.name = name
        self._target = target
        self._target_args = target_args
        self._startup_timeout_seconds = startup_timeout_seconds
        self._terminate_timeout_seconds = terminate_timeout_seconds
        self._kill_timeout_seconds = kill_timeout_seconds
        self._process: Any | None = None
        self._conn: Connection | None = None
        self._request_id = 0
        self._generation = 0
        self._phase_deadlines = {}
        self._observation_seconds = 300.0
        self._poisoned = False

    @property
    def pid(self) -> int | None:
        process = self._process
        return None if process is None else process.pid

    @property
    def is_alive(self) -> bool:
        process = self._process
        return bool(process is not None and process.is_alive())

    def start(self) -> None:
        self.assert_gpu_safe()
        if self.is_alive and self._conn is not None and not self._poisoned:
            return
        self.close(force=True)
        self.assert_gpu_safe()
        self._poisoned = False
        context = multiprocessing.get_context("spawn")
        parent_conn, child_conn = context.Pipe()
        process = context.Process(
            target=self._target,
            args=(child_conn, *self._target_args),
            # A non-daemon worker gets a normal interpreter shutdown path, which
            # lets model/runtime libraries release multiprocessing resources.
            # The transport still enforces terminate/kill deadlines.
            daemon=False,
        )
        process.start()
        child_conn.close()
        self._process = process
        self._conn = parent_conn
        self._generation += 1
        result = self.request("ping", timeout_seconds=self._startup_timeout_seconds)
        # Historical/custom workers may omit it; an explicitly incompatible
        # protocol must never continue into expensive model operations.
        if isinstance(result, dict) and result.get("protocol_version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
            self._fatal("worker_protocol_version", op="ping")
        emit("worker.started", worker_pid=self.pid, worker_generation=self._generation, engine_id=self.name)

    def request(self, op: str, *, timeout_seconds: float, **payload: object) -> Any:
        conn = self._conn
        process = self._process
        if self._poisoned or conn is None or process is None or not process.is_alive():
            self._fatal("worker_not_running", op=op)
        assert conn is not None
        assert process is not None
        self._request_id += 1
        request_id = self._request_id
        worker_pid = process.pid
        started = time.perf_counter()
        phase, timeout_seconds = self._phase_deadline(op, timeout_seconds, payload)
        try:
            conn.send(request_envelope(request_id, op, payload, current_context()))
            observation = min(self._observation_seconds, timeout_seconds)
            ready = conn.poll(observation)
            if not ready and observation < timeout_seconds:
                emit("worker.deadline_observed", operation=op, phase=phase, worker_pid=worker_pid,
                     worker_generation=self._generation, duration_ms=observation * 1000,
                     deadline_seconds=timeout_seconds)
                ready = conn.poll(max(0.0, timeout_seconds - (time.perf_counter() - started)))
            if not ready:
                self._fatal(
                    "worker_timeout",
                    op=op,
                    request_id=request_id,
                    worker_pid=worker_pid,
                    details={"timeout_seconds": timeout_seconds, "phase": phase},
                )
            response = conn.recv()
        except AsrError:
            raise
        except (BrokenPipeError, EOFError, OSError) as exc:
            self._fatal(
                "worker_pipe_closed",
                op=op,
                request_id=request_id,
                worker_pid=worker_pid,
                details={"error_type": type(exc).__name__},
            )
        if not isinstance(response, dict) or response.get("id") != request_id:
            self._fatal(
                "worker_protocol_error",
                op=op,
                request_id=request_id,
                worker_pid=worker_pid,
            )
        if response.get("ok"):
            emit("worker.rpc_completed", level=logging.DEBUG, operation=op, rpc_request_id=request_id,
                 worker_pid=worker_pid, worker_generation=self._generation,
                 phase=phase, deadline_seconds=timeout_seconds,
                 duration_ms=(time.perf_counter() - started) * 1000)
            return response.get("result")
        raw_error = response.get("error")
        error = raw_error if isinstance(raw_error, dict) else {}
        details_value = error.get("details")
        details = dict(details_value) if isinstance(details_value, dict) else {}
        details.update({"operation": op, "request_id": request_id, "worker_pid": worker_pid})
        code = str(error.get("code", "inference_failed"))
        status_code = int(error.get("status_code", 503))
        emit("worker.rpc_failed", level=logging.ERROR, operation=op, rpc_request_id=request_id,
             worker_pid=worker_pid, worker_generation=self._generation, error_code=code,
             diagnostics=response.get("diagnostics"), duration_ms=(time.perf_counter() - started) * 1000)
        if status_code >= 500:
            details["worker_fatal"] = True
            self._poisoned = True
            self.close(force=True)
            details["worker_reaped"] = self._process is None
        raise AsrError(
            status_code,
            code,
            str(error.get("message", f"{self.name} worker failed")),
            details,
        )

    def shutdown(self, *, timeout_seconds: float, cuda_empty_cache: bool = True) -> None:
        if self._process is None:
            self._close_connection()
            return
        if self.is_alive and self._conn is not None:
            try:
                self.request(
                    "shutdown",
                    timeout_seconds=timeout_seconds,
                    cuda_empty_cache=cuda_empty_cache,
                )
            except AsrError as exc:
                emit("worker.shutdown_failed", level=logging.WARNING, engine_id=self.name, error_code=exc.code)
        process = self._process
        if process is not None and process.is_alive():
            # The worker acknowledges before its outer finally block closes CUDA
            # and multiprocessing resources. Give that path time to finish.
            process.join(timeout=timeout_seconds)
        self.close(force=True)
        if self._process is not None:
            raise AsrError(503, "worker_isolated", "model worker could not be reaped during shutdown",
                           {"worker_fatal": True, "worker_reaped": False, "retryable": False})

    def close(self, *, force: bool) -> None:
        process = self._process
        self._close_connection()
        if process is None:
            return
        self._poisoned = True
        process.join(timeout=0)
        if force and process.is_alive():
            try:
                process.terminate()
                process.join(timeout=self._terminate_timeout_seconds)
            except OSError:
                logger.exception("worker terminate failed")
        if force and process.is_alive():
            try:
                process.kill()
                process.join(timeout=self._kill_timeout_seconds)
            except OSError:
                logger.exception("worker kill failed")
        if process.is_alive():
            with self._quarantine_lock:
                self._quarantined[process.pid] = process
            logger.error("failed to reap %s worker pid=%s; GPU isolated", self.name, process.pid)
            return
        with self._quarantine_lock:
            self._quarantined.pop(process.pid, None)
        emit("worker.reaped", worker_pid=process.pid, worker_generation=self._generation,
             exit_code=process.exitcode, engine_id=self.name)
        process.close()
        self._process = None

    def _close_connection(self) -> None:
        conn = self._conn
        self._conn = None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    def _fatal(
        self,
        reason: str,
        *,
        op: str,
        request_id: int | None = None,
        worker_pid: int | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        payload = {
            "operation": op,
            "request_id": request_id,
            "worker_pid": worker_pid if worker_pid is not None else self.pid,
            "worker_fatal": True,
            **(details or {}),
        }
        emit("worker.rpc_failed", level=logging.ERROR, operation=op, rpc_request_id=request_id,
             worker_pid=payload["worker_pid"], worker_generation=self._generation, error_code=reason)
        logger.error(
            "%s worker fatal reason=%s operation=%s request_id=%s pid=%s",
            self.name,
            reason,
            op,
            request_id,
            payload["worker_pid"],
        )
        self._poisoned = True
        self.close(force=True)
        payload["worker_reaped"] = self._process is None
        raise AsrError(503, "worker_unavailable", f"{self.name} worker failure: {reason}", payload)
