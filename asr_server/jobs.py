from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

from asr_server.adapters.base import TranscriptionResult
from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.audio.workspace import AudioWorkspace, ResourceLease, remaining_workspace_output_bytes
from asr_server.audio.assets import NormalizedAudioAsset
from asr_server.audio.preprocess import normalize_audio_path_to_wav_async
from asr_server.lifecycle import ModelLifecycleManager
from asr_server.registry import Backend
from asr_server.scheduler import OperationScheduler
from asr_server.observability import ExecutionContext, current_context, bind_context, emit, safe_exception
from asr_server.transcription import (
    TranscriptionRequest,
    ValidatedTranscription,
    run_transcription_path,
    validate_transcription_request,
)


JobStatus = Literal[
    "queued",
    "preprocessing",
    "splitting",
    "loading_model",
    "transcribing",
    "merging",
    "completed",
    "failed",
    "cancel_requested",
    "cancelled",
    "expired",
]
logger = logging.getLogger(__name__)

TERMINAL_STATUSES: set[JobStatus] = {"completed", "failed", "cancelled", "expired"}


class JobCancelled(Exception):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")


def _partial_chunks(texts: list[str]) -> list[dict[str, object]]:
    return [{"index": index, "text": text} for index, text in enumerate(texts, start=1)]


def _request_fingerprint(path: Path, request: TranscriptionRequest) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    digest.update(
        json.dumps(asdict(request), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


@dataclass
class JobProgress:
    phase: str
    percent: float = 0.0
    total_chunks: int | None = None
    completed_chunks: int | None = None
    current_chunk: int | None = None
    current_chunk_start: float | None = None
    current_chunk_end: float | None = None
    text: str | None = None
    chunk_text: str | None = None
    chunks: list[dict[str, object]] | None = None
    message: str | None = None

    def to_api(self) -> dict[str, object]:
        payload: dict[str, object] = {"phase": self.phase, "percent": self.percent}
        if self.total_chunks is not None:
            payload["total_chunks"] = self.total_chunks
        if self.completed_chunks is not None:
            payload["completed_chunks"] = self.completed_chunks
        if self.current_chunk is not None:
            payload["current_chunk"] = self.current_chunk
        if self.current_chunk_start is not None:
            payload["current_chunk_start"] = self.current_chunk_start
        if self.current_chunk_end is not None:
            payload["current_chunk_end"] = self.current_chunk_end
        if self.text is not None:
            payload["text"] = self.text
        if self.chunk_text is not None:
            payload["chunk_text"] = self.chunk_text
        if self.chunks is not None:
            payload["chunks"] = self.chunks
        if self.message is not None:
            payload["message"] = self.message
        return payload


@dataclass
class TranscriptionJob:
    id: str
    status: JobStatus
    model: str
    backend: str
    language: str
    request: TranscriptionRequest
    validated: ValidatedTranscription
    upload_path: Path
    work_dir: Path
    created_at: datetime
    expires_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: JobProgress = field(default_factory=lambda: JobProgress(phase="queued", message="waiting for previous transcription jobs"))
    split: dict[str, object] | None = None
    result: dict[str, object] | None = None
    error: dict[str, object] | None = None
    request_summary: dict[str, object] = field(default_factory=dict)
    temp_paths: list[Path] = field(default_factory=list)
    workspace: AudioWorkspace | None = None
    chunk_windows: list[tuple[float, float]] = field(default_factory=list)
    partial_texts: list[str] = field(default_factory=list)
    idempotency_key: str | None = None
    request_fingerprint: str | None = None
    execution_context: ExecutionContext = field(default_factory=ExecutionContext.new)
    normalized_asset: NormalizedAudioAsset | None = None
    preparation_lease: ResourceLease | None = None
    preparation_error: BaseException | None = None


class JobManager:
    def __init__(self, manager: ModelLifecycleManager, settings: Settings) -> None:
        self._manager = manager
        self._settings = settings
        self._jobs: dict[str, TranscriptionJob] = {}
        self._idempotency: dict[str, str] = {}
        self.scheduler = OperationScheduler(settings.max_queued_jobs, prefetch=settings.preprocessing_prefetch)
        self._queue = self.scheduler.queue
        self._external: dict[str, Callable[[], Awaitable[None]]] = {}
        self._running_external: str | None = None
        self._condition = asyncio.Condition()
        self._worker: asyncio.Task[None] | None = None
        self._sweeper: asyncio.Task[None] | None = None
        self._closed = False
        self._stale_temp_dirs_cleaned = False

    async def start(self) -> None:
        if not self._stale_temp_dirs_cleaned:
            await asyncio.to_thread(self._cleanup_stale_temp_dirs)
            self._stale_temp_dirs_cleaned = True
        await self.scheduler.start()
        self._closed = False
        self._worker = self.scheduler.worker
        if self._sweeper is None or self._sweeper.done():
            self._sweeper = asyncio.create_task(self._expiry_loop())

    async def shutdown(self) -> None:
        self._closed = True
        queued_ids = list(self._queue)
        for job in self._jobs.values():
            if job.status not in TERMINAL_STATUSES:
                job.status = "cancel_requested"
                job.execution_context.cancel_event.set()
        await self.scheduler.shutdown(self._settings.job_shutdown_grace_seconds, self._manager.abort_all_workers)
        self._worker = None
        for job_id in queued_ids:
            queued_job = self._jobs.get(job_id)
            if queued_job is not None:
                await self._cleanup_cancelled_job(queued_job)
        if self._sweeper is not None:
            self._sweeper.cancel()
            try:
                await self._sweeper
            except asyncio.CancelledError:
                pass
            self._sweeper = None
        for job in list(self._jobs.values()):
            await self._cleanup_job_files(job)
        self._external.clear()

    async def enqueue_external(self, key: str, operation: Callable[[], Awaitable[None]],
                               *, context: ExecutionContext | None = None,
                               prepare: Callable[[], Awaitable[None]] | None = None,
                               dispose: Callable[[], Awaitable[None]] | None = None) -> None:
        await self.start()
        if key in self._jobs:
            raise AsrError(409, "job_id_conflict", "job identifier already exists")
        if key in self._external:
            return
        async def owned() -> None:
            self._running_external = key
            try:
                await operation()
            finally:
                self._external.pop(key, None)
                self._running_external = None
        await self.scheduler.submit(key, owned, kind="workbench", context=context, prepare=prepare, dispose=dispose)
        self._external[key] = operation

    def external_queue_position(self, key: str) -> int:
        return self._queue_position(key)

    async def cancel_external(self, key: str) -> bool:
        removed = await self.scheduler.cancel(key)
        if removed:
            self._external.pop(key, None)
        return removed

    async def create_job(
        self,
        *,
        workspace: AudioWorkspace,
        filename: str | None,
        content_type: str | None,
        request: TranscriptionRequest,
    ) -> dict[str, object]:
        await self.start()
        self._expire_jobs()
        validated = validate_transcription_request(self._manager, request)
        if request.idempotency_key is not None and len(request.idempotency_key) > 200:
            raise AsrError(400, "bad_request", "idempotency_key exceeds 200 characters")
        request_fingerprint = await asyncio.to_thread(
            _request_fingerprint,
            workspace.upload_path,
            request,
        )
        job_id = f"job_{uuid4().hex}"
        created_at = utc_now()
        upload_name = filename or "upload.bin"
        async with self._condition:
            if request.idempotency_key:
                existing_id = self._idempotency.get(request.idempotency_key)
                existing = self._jobs.get(existing_id) if existing_id is not None else None
                if existing is not None:
                    if existing.request_fingerprint != request_fingerprint:
                        raise AsrError(
                            409,
                            "idempotency_conflict",
                            "idempotency_key was already used with different audio or parameters",
                        )
                    await workspace.cleanup()
                    return {**self._job_create_payload(existing), "deduplicated": True}
            if self._active_job_count() >= self._settings.max_queued_jobs:
                raise AsrError(
                    429,
                    "job_queue_full",
                    "too many queued or running transcription jobs",
                    {"max_queued_jobs": self._settings.max_queued_jobs},
                )
            work_dir = workspace.root
            upload_path = workspace.upload_path
            job = TranscriptionJob(
                id=job_id,
                status="queued",
                model=validated.selected_model,
                backend=validated.resolved_backend,
                language=request.language,
                request=request,
                validated=validated,
                upload_path=upload_path,
                work_dir=work_dir,
                created_at=created_at,
                request_summary={
                    "filename": upload_name,
                    "content_type": content_type,
                    "size_bytes": workspace.size_bytes,
                    "response_format": request.response_format,
                    "purpose": request.purpose,
                    "timestamps": request.timestamps,
                    "split_strategy": request.split_strategy,
                    "max_chunk_seconds": request.max_chunk_seconds,
                    "overlap_seconds": request.overlap_seconds,
                    "preserve_segments": request.preserve_segments,
                    "context_chars": len(request.context),
                    "hotwords_chars": len(request.hotwords or ""),
                    "context": validated.context.request_summary(),
                    "pause_threshold_seconds": request.pause_threshold_seconds,
                    "idempotency_key": request.idempotency_key,
                },
                temp_paths=[upload_path, work_dir],
                workspace=workspace,
                idempotency_key=request.idempotency_key,
                request_fingerprint=request_fingerprint,
                execution_context=current_context().with_fields(job_id=job_id, model_id=validated.selected_model, pipeline_id=request.quality_profile),
            )
            self._jobs[job_id] = job
            if request.idempotency_key:
                self._idempotency[request.idempotency_key] = job_id
            try:
                await self.scheduler.submit(job_id, lambda: self._run_job(job), context=job.execution_context,
                    prepare=(lambda: self._prepare_job(job)) if self._settings.preprocessing_prefetch else None,
                    dispose=lambda: self._cleanup_cancelled_job(job))
            except BaseException:
                self._jobs.pop(job_id, None)
                if request.idempotency_key:
                    self._idempotency.pop(request.idempotency_key, None)
                raise
            self._condition.notify()
            return self._job_create_payload(job)

    async def get_job(self, job_id: str, *, detail: str = "compact") -> dict[str, object]:
        if detail not in {"compact", "full"}:
            raise AsrError(400, "bad_request", "detail must be compact or full")
        self._expire_jobs()
        job = self._job_or_404(job_id)
        return self._job_payload(job, detail=detail)

    async def cancel_job(self, job_id: str) -> dict[str, object]:
        self._expire_jobs()
        job = self._job_or_404(job_id)
        async with self._condition:
            if job.status == "queued" or self.scheduler.position(job_id) > 0:
                job.execution_context.cancel_event.set()
                await self.scheduler.cancel(job_id)
                await self._cleanup_cancelled_job(job)
                self._condition.notify()
                return {"id": job.id, "status": job.status, "message": "job cancelled"}
            if job.status in TERMINAL_STATUSES:
                return {"id": job.id, "status": job.status, "message": f"job is already {job.status}"}
            job.status = "cancel_requested"
            job.execution_context.cancel_event.set()
            native_long_form = isinstance(job.split, dict) and job.split.get("execution_mode") == "native_long_form"
            message = (
                "cancellation will take effect after the current model invocation finishes"
                if native_long_form
                else "cancellation will take effect after the current chunk finishes"
            )
            job.progress.message = message
            return {
                "id": job.id,
                "status": job.status,
                "message": message,
            }

    def _job_or_404(self, job_id: str) -> TranscriptionJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise AsrError(404, "job_not_found", f"unknown job: {job_id}") from exc

    async def _prepare_job(self, job: TranscriptionJob) -> None:
        # Prefetch owns a full output reservation before starting the decoder.
        # Failure to obtain spare capacity defers decoding to ordinary execution.
        try:
            if job.workspace is None or job.execution_context.cancel_event.is_set():
                return
            spec = self._manager.runtime_for(job.model).definition.audio_spec
            budget = min(await asyncio.to_thread(remaining_workspace_output_bytes, job.work_dir, self._settings),
                         int(21600 * spec.bytes_per_second) + 44)
            try:
                lease = await job.workspace._manager.reserve_resource(budget, job.work_dir)
            except AsrError as exc:
                if exc.code in {"storage_unavailable", "job_queue_full"}:
                    return
                raise
            job.preparation_lease = lease
            normalized = await normalize_audio_path_to_wav_async(
                job.upload_path, job.work_dir / "normalized.wav", spec=spec,
                max_output_bytes=budget, cancel_event=job.execution_context.cancel_event, resource_lease=lease,
                timeout_seconds=self._settings.ffmpeg_timeout_seconds,
            )
            job.normalized_asset = normalized.asset
            await lease.release_unused()
        except BaseException as exc:
            job.preparation_error = exc

    async def _run_job(self, job: TranscriptionJob) -> None:
        emit("job.processing", model_id=job.model)
        job.started_at = utc_now()
        job.status = "preprocessing"
        job.progress = JobProgress(phase="preprocessing", percent=1.0, message="preprocessing uploaded audio")
        final_status: Literal["completed", "failed", "cancelled"] = "cancelled"
        result: dict[str, object] | None = None
        error: dict[str, object] | None = None
        worker_cancelled = False
        try:
            if job.preparation_error is not None:
                raise job.preparation_error
            async def stage_callback(phase: str, update: dict[str, object]) -> None:
                self._raise_if_cancel_requested(job)
                self._update_phase(job, phase, update)

            async def before_chunk(chunk_index: int, total_chunks: int) -> None:
                self._raise_if_cancel_requested(job)
                current = chunk_index + 1
                completed = max(current - 1, 0)
                percent = (completed / total_chunks) * 100 if total_chunks else 0.0
                job.status = "transcribing" if job.status != "cancel_requested" else "cancel_requested"
                job.progress = JobProgress(
                    phase="transcribing",
                    percent=percent,
                    total_chunks=total_chunks,
                    completed_chunks=completed,
                    current_chunk=current,
                    current_chunk_start=job.chunk_windows[chunk_index][0] if chunk_index < len(job.chunk_windows) else None,
                    current_chunk_end=job.chunk_windows[chunk_index][1] if chunk_index < len(job.chunk_windows) else None,
                    text="\n".join(job.partial_texts),
                    chunk_text=job.progress.chunk_text,
                    chunks=_partial_chunks(job.partial_texts),
                    message=f"transcribing chunk {current} of {total_chunks}",
                )

            async def after_chunk(chunk_index: int, total_chunks: int, result: TranscriptionResult) -> None:
                completed = chunk_index + 1
                percent = (completed / total_chunks) * 100 if total_chunks else 100.0
                chunk_text = result.text.strip()
                job.partial_texts.append(chunk_text)
                job.progress = JobProgress(
                    phase="transcribing",
                    percent=percent,
                    total_chunks=total_chunks,
                    completed_chunks=completed,
                    current_chunk=completed,
                    text="\n".join(job.partial_texts),
                    chunk_text=chunk_text,
                    chunks=_partial_chunks(job.partial_texts),
                    message=f"completed chunk {completed} of {total_chunks}",
                )
                self._raise_if_cancel_requested(job)

            result = await run_transcription_path(
                job.upload_path,
                workspace=job.work_dir,
                manager=self._manager,
                settings=self._settings,
                request=job.request,
                validated=job.validated,
                stage_callback=stage_callback,
                before_chunk=before_chunk,
                after_chunk=after_chunk,
                normalized_asset=job.normalized_asset,
                cancel_event=job.execution_context.cancel_event, owner_workspace=job.workspace,
            )
            self._raise_if_cancel_requested(job)
            final_status = "completed"
        except JobCancelled:
            final_status = "cancelled"
        except asyncio.CancelledError:
            job.status = "cancel_requested"
            task = asyncio.current_task()
            worker_cancelled = bool(task and task.cancelling())
        except AsrError as exc:
            final_status = "failed"
            error = {"code": exc.code, "message": exc.message, "details": exc.details}
        except Exception as exc:
            emit("job.failed", diagnostics=safe_exception(exc), error_code="inference_failed")
            final_status = "failed"
            error = {
                "code": "inference_failed",
                "message": "transcription job failed",
                "details": {"error_type": type(exc).__name__},
            }
        finally:
            try:
                await self._cleanup_job_files(job)
            except asyncio.CancelledError:
                # AudioWorkspace completes deletion and budget release before
                # propagating cancellation to its caller.
                job.status = "cancel_requested"
                worker_cancelled = True
        if job.status == "cancel_requested":
            final_status = "cancelled"
        if final_status == "cancelled":
            self._mark_cancelled(job)
        elif final_status == "failed":
            job.error = error
            self._mark_terminal(job, "failed")
        else:
            assert result is not None
            job.result = result
            split = result.get("split")
            job.split = split if isinstance(split, dict) else None
            self._mark_terminal(job, "completed")
            split_count = _int_from_progress(job.progress.total_chunks)
            job.progress = JobProgress(
                phase="completed",
                percent=100.0,
                total_chunks=split_count,
                completed_chunks=split_count,
            )
        emit("job." + job.status, error_code=job.error.get("code") if job.error else None)
        if worker_cancelled:
            raise asyncio.CancelledError

    def _raise_if_cancel_requested(self, job: TranscriptionJob) -> None:
        if job.status == "cancel_requested":
            raise JobCancelled()

    def _update_phase(self, job: TranscriptionJob, phase: str, update: dict[str, object]) -> None:
        if phase in {"preprocessing", "splitting", "loading_model", "transcribing", "merging"}:
            job.status = phase  # type: ignore[assignment]
        split = update.get("split")
        if isinstance(split, dict):
            job.split = split
        chunk_windows = update.get("chunk_windows")
        if isinstance(chunk_windows, list):
            parsed_windows: list[tuple[float, float]] = []
            for item in chunk_windows:
                if isinstance(item, tuple) and len(item) == 2:
                    parsed_windows.append((float(item[0]), float(item[1])))
            job.chunk_windows = parsed_windows
        percent = _optional_float(update.get("percent")) or job.progress.percent
        total_chunks = _optional_int(update.get("total_chunks"))
        completed_chunks = _optional_int(update.get("completed_chunks"))
        native_long_form = (
            phase == "transcribing"
            and isinstance(job.split, dict)
            and job.split.get("execution_mode") == "native_long_form"
        )
        job.progress = JobProgress(
            phase=phase,
            percent=percent,
            total_chunks=None
            if native_long_form
            else total_chunks if total_chunks is not None else job.progress.total_chunks,
            completed_chunks=None
            if native_long_form
            else completed_chunks if completed_chunks is not None else job.progress.completed_chunks,
            text="\n".join(job.partial_texts) if job.partial_texts else None,
            chunk_text=job.progress.chunk_text,
            chunks=_partial_chunks(job.partial_texts) if job.partial_texts else None,
            message=_phase_message(phase),
        )

    def _mark_cancelled(self, job: TranscriptionJob) -> None:
        self._mark_terminal(job, "cancelled")
        job.progress.message = "job cancelled"

    def _mark_terminal(
        self,
        job: TranscriptionJob,
        status: Literal["completed", "failed", "cancelled"],
    ) -> None:
        job.status = status
        job.completed_at = utc_now()
        job.expires_at = job.completed_at + timedelta(seconds=self._settings.job_result_ttl_seconds)
        job.progress.phase = status

    async def _cleanup_cancelled_job(self, job: TranscriptionJob) -> None:
        job.status = "cancel_requested"
        try:
            await self._cleanup_job_files(job)
        except asyncio.CancelledError:
            self._mark_cancelled(job)
            raise
        self._mark_cancelled(job)

    def _job_create_payload(self, job: TranscriptionJob) -> dict[str, object]:
        return {
            "id": job.id,
            "status": job.status,
            "model": job.model,
            "backend": job.backend,
            "queue_position": self._queue_position(job.id),
            "created_at": utc_iso(job.created_at),
            "status_url": f"/v1/jobs/{job.id}",
        }

    def _job_payload(self, job: TranscriptionJob, *, detail: str = "full") -> dict[str, object]:
        progress = job.progress.to_api()
        if detail == "compact":
            progress.pop("text", None)
            progress.pop("chunks", None)
            progress.pop("chunk_text", None)
        payload: dict[str, object] = {
            "id": job.id,
            "status": job.status,
            "model": job.model,
            "backend": job.backend,
            "queue_position": self._queue_position(job.id),
            "progress": progress,
            "created_at": utc_iso(job.created_at),
            "started_at": utc_iso(job.started_at),
            "completed_at": utc_iso(job.completed_at),
            "expires_at": utc_iso(job.expires_at),
            "elapsed_seconds": self._elapsed_seconds(job),
            "request": job.request_summary,
        }
        if job.split is not None:
            payload["split"] = job.split
        if job.status in {"completed", "failed", "cancelled", "expired"} and detail == "full":
            payload["result"] = job.result
            payload["error"] = job.error
        elif job.status == "failed":
            payload["error"] = job.error
        return payload

    def _queue_position(self, job_id: str) -> int:
        if job_id in self._queue:
            return list(self._queue).index(job_id) + 1
        return 0

    def _active_job_count(self) -> int:
        return len(self.scheduler.operations)

    def _elapsed_seconds(self, job: TranscriptionJob) -> float:
        end = job.completed_at or utc_now()
        return max((end - job.created_at).total_seconds(), 0.0)

    def _expire_jobs(self) -> None:
        now = utc_now()
        for job_id, job in list(self._jobs.items()):
            if job.expires_at is None:
                continue
            if job.expires_at <= now:
                job.result = None
                job.error = None
                if job.idempotency_key:
                    self._idempotency.pop(job.idempotency_key, None)
                del self._jobs[job_id]

    async def _cleanup_job_files(self, job: TranscriptionJob) -> None:
        try:
            if job.workspace is not None:
                await job.workspace.cleanup()
            else:
                for path in sorted(job.temp_paths, key=lambda item: len(item.parts), reverse=True):
                    try:
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink(missing_ok=True)
                    except OSError:
                        pass
        finally:
            if job.preparation_lease is not None:
                await job.preparation_lease.release()
                job.preparation_lease = None

    def _cleanup_stale_temp_dirs(self) -> None:
        from asr_server.audio.workspace import service_temporary_root
        temp_dir = service_temporary_root()
        for path in temp_dir.glob("asr_job_*"):
            try:
                if path.is_dir():
                    owner_path = path / "owner.pid"
                    if owner_path.exists():
                        try:
                            owner_pid = int(owner_path.read_text(encoding="ascii").strip())
                        except (OSError, ValueError):
                            owner_pid = -1
                        if _pid_is_running(owner_pid):
                            continue
                    shutil.rmtree(path)
            except OSError:
                pass

    async def _expiry_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(min(max(self._settings.job_result_ttl_seconds / 2, 1), 60))
                self._expire_jobs()
        except asyncio.CancelledError:
            return


def _phase_message(phase: str) -> str:
    messages = {
        "queued": "waiting for previous transcription jobs",
        "preprocessing": "preprocessing uploaded audio",
        "splitting": "splitting audio into chunks",
        "loading_model": "loading or confirming model",
        "transcribing": "running native long-form model invocation",
        "merging": "merging chunk transcripts",
    }
    return messages.get(phase, phase)


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _int_from_progress(value: int | None) -> int | None:
    return value


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
