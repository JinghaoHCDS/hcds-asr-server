from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, TypeVar

from fastapi import FastAPI

from asr_server.adapters.base import AudioPath, TranscriptionResult
from asr_server.audio.preprocess import normalize_audio_path_to_wav_async, probe_audio_duration_path_async
from asr_server.audio.retained import decoder_identity, prepare_retained_audio, validate_retained_audio
from asr_server.audio.video import probe_video, prepare_video, validate_video, select_video_encoder
from asr_server.audio.workspace import ResourceLease
from asr_server.quality.checkpoints import atomic_json, fingerprint
from asr_server.audio.assets import NormalizedAudioAsset
from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.observability import ExecutionContext, bind_context, current_context, emit, safe_exception
from asr_server.transcription import TranscriptionRequest, run_transcription_path, validate_transcription_request
from asr_server.workbench.speakers import ensure_speakers, pending_speaker, speaker_view
from asr_server.workbench.store import WorkbenchStore, after_days, identifier, now


CHUNK_SIZE = 8 * 1024 * 1024
TERMINAL = {"completed", "failed", "cancelled"}
T = TypeVar("T")


class WorkbenchInterrupted(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while data := source.read(CHUNK_SIZE):
            digest.update(data)
    return digest.hexdigest()


class WorkbenchService:
    def __init__(self, app: FastAPI, settings: Settings) -> None:
        self.app = app
        self.settings = settings
        self.store = WorkbenchStore(Path(settings.workbench_data_dir))
        self.lock = asyncio.Lock()
        self.video_lock = asyncio.Lock()
        self.video_queue_lock = asyncio.Lock()
        self.stopping = False
        self.draining = False
        self.storage_ready = True
        self.disk_ready = True
        self.store.storage_error_callback = self.storage_failure
        self.maintenance_at = time.monotonic()
        self.maintenance_failures = 0
        self.attempts: dict[str, dict[str, Any]] = {}
        self.maintenance: asyncio.Task[None] | None = None
        self.active: set[str] = set()
        self.process_lock: BinaryIO | None = None
        self.upload_locks: dict[str, asyncio.Lock] = {}
        self.contexts: dict[str, ExecutionContext] = {}
        self.cancel_intents: set[str] = set()
        self.progress: dict[str, dict[str, Any]] = {}
        self.progress_saved: dict[str, float] = {}
        self.video_tasks: dict[str, asyncio.Task[None]] = {}
        self.video_encoder: str | None = None
        self.reconciling = False
        self.preparation_errors: dict[str, BaseException] = {}
        self.prepared_assets: dict[str, NormalizedAudioAsset] = {}
        self.operation_leases: dict[str, ResourceLease] = {}
        self.playback_consumers: dict[str, int] = {}

    def storage_failure(self, _error: BaseException) -> None:
        self.storage_ready = False

    async def transaction(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        def execute() -> T:
            with self.store.connection() as db:
                return operation(db)
        try:
            return await self.store.run(execute)
        except (sqlite3.Error, OSError):
            self.storage_ready = False
            raise

    def readiness(self) -> dict[str, Any]:
        reasons = []
        if self.stopping or self.draining:
            reasons.append("draining")
        if not self.storage_ready:
            reasons.append("storage_unavailable")
        if not self.disk_ready:
            reasons.append("disk_space_low")
        if self.process_lock is None or self.maintenance is None or self.maintenance.done():
            reasons.append("maintenance_unavailable")
        elif time.monotonic() - self.maintenance_at > 180:
            reasons.append("maintenance_unresponsive")
        return {"ready": not reasons, "reasons": reasons}

    def set_draining(self, draining: bool) -> None:
        self.draining = draining
        if draining:
            for job_id in self.active | self.video_tasks.keys():
                context = self.contexts.get(job_id)
                if context is not None:
                    context.cancel_event.set()

    def assert_accepting(self, model_id: str | None = None) -> None:
        if self.stopping or self.draining:
            raise AsrError(503, "service_draining", "service is saving progress for maintenance")
        if not self.storage_ready or not self.disk_ready:
            raise AsrError(503, "storage_unavailable", "persistent storage is temporarily unavailable")
        if model_id is not None:
            self.app.state.manager.assert_inference_available(model_id)

    def attempt_arguments(self, job_id: str) -> dict[str, Any]:
        attempt = self.attempts.get(job_id)
        return {key: attempt[key] for key in ("attempt_id", "generation")} if attempt else {}

    async def sync_disk_reservations(self, projected: dict[str, int] | None = None) -> None:
        """Caller serializes DB changes with service.lock; settle even on cancel."""
        async def synchronize() -> None:
            declarations = projected if projected is not None else await self.store.run(self.store.future_disk_reservations)
            await self.app.state.workspace_manager.set_persistent_reservations(self.store.root, declarations,
                                                                             check_growth=projected is not None)
        task = asyncio.create_task(synchronize())
        interrupted = False
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                interrupted = True
        if interrupted:
            raise asyncio.CancelledError

    def upload_lock(self, upload_id: str) -> asyncio.Lock:
        return self.upload_locks.setdefault(upload_id, asyncio.Lock())

    def context_for(self, job: dict[str, Any]) -> ExecutionContext:
        if job["id"] not in self.contexts:
            self.contexts[job["id"]] = ExecutionContext.from_wire(job.get("execution_context")).with_fields(
                job_id=job["id"], model_id=job["request"]["model"], pipeline_id=job["request"].get("quality_profile", "legacy"))
        return self.contexts[job["id"]]

    def job_dir(self, job_id: str) -> Path:
        return self.store.root / "jobs" / job_id

    def upload_path(self, upload_id: str) -> Path:
        return self.store.root / "uploads" / upload_id

    async def start(self) -> None:
        if self.process_lock is not None:
            return
        await self.store.run(self.store.root.mkdir, parents=True, exist_ok=True)
        import fcntl
        process_lock = (self.store.root / "service.lock").open("a+b")
        try:
            fcntl.flock(process_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            process_lock.close()
            raise AsrError(409, "data_directory_in_use", "another workbench process owns this data directory") from exc
        self.process_lock = process_lock
        try:
            await self.store.run(self.store.initialize)
        except BaseException:
            fcntl.flock(process_lock.fileno(), fcntl.LOCK_UN)
            process_lock.close()
            self.process_lock = None
            raise
        self.stopping = False
        await self.recover_audio()
        await self.cleanup()
        async with self.lock:
            uploads = await self.transaction(lambda db: db.execute("SELECT * FROM uploads WHERE job_id IS NULL").fetchall())
            for upload in uploads:
                await self.store.run(self.reconcile_upload, dict(upload))
        for job_id in await self.store.run(self.store.pending_jobs):
            job = await self.store.run(self.store.job, job_id)
            if job.get("delete_requested"):
                await self.delete_now(job)
            elif job["status"] == "cancel_requested" or job.get("cancel_requested"):
                await self.finish_cancel(job)
            elif job["status"] not in TERMINAL:
                if job.get("next_attempt_at"):
                    continue
                job["status"] = "queued"
                job["progress"] = {"phase": "queued", "message": "recovering saved task"}
                (await self.store.run(self.store.save_job, job))
                try:
                    await self.enqueue(job["id"])
                except AsrError as exc:
                    if exc.code != "job_queue_full":
                        raise
            elif (job.get("result") or {}).get("alignment_status") == "pending":
                try:
                    await self.enqueue_alignment(job["id"])
                except AsrError as exc:
                    if exc.code != "job_queue_full":
                        raise
        self.app.state.job_manager.scheduler.refill = self.fill_queue
        await self.app.state.job_manager.scheduler.refill_available()
        self.maintenance = asyncio.create_task(self.maintenance_loop())

    async def shutdown(self) -> None:
        self.stopping = True
        self.app.state.job_manager.scheduler.refill = None
        for context in self.contexts.values():
            context.cancel_event.set()
        if self.maintenance is not None:
            self.maintenance.cancel()
            try:
                await self.maintenance
            except asyncio.CancelledError:
                pass
        if self.video_tasks:
            await asyncio.gather(*list(self.video_tasks.values()), return_exceptions=True)

    def close(self) -> None:
        """Release only after the shared queue/model manager has shut down."""
        if self.process_lock is not None:
            import fcntl
            fcntl.flock(self.process_lock.fileno(), fcntl.LOCK_UN)
            self.process_lock.close()
            self.process_lock = None
        self.store.close()

    async def maintenance_loop(self) -> None:
        next_cleanup = 0.0
        while True:
            try:
                if time.monotonic() >= next_cleanup:
                    # A real, small transaction verifies write access without
                    # scanning media during readiness requests.
                    await self.transaction(lambda db: db.execute("INSERT INTO metadata(key,payload) VALUES('write_probe','1') ON CONFLICT(key) DO UPDATE SET payload='1'"))
                    self.storage_ready = True
                    disk = await self.store.run(shutil.disk_usage, self.store.root)
                    self.disk_ready = disk.free >= self.settings.min_free_disk_mb * 1024 * 1024
                    await self.cleanup()
                    next_cleanup = time.monotonic() + 60
                await self.fill_queue()
                self.maintenance_failures = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.maintenance_failures += 1
                emit("maintenance.failed", diagnostics=safe_exception(exc), failures=self.maintenance_failures)
            finally:
                self.maintenance_at = time.monotonic()
            await asyncio.sleep(min(30, max(1, 2 ** min(self.maintenance_failures, 4))))

    async def fill_queue(self) -> None:
        if self.stopping or self.draining or not self.storage_ready:
            return
        await self.fill_video_queue()
        for job_id in await self.store.run(self.store.due_retry_jobs):
            if job_id not in self.active:
                await self.store.run(self.store.claim_due_retry, job_id)
        for job_id in await self.store.run(self.store.pending_jobs):
            try:
                job = await self.store.run(self.store.job, job_id)
            except AsrError as exc:
                if exc.code == "job_not_found":
                    continue
                raise
            if job["id"] in self.active:
                continue
            if job.get("delete_requested") or job.get("cancel_requested"):
                continue
            if job.get("next_attempt_at"):
                continue
            if not self.app.state.manager.inference_readiness(job["request"]["model"]).get("ready", True):
                continue
            pending_alignment = (job.get("result") or {}).get("alignment_status") == "pending"
            if job["status"] == "queued" or pending_alignment:
                try:
                    if pending_alignment:
                        await self.enqueue_alignment(job["id"])
                    else:
                        await self.enqueue(job["id"])
                except AsrError as exc:
                    if exc.code != "job_queue_full":
                        raise
                    break

    async def fill_video_queue(self) -> None:
        if self.stopping or self.draining or self.video_tasks:
            return
        async with self.video_queue_lock:
            if self.video_tasks:
                return
            rows = await self.transaction(lambda db: db.execute("SELECT id FROM jobs WHERE json_extract(payload,'$.video_asset.state')='queued' AND coalesce(json_extract(payload,'$.delete_requested'),0)=0 AND coalesce(json_extract(payload,'$.cancel_requested'),0)=0 ORDER BY sequence LIMIT 1").fetchall())
            if rows:
                key = str(rows[0]['id'])
                self.video_tasks[key] = asyncio.create_task(self.run_video_retry(key))

    async def queue_video(self, job_id: str) -> None:
        async with self.lock:
            job = await self.store.run(self.store.job_metadata, job_id)
            if (job.get('video_asset') or {}).get('state') == 'pending':
                job['video_asset'] = {'state': 'queued'}
                await self.store.run(self.store.save_job, job)
        await self.fill_video_queue()

    async def wait_for_video_capacity(self, job_id: str) -> None:
        """Bound the first model call's contention with this job's video encode.

        Audio preparation remains concurrent. A failed or slow video never
        becomes a prerequisite for transcription: the grace has a hard limit.
        """
        deadline = time.monotonic() + self.settings.video_inference_grace_seconds
        announced = False
        while job_id in self.video_tasks and time.monotonic() < deadline:
            self.check_running(job_id)
            job = await self.store.run(self.store.job_metadata, job_id)
            video = job.get('video_asset') or {}
            if (video.get('state') != 'preparing' or video.get('encoder') != 'h264_nvenc'
                    or (video.get('progress') or {}).get('phase') != 'encoding_video'):
                return
            if not announced:
                await self.audio_phase(job_id, 'waiting_for_video_capacity')
                announced = True
            await asyncio.sleep(min(.25, max(0., deadline-time.monotonic())))
        self.check_running(job_id)

    async def reserve_video_processing(self, job_id: str, remaining: int) -> None:
        async with self.lock:
            current = await self.store.run(self.store.job_metadata, job_id)
            physical = sum(current.get('retained_file_bytes', {}).values())
            reserved = await self.transaction(lambda db: int(db.execute('SELECT reserved_bytes FROM jobs WHERE id=?', (job_id,)).fetchone()[0]))
            audio = max(0, reserved - physical - int(current.get('video_reserved_bytes', 0)))
            desired = physical + audio + max(0, remaining)
            await self.store.run(self.check_capacity, current['owner_id'], max(0, desired-reserved))
            current['video_reserved_bytes'] = max(0, remaining)
            await self.store.run(self.store.save_job, current, reserved_bytes=desired)
            await self.sync_disk_reservations()

    def check_capacity(self, owner_id: str, additional: int) -> None:
        if self.reconciling:
            raise AsrError(503, "storage_unavailable", "storage reconciliation is in progress; try again shortly")
        if self.store.usage() + additional > self.settings.workbench_total_quota_bytes:
            raise AsrError(429, "storage_quota_exceeded", "workbench storage quota is exhausted")
        if self.store.usage(owner_id) + additional > self.settings.workbench_user_quota_bytes:
            raise AsrError(429, "storage_quota_exceeded", "your storage quota is exhausted")
        free = shutil.disk_usage(self.store.root).free
        promised = self.app.state.workspace_manager.pending_disk_bytes(self.store.root)
        if free - additional - promised < self.settings.min_free_disk_mb * 1024 * 1024:
            raise AsrError(503, "storage_unavailable", "not enough free disk space")

    async def reserve_processing(self, job: dict[str, Any], extra_bytes: int) -> None:
        """Reserve extraction/work copies before releasing the capacity lock."""
        async with self.lock:
            job = await self.store.run(self.store.job_metadata, job["id"])
            physical = sum(job.get("retained_file_bytes", {}).values())
            reserved = await self.transaction(lambda db: int(db.execute("SELECT reserved_bytes FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]))
            desired = physical + extra_bytes + int(job.get("video_reserved_bytes", 0))
            additional = max(0, desired - reserved)
            await self.store.run(self.check_capacity, job["owner_id"], additional)
            current = await self.store.run(self.store.job_metadata, job["id"])
            declarations = await self.store.run(self.store.future_disk_reservations)
            key = "job:" + job["id"]
            declarations[key] = declarations.get(key, 0) + additional
            try:
                await self.sync_disk_reservations(declarations)
                await self.store.run(self.store.save_job, current, reserved_bytes=max(reserved, desired))
            except BaseException:
                await self.sync_disk_reservations()
                raise

    async def settle_processing_reservation(self, job_id: str, margin: int) -> None:
        async with self.lock:
            current = await self.store.run(self.store.job_metadata, job_id)
            retained = sum(current.get("retained_file_bytes", {}).values())
            await self.store.run(self.store.save_job, current, reserved_bytes=retained + margin + int(current.get("video_reserved_bytes", 0)))
            await self.sync_disk_reservations()

    def processing_margin(self, request: TranscriptionRequest, duration: float) -> int:
        definition = self.app.state.manager.runtime_for(request.model).definition
        if request.quality_profile == "quality_v1":
            seconds = min(duration, 180.0)
        else:
            plan = definition.execution_policy.plan(requested_split_strategy=request.split_strategy,
                audio_duration_seconds=duration, max_chunk_seconds=request.max_chunk_seconds, overlap_seconds=request.overlap_seconds)
            seconds = min(duration, plan.max_chunk_seconds or plan.hard_chunk_seconds, plan.hard_chunk_seconds)
            if request.speaker_resolution != "off":
                seconds += 60.0
        return int(seconds * definition.audio_spec.bytes_per_second) + 4096 + 1024 * 1024

    async def create_upload(self, owner: str, filename: str, size: int) -> dict[str, Any]:
        if not filename or len(filename) > 255 or size <= 0:
            raise AsrError(400, "bad_request", "filename and a positive file size are required")
        if size > self.settings.workbench_max_upload_bytes:
            raise AsrError(413, "audio_too_large", "upload exceeds the workbench file size limit")
        async with self.lock:
            await self.store.run(self.store.initialize)
            await self.store.run(self.check_capacity, owner, size)
            upload_id = identifier("upload")
            path = self.upload_path(upload_id)
            def create() -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                try:
                    with self.store.connection() as db:
                        db.execute("INSERT INTO uploads VALUES (?,?,?,?,?,?,?,NULL)", (
                        upload_id, owner, Path(filename).name, size, 0, now(), after_days(1),
                        ))
                except BaseException:
                    path.unlink(missing_ok=True)
                    raise
            declarations = await self.store.run(self.store.future_disk_reservations)
            declarations["upload:" + upload_id] = size
            try:
                await self.sync_disk_reservations(declarations)
                await self.store.run(create)
            except BaseException:
                await self.sync_disk_reservations()
                raise
            return await self.store.run(self.upload, owner, upload_id)

    def upload(self, owner: str, upload_id: str) -> dict[str, Any]:
        with self.store.connection() as db:
            row = db.execute("SELECT * FROM uploads WHERE id=? AND owner_id=?", (upload_id, owner)).fetchone()
        if row is None:
            raise AsrError(404, "upload_not_found", "upload not found")
        upload = dict(row)
        if upload["job_id"] is None:
            path = self.upload_path(upload_id)
            if not path.exists() or path.stat().st_size < upload["offset"]:
                raise AsrError(409, "upload_corrupt", "saved upload is incomplete; start a new upload")
        return {**upload, "chunk_size": CHUNK_SIZE}

    def reconcile_upload(self, upload: dict[str, Any]) -> None:
        """Called under the upload lock, never concurrently with a write."""
        path = self.upload_path(upload["id"])
        if not path.exists() or path.stat().st_size < upload["offset"]:
            return  # API exposes upload_corrupt; do not prevent other recovery.
        if path.stat().st_size > upload["offset"]:
            with path.open("r+b") as target:
                target.truncate(upload["offset"])

    async def append_upload(self, owner: str, upload_id: str, offset: int, data: bytes) -> dict[str, Any]:
        if not data or len(data) > CHUNK_SIZE or offset < 0:
            raise AsrError(400, "bad_request", "invalid upload chunk")
        await self.store.run(self.upload, owner, upload_id)
        async with self.upload_lock(upload_id):
            try:
                result = await self.store.run(self._append_upload, owner, upload_id, offset, data)
            finally:
                async with self.lock:
                    await self.sync_disk_reservations()
            return result

    def _append_upload(self, owner: str, upload_id: str, offset: int, data: bytes) -> dict[str, Any]:
        import os
        upload = self.upload(owner, upload_id)
        if upload["job_id"] is not None:
            raise AsrError(409, "upload_completed", "upload already submitted")
        if offset + len(data) > upload["size_bytes"]:
            raise AsrError(400, "bad_request", "upload chunk exceeds declared size")
        path = self.upload_path(upload_id)
        self.reconcile_upload(upload)
        if offset < upload["offset"]:
            with path.open("rb") as source:
                source.seek(offset)
                same = source.read(len(data)) == data
            if same and offset + len(data) <= upload["offset"]:
                return upload
            raise AsrError(409, "upload_offset_mismatch", "replayed chunk differs from saved content", {"offset": upload["offset"]})
        if offset != upload["offset"]:
            raise AsrError(409, "upload_offset_mismatch", "resume at the server offset", {"offset": upload["offset"]})
        if shutil.disk_usage(self.store.root).free - len(data) < self.settings.min_free_disk_mb * 1024 * 1024:
            raise AsrError(503, "storage_unavailable", "not enough free disk space")
        with path.open("r+b") as target:
            target.seek(offset)
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        with self.store.connection() as db:
            db.execute("UPDATE uploads SET offset=?,expires_at=? WHERE id=?", (offset + len(data), after_days(1), upload_id))
        self.store.set_physical_usage(upload_id, owner, offset + len(data))
        return self.upload(owner, upload_id)

    def snapshot(self, domain: str, hotwords: str) -> dict[str, Any]:
        if domain not in {"general", "ai"}:
            raise AsrError(400, "bad_request", "unknown glossary domain")
        glossary = self.store.glossary()
        from asr_server.context import parse_hotwords
        terms = parse_hotwords(hotwords)
        public_entries = [entry for entry in glossary["entries"] if entry["domain"] == domain]
        if domain != "general":
            public_entries.extend(entry for entry in glossary["entries"] if entry["domain"] == "general")
        public_by_name: dict[str, dict[str, Any]] = {}
        for entry in public_entries:
            public_by_name.setdefault(entry["canonical"].casefold(), entry)
        # A per-request mention changes priority, not an administrator's
        # approved spelling/alias policy. Domain entries win over general ones.
        entries: list[dict[str, Any]] = [copy.deepcopy(public_by_name.get(term.casefold(), {
            "canonical": term, "aliases": [], "domain": "request", "auto_normalize": False,
        })) for term in terms]
        entries.extend(public_entries)
        seen: set[str] = set()
        selected = []
        for entry in entries:
            key = entry["canonical"].casefold()
            if key not in seen:
                selected.append(entry)
                seen.add(key)
        return {"version": glossary["version"], "domain": domain, "entries": selected}

    def default_selection(self) -> tuple[str, str]:
        manager = self.app.state.manager
        definitions = [manager.runtime_for(str(item["id"])).definition for item in manager.list_models()]
        candidates = [item for item in definitions if item.supports_pipeline("quality_v1")]
        if candidates:
            definition = next((item for item in candidates if item.default), candidates[0])
            return definition.id, "quality_v1"
        return manager.default_model_id, "legacy"

    def submission_request(self, parameters: dict[str, Any]) -> TranscriptionRequest:
        model, profile = self.default_selection()
        selected = str(parameters.get("model") or model)
        profile = str(parameters.get("quality_profile") or profile)
        definition = self.app.state.manager.runtime_for(selected).definition
        request = TranscriptionRequest(
            model=selected, language=str(parameters.get("language", "auto")), response_format="verbose_json",
            timestamps="auto" if definition.capabilities.forced_alignment else "none",
            backend="auto", max_new_tokens=65536 if definition.capabilities.diarization else None, context="", hotwords=str(parameters.get("hotwords", "")) or None,
            split_strategy="auto", max_chunk_seconds=None, overlap_seconds=None, preserve_segments=True,
            speaker_resolution=str(parameters.get("speaker_resolution") or ("auto" if definition.capabilities.diarization else "off")),
            quality_profile=profile, glossary_snapshot=self.snapshot(str(parameters.get("domain", "general")), str(parameters.get("hotwords", ""))),
        )
        expected = parameters.get("expected_speakers")
        if expected is not None and (not definition.capabilities.diarization or isinstance(expected, bool) or not isinstance(expected, int) or not 1 <= expected <= 4):
            raise AsrError(400, "bad_request", "expected_speakers requires diarization and a count from 1 to 4")
        validate_transcription_request(self.app.state.manager, request)
        return request

    def execution_identity(self, request: TranscriptionRequest) -> dict[str, Any]:
        identity = dict(self.app.state.manager.runtime_for(request.model).definition.identity())
        if request.quality_profile == "quality_v1":
            from asr_server.quality import PROFILE_VERSION
            from asr_server.quality.speech_evidence import detector_identity
            from asr_server.registry import ALIGNER_SOURCE
            identity.update({"quality_profile_version": PROFILE_VERSION, "aligner_revision": ALIGNER_SOURCE[1],
                             "speech_detector": detector_identity()})
        return identity

    async def complete_upload(self, owner: str, upload_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        self.assert_accepting()
        await self.store.run(self.upload, owner, upload_id)
        async with self.upload_lock(upload_id):
            upload = await self.store.run(self.upload, owner, upload_id)
            if upload["job_id"]:
                existing = await self.store.run(self.store.job, upload["job_id"], owner)
                original = existing["request"]
                options = {"domain": original["glossary_snapshot"]["domain"], "hotwords": original["hotwords"] or "",
                           "language": original["language"], "model": original["model"], "quality_profile": original.get("quality_profile", "legacy")}
                requested = {"domain": str(parameters.get("domain", "general")), "hotwords": str(parameters.get("hotwords", "")),
                             "language": str(parameters.get("language", "auto")), "model": parameters.get("model") or original["model"],
                             "quality_profile": parameters.get("quality_profile") or original.get("quality_profile", "legacy")}
                options.update(speaker_resolution=original.get("speaker_resolution", "off"), expected_speakers=existing.get("expected_speakers"))
                requested.update(speaker_resolution=parameters.get("speaker_resolution") or original.get("speaker_resolution", "off"), expected_speakers=parameters.get("expected_speakers"))
                saved_key = await self.transaction(lambda db: db.execute("SELECT idempotency_key FROM jobs WHERE id=?", (existing["id"],)).fetchone()[0])
                if requested != options or (parameters.get("idempotency_key") or upload_id) != saved_key:
                    raise AsrError(409, "idempotency_conflict", "upload was submitted with different options or idempotency key")
                return self.public_job(existing)
            if upload["offset"] != upload["size_bytes"]:
                raise AsrError(409, "upload_incomplete", "upload has not finished", {"offset": upload["offset"]})
            request = await self.store.run(self.submission_request, parameters)
            definition = self.app.state.manager.runtime_for(request.model).definition
            identity = await self.store.run(self.execution_identity, request)
            # Only this upload is locked while reading its complete source.
            digest = await self.store.run(sha256_file, self.upload_path(upload_id))
            # Classify streams before queueing so pure audio never acquires a
            # video owner. Video-specific failures are handled independently
            # after submission and must not prevent audio transcription.
            video_state = "queued"
            try:
                if await probe_video(self.upload_path(upload_id)) is None:
                    video_state = "not_applicable"
            except AsrError:
                pass
            key = parameters.get("idempotency_key") or upload_id
            if not isinstance(key, str) or len(key) > 200:
                raise AsrError(400, "bad_request", "invalid idempotency key")
            async with self.lock:
                self.assert_accepting(request.model)
                # Revalidate all mutable inputs after the full-file hash.
                latest = await self.store.run(self.upload, owner, upload_id)
                if latest["offset"] != upload["offset"] or latest["job_id"] is not None:
                    raise AsrError(409, "idempotency_conflict", "upload changed during submission")
                current_request = await self.store.run(self.submission_request, parameters)
                if await self.store.run(self.execution_identity, current_request) != identity or asdict(current_request) != asdict(request):
                    raise AsrError(409, "version_conflict", "model or glossary changed during submission; retry")
                fingerprint = hashlib.sha256(json.dumps({"source_sha256": digest, "model_identity": identity,
                    "request": asdict(request), "expected_speakers": parameters.get("expected_speakers")}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                existing = await self.transaction(lambda db: db.execute("SELECT id,fingerprint FROM jobs WHERE owner_id=? AND idempotency_key=?", (owner, key)).fetchone())
                if existing:
                    if existing["fingerprint"] != fingerprint:
                        raise AsrError(409, "idempotency_conflict", "idempotency key was used for different content or options")
                    def reuse() -> None:
                        with self.store.connection() as db:
                            db.execute("UPDATE uploads SET job_id=? WHERE id=?", (existing["id"], upload_id))
                        self.upload_path(upload_id).unlink(missing_ok=True)
                        self.store.set_physical_usage(upload_id, owner, 0)
                    await self.store.run(reuse)
                    await self.sync_disk_reservations()
                    return self.public_job(await self.store.run(self.store.job, existing["id"], owner))
                job_id = identifier("wb")
                root = self.job_dir(job_id)
                context = ExecutionContext.from_wire(current_context().to_wire()).with_fields(job_id=job_id,
                    model_id=request.model, engine_id=definition.engine_id, pipeline_id=request.quality_profile)
                job: dict[str, Any] = {
                    "id": job_id, "owner_id": owner, "filename": upload["filename"], "status": "queued",
                    "created_at": now(), "started_at": None, "completed_at": None, "expires_at": None,
                    "progress": {"phase": "queued", "percent": 0}, "error": None, "result": None,
                    "request": asdict(request), "model_revision": definition.revision, "model_identity": identity,
                    "source_sha256": digest, "delete_requested": False, "cancel_requested": False,
                    "execution_context": context.to_wire(), "retained_file_bytes": {"source": upload["size_bytes"]},
                    "audio_asset": {"state": "preparing"}, "video_asset": {"state": video_state},
                    "expected_speakers": parameters.get("expected_speakers"),
                }
                def commit() -> None:
                    root.mkdir(parents=True)
                    (root / "source").hardlink_to(self.upload_path(upload_id))
                    try:
                        with self.store.connection() as db:
                            db.execute("INSERT INTO jobs (id,owner_id,status,reserved_bytes,payload,idempotency_key,fingerprint) VALUES (?,?,?,?,?,?,?)", (
                                job_id, owner, "queued", upload["size_bytes"], json.dumps(job, ensure_ascii=False), key, fingerprint))
                            db.execute("UPDATE uploads SET job_id=? WHERE id=?", (job_id, upload_id))
                    except BaseException:
                        shutil.rmtree(root, ignore_errors=True)
                        raise
                    self.upload_path(upload_id).unlink(missing_ok=True)
                    self.store.set_physical_usage(upload_id, owner, 0)
                    self.store.set_physical_usage(job_id, owner, upload["size_bytes"])
                await self.store.run(commit)
                await self.sync_disk_reservations()
                self.contexts[job_id] = context
        await self.fill_queue()
        return self.public_job(await self.store.run(self.store.job, job_id, owner))

    def public_job(self, job: dict[str, Any], *, include_result: bool = True) -> dict[str, Any]:
        fields = {"id", "filename", "status", "revision", "created_at", "started_at", "completed_at", "expires_at",
                  "progress", "error", "result", "model_revision", "audio_asset", "audio_size_bytes", "video_asset", "expected_speakers",
                  "retryability", "next_action", "next_attempt_at", "completeness", "gaps", "recovery"}
        payload = {key: value for key, value in job.items() if key in fields}
        if include_result and payload.get("result"):
            payload["result"] = speaker_view(payload["result"])
        if not include_result:
            payload.pop("result", None)
        payload["audio_size_bytes"] = int((job.get("audio_asset") or {}).get("size_bytes", 0))
        payload["model"] = job.get("request", {}).get("model")
        payload["queue_position"] = self.app.state.job_manager.external_queue_position(job["id"])
        return payload

    async def enqueue(self, job_id: str) -> None:
        job = await self.store.run(self.store.job, job_id)
        context = self.context_for(job)
        if context.cancel_event.is_set() and not self.cancelled(job_id):
            context = context.with_fields(cancel_event=threading.Event())
            self.contexts[job_id] = context
        await self.app.state.job_manager.enqueue_external(job_id, lambda: self.run_job(job_id), context=self.context_for(job),
            prepare=lambda: self.prepare_for_scheduler(job_id), dispose=lambda: self.dispose_preparation(job_id))

    async def prepare_for_scheduler(self, job_id: str) -> None:
        added = job_id not in self.active
        if added:
            self.active.add(job_id)
        try:
            self.prepared_assets[job_id] = await self.prepare_audio(job_id)
        except AsrError as exc:
            # A speculative successor cannot take resources needed by the
            # running owner. Retry admission when this operation gets its turn.
            if exc.code not in {"storage_unavailable", "storage_quota_exceeded", "job_queue_full", "workbench_busy"}:
                self.preparation_errors[job_id] = exc
        except asyncio.CancelledError as exc:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise
            self.preparation_errors[job_id] = exc
        except Exception as exc:
            self.preparation_errors[job_id] = exc
        finally:
            if added:
                self.active.discard(job_id)

    async def dispose_preparation(self, job_id: str) -> None:
        """May run while cancel/delete owns service.lock; never acquire it here."""
        self.preparation_errors.pop(job_id, None)
        self.prepared_assets.pop(job_id, None)
        await self.cleanup_operation(job_id, locked=False)

    def cancelled(self, job_id: str) -> bool:
        return job_id in self.cancel_intents

    def check_running(self, job_id: str) -> None:
        context = self.contexts.get(job_id)
        if self.stopping or self.draining or self.cancelled(job_id) or (context is not None and context.cancel_event.is_set()):
            raise WorkbenchInterrupted()

    def is_stopping(self) -> bool:
        # Re-evaluate after awaited cleanup; shutdown may have started meanwhile.
        return self.stopping or self.draining

    def terminal(self, job: dict[str, Any], status: str) -> None:
        job["status"] = status
        job["completed_at"] = now()
        job["expires_at"] = after_days(self.settings.workbench_retention_days)

    async def run_job(self, job_id: str) -> None:
        try:
            job = await self.store.run(self.store.job, job_id)
        except AsrError as exc:
            if exc.code == "job_not_found":
                return
            raise
        with bind_context(self.context_for(job)):
            await self._run_job(job_id)

    async def record_files(self, job_id: str, names: list[str]) -> None:
        async with self.lock:
            await self.store.run(self.record_files_sync, job_id, names)
            await self.sync_disk_reservations()

    def record_files_sync(self, job_id: str, names: list[str]) -> None:
        current = self.store.job_metadata(job_id)
        sizes = current.setdefault("retained_file_bytes", {})
        previous = dict(sizes)
        for name in names:
            path = self.job_dir(job_id) / name
            try:
                sizes[name] = path.stat().st_size
            except FileNotFoundError:
                sizes.pop(name, None)
        if sizes == previous:
            return
        temporary, text = self.store.file_categories(sizes)
        self.store.set_physical_usage(job_id, current["owner_id"], sum(sizes.values()), temporary_bytes=temporary, text_file_bytes=text)
        self.store.save_job(current)

    async def audio_phase(self, job_id: str, phase: str) -> None:
        async with self.lock:
            self.check_running(job_id)
            job = await self.store.run(self.store.job_metadata, job_id)
            job["progress"] = {**job.get("progress", {}), "phase": phase}
            if job["status"] not in TERMINAL:
                job["status"] = "preprocessing"
            await self.store.run(self.store.save_job, job)

    async def recover_audio(self) -> None:
        """No live consumers exist at startup; discard abandoned operation files."""
        rows = await self.transaction(lambda db: db.execute("SELECT id FROM jobs").fetchall())
        for row in rows:
            job_id = str(row["id"])
            candidate = await self.store.run(self.store.job_metadata, job_id)
            if candidate.get("delete_requested"):
                await self.delete_now(candidate)
                continue
            await self.cleanup_operation(job_id)
            job = await self.store.run(self.store.job_metadata, job_id)
            video = job.get("video_asset") or {"state": "unavailable"}
            if video.get("state") == "ready":
                try:
                    await self.store.run(validate_video, self.job_dir(job_id) / "video.mp4", video)
                except (AsrError, OSError):
                    video = {"state": "failed", "error_code": "retained_video_invalid"}
            elif video.get("state") == "preparing":
                video = {"state": "queued"}
            job["video_asset"] = video
            job["video_reserved_bytes"] = 0
            metadata = job.get("audio_asset") or {"state": "preparing"}
            if metadata.get("state") != "ready" and (self.job_dir(job_id) / "source").exists():
                await self.store.run((self.job_dir(job_id) / "audio.mp3").unlink, missing_ok=True)
            if metadata.get("state") == "ready":
                try:
                    await self.store.run(validate_retained_audio, self.job_dir(job_id) / "audio.mp3", metadata)
                    if video.get("state") in {"ready", "not_applicable", "unavailable"}:
                        await self.store.run((self.job_dir(job_id) / "source").unlink, missing_ok=True)
                except (AsrError, OSError):
                    metadata = {**metadata, "state": "failed", "error_code": "retained_audio_invalid"}
                    job["error"] = {"code": "retained_audio_invalid", "message": "saved playback audio failed identity validation", "details": {}}
                    self.terminal(job, "cancelled" if job.get("cancel_requested") else "failed")
            job["audio_asset"] = metadata
            await self.store.run(self.store.save_job, job)
            await self.record_files(job_id, ["source", "audio.mp3", "video.mp4"])

    async def cleanup_operation(self, job_id: str, *, locked: bool = True) -> None:
        root = self.job_dir(job_id)
        def remove() -> None:
            for name in ("work", "operation"):
                path = root / name
                if path.exists():
                    shutil.rmtree(path)
            for path in root.glob("*.partial*"):
                if job_id not in self.video_tasks or not path.name.startswith("video"):
                    path.unlink(missing_ok=True)
        try:
            await asyncio.to_thread(remove)
        except OSError as exc:
            async def pending() -> None:
                current = await self.store.run(self.store.job_metadata, job_id)
                current["status"] = "cleanup_pending"
                current["progress"] = {"phase": "cleanup_pending"}
                current["error"] = {"code": "cleanup_failed", "message": "temporary audio cleanup will be retried", "details": {}}
                await self.store.run(self.store.save_job, current)
            if locked:
                async with self.lock:
                    await pending()
            else:
                await pending()
            raise AsrError(503, "cleanup_failed", "temporary audio could not be removed; cleanup will be retried") from exc
        lease = self.operation_leases.pop(job_id, None)
        if lease is not None:
            await lease.release()
        async def settle() -> None:
            current = await self.store.run(self.store.job_metadata, job_id)
            names = [name for name in current.get("retained_file_bytes", {})
                     if job_id not in self.video_tasks or not name.startswith("video")]
            # Merge reviews and unsuccessful final calls can write durable
            # artifacts after the last successful chunk callback.
            checkpoint_names = await self.store.run(lambda: [str(path.relative_to(root))
                for path in (root / "checkpoints").rglob("*.json")])
            names = sorted(set(names).union(checkpoint_names))
            await self.store.run(self.record_files_sync, job_id, names)
            current = await self.store.run(self.store.job_metadata, job_id)
            retained = sum(current.get("retained_file_bytes", {}).values())
            reserved = await self.transaction(lambda db: db.execute("SELECT reserved_bytes FROM jobs WHERE id=?", (job_id,)).fetchone()[0])
            desired = retained + int(current.get("video_reserved_bytes", 0))
            if reserved != desired:
                await self.store.run(self.store.save_job, current, reserved_bytes=desired)
            await self.sync_disk_reservations()
        if locked:
            async with self.lock:
                await settle()
        else:
            await settle()

    async def discard_ready_source(self, job_id: str) -> None:
        async with self.lock:
            job = await self.store.run(self.store.job_metadata, job_id)
            if ((job.get("audio_asset") or {}).get("state") == "ready"
                    and (job.get("video_asset") or {}).get("state", "not_applicable") in {"ready", "not_applicable"}):
                source_bytes = int(job.get('retained_file_bytes', {}).get('source', 0))
                reserved = await self.transaction(lambda db: int(db.execute('SELECT reserved_bytes FROM jobs WHERE id=?', (job_id,)).fetchone()[0]))
                await self.store.run((self.job_dir(job_id) / "source").unlink, missing_ok=True)
                await self.store.run(self.record_files_sync, job_id, ["source", "audio.mp3", "video.mp4"])
                current = await self.store.run(self.store.job_metadata, job_id)
                await self.store.run(self.store.save_job, current, reserved_bytes=max(sum(current.get('retained_file_bytes', {}).values()), reserved-source_bytes))
                await self.sync_disk_reservations()

    async def prepare_video_asset(self, job_id: str) -> None:
        async with self.video_lock:
            self.check_running(job_id)
            job = await self.store.run(self.store.job_metadata, job_id)
            state = (job.get("video_asset") or {}).get("state", "unavailable")
            if state in {"ready", "not_applicable", "unavailable", "failed"}:
                return
            root = self.job_dir(job_id)
            context = self.context_for(job)
            lease: ResourceLease | None = None
            written_bytes = 0
            try:
                info = await probe_video(root / "source", context)
                if info is None:
                    metadata: dict[str, Any] = {"state": "not_applicable"}
                else:
                    if self.video_encoder is None:
                        self.video_encoder = await select_video_encoder(self.settings.video_encoder, context)
                    async with self.lock:
                        current = await self.store.run(self.store.job_metadata, job_id)
                        current["video_asset"] = {"state": "preparing", "encoder": self.video_encoder}
                        await self.store.run(self.store.save_job, current)
                    async def video_progress(value: dict[str, Any]) -> None:
                        self.check_running(job_id)
                        async with self.lock:
                            current = await self.store.run(self.store.job_metadata, job_id)
                            if current.get("delete_requested"):
                                raise WorkbenchInterrupted
                            current["video_asset"]["progress"] = value
                            await self.store.run(self.store.save_job, current)
                    (root / "video.mp4").unlink(missing_ok=True)
                    await self.reserve_video_processing(job_id, 8 * 1024 * 1024)
                    lease = await self.app.state.workspace_manager.handoff_persistent(root, "video:" + job_id, 8 * 1024 * 1024)
                    async def ensure(required: int) -> None:
                        assert lease is not None
                        if required > lease.size_bytes:
                            desired = max(required, lease.size_bytes + 8 * 1024 * 1024)
                            await self.reserve_video_processing(job_id, desired - written_bytes)
                            await lease.grow_persistent(desired - lease.size_bytes)
                    async def wrote(total: int) -> None:
                        nonlocal written_bytes
                        assert lease is not None
                        delta = total - written_bytes
                        await lease.materialize(delta)
                        written_bytes = total
                        async with self.lock:
                            current = await self.store.run(self.store.job_metadata, job_id)
                            current['video_reserved_bytes'] = max(0, int(current.get('video_reserved_bytes', 0)) - delta)
                            await self.store.run(self.store.save_job, current)
                            names = await self.store.run(lambda: [path.name for path in root.glob('video*.partial*')])
                            await self.store.run(self.record_files_sync, job_id, names)
                            await self.sync_disk_reservations()
                    metadata = await prepare_video(root / "source", root / "video.mp4", info,
                        context=context, ensure_capacity=ensure, written=wrote, encoder=self.video_encoder,
                        progress=video_progress, max_fps=self.settings.video_max_fps)
                async with self.lock:
                    self.check_running(job_id)
                    current = await self.store.run(self.store.job_metadata, job_id)
                    if current.get("delete_requested"):
                        raise WorkbenchInterrupted
                    current["video_asset"] = metadata
                    await self.store.run(self.store.save_job, current)
            except (asyncio.CancelledError, WorkbenchInterrupted):
                async with self.lock:
                    current = await self.store.run(self.store.job_metadata, job_id)
                    current["video_asset"] = {"state": "queued" if self.is_stopping() else "failed", "error_code": "video_interrupted"}
                    await self.store.run(self.store.save_job, current)
                raise
            except Exception as exc:
                async with self.lock:
                    current = await self.store.run(self.store.job_metadata, job_id)
                    current["video_asset"] = {"state": "failed", "error_code": exc.code if isinstance(exc, AsrError) else "video_preparation_failed"}
                    await self.store.run(self.store.save_job, current)
            finally:
                if lease is not None:
                    await lease.release()
                async with self.lock:
                    current = await self.store.run(self.store.job_metadata, job_id)
                    physical_before = sum(current.get('retained_file_bytes', {}).values())
                    reserved_before = await self.transaction(lambda db: int(db.execute('SELECT reserved_bytes FROM jobs WHERE id=?', (job_id,)).fetchone()[0]))
                    audio_remaining = max(0, reserved_before - physical_before - int(current.get('video_reserved_bytes', 0)))
                    if (current.get("video_asset") or {}).get("state") != "ready":
                        await self.store.run((root / "video.mp4").unlink, missing_ok=True)
                    names = sorted(set(["video.mp4", "video.partial.mp4", "video.faststart.partial.mp4", *[name for name in current.get('retained_file_bytes', {}) if name.startswith('video')]]))
                    await self.store.run(self.record_files_sync, job_id, names)
                    current = await self.store.run(self.store.job_metadata, job_id)
                    current['video_reserved_bytes'] = 0
                    await self.store.run(self.store.save_job, current,
                        reserved_bytes=sum(current.get('retained_file_bytes', {}).values()) + audio_remaining)
                    await self.sync_disk_reservations()
            await self.discard_ready_source(job_id)

    async def retry_video(self, owner: str, job_id: str) -> dict[str, Any]:
        self.assert_accepting()
        async with self.lock:
            job = await self.store.run(self.store.job_metadata, job_id, owner)
            if job.get("delete_requested") or job_id in self.active or job_id in self.video_tasks or job["status"] not in {"completed", "failed"}:
                raise AsrError(409, "job_busy", "wait for the active operation before retrying video")
            if (job.get("video_asset") or {}).get("state") not in {"failed", "queued"}:
                raise AsrError(409, "video_not_retryable", "video does not need retrying")
            if not (self.job_dir(job_id) / "source").exists():
                raise AsrError(409, "source_unavailable", "original video is unavailable")
            self.cancel_intents.discard(job_id)
            self.contexts.pop(job_id, None)
            job["video_asset"] = {"state": "queued"}
            await self.store.run(self.store.save_job, job)
        await self.fill_queue()
        return self.public_job(await self.store.run(self.store.job, job_id, owner))

    async def run_video_retry(self, job_id: str) -> None:
        try:
            job = await self.store.run(self.store.job_metadata, job_id)
            if job.get("delete_requested") or job.get("cancel_requested"):
                return
            with bind_context(self.context_for(job)):
                await self.prepare_video_asset(job_id)
        except (asyncio.CancelledError, WorkbenchInterrupted):
            pass
        except Exception as exc:
            emit("video.preparation_failed", error_code=getattr(exc, "code", "video_preparation_failed"))
        finally:
            self.video_tasks.pop(job_id, None)
            async with self.lock:
                try:
                    job = await self.store.run(self.store.job_metadata, job_id)
                except AsrError as exc:
                    if exc.code != "job_not_found":
                        raise
                    return
                if job.get("delete_requested") and job_id not in self.active and not self.playback_consumers.get(job_id):
                    await self.delete_now(job)
            await self.fill_video_queue()

    async def prepare_audio(self, job_id: str) -> NormalizedAudioAsset:
        self.check_running(job_id)
        job = await self.store.run(self.store.job_metadata, job_id)
        request = TranscriptionRequest(**job["request"])
        definition = self.app.state.manager.runtime_for(request.model or self.app.state.manager.default_model_id).definition
        if job.get("model_identity") is not None and await self.store.run(self.execution_identity, request) != job["model_identity"]:
            raise AsrError(409, "resume_version_unavailable", "saved execution identity is unavailable; start from the beginning")
        root, context = self.job_dir(job_id), self.context_for(job)
        source, playback = root / "source", root / "audio.mp3"
        metadata = job.get("audio_asset") or {"state": "preparing"}
        if metadata.get("state") != "ready":
            if source.exists():
                await self.store.run(playback.unlink, missing_ok=True)
            async with self.lock:
                current = await self.store.run(self.store.job_metadata, job_id)
                current["audio_asset"] = {"state": "preparing"}
                await self.store.run(self.store.save_job, current)
            await self.audio_phase(job_id, "preparing_playback_audio")
            estimate = await probe_audio_duration_path_async(source, timeout_seconds=self.settings.ffprobe_timeout_seconds, cancel_event=context.cancel_event)
            mp3_limit = 21600 * 24000 + 1024 * 1024
            mp3_budget = min(mp3_limit, int(min(estimate or 60, 21600) * 24000) + 1024 * 1024)
            await self.reserve_processing(job, mp3_budget)
            lease = await self.app.state.workspace_manager.handoff_persistent(root, "job:" + job_id, mp3_budget)
            written = 0
            async def grow_mp3(required: int) -> None:
                if required > lease.size_bytes:
                    desired = min(mp3_limit, max(required, lease.size_bytes + 1024 * 1024))
                    await self.reserve_processing(await self.store.run(self.store.job_metadata, job_id), desired - written)
                    await lease.grow_persistent(desired - lease.size_bytes)
            async def wrote_mp3(total: int) -> None:
                nonlocal written
                await lease.materialize(total - written)
                written = total
                await self.record_files(job_id, ["audio.partial.mp3"])
            try:
                metadata = await prepare_retained_audio(source, playback, max_duration_seconds=21600,
                    max_output_bytes=mp3_limit, min_free_disk_bytes=self.settings.min_free_disk_mb * 1024 * 1024,
                    context=context, ensure_output_capacity=grow_mp3, output_written=wrote_mp3)
                # Commit ready metadata before removing the sole original upload.
                async with self.lock:
                    current = await self.store.run(self.store.job_metadata, job_id)
                    current["audio_asset"] = metadata
                    await self.store.run(self.store.save_job, current)
                await self.record_files(job_id, ["source", "audio.mp3", "audio.partial.mp3"])
                await self.discard_ready_source(job_id)
            except BaseException:
                async with self.lock:
                    current = await self.store.run(self.store.job_metadata, job_id)
                    if (current.get("audio_asset") or {}).get("state") != "ready":
                        current["audio_asset"] = {"state": "failed"}
                        await self.store.run(self.store.save_job, current)
                raise
            finally:
                await lease.release()
        else:
            await self.store.run(validate_retained_audio, playback, metadata)
        await self.queue_video(job_id)
        self.check_running(job_id)
        await self.audio_phase(job_id, "preparing_recognition_audio")
        operation = root / "operation"
        operation.mkdir(exist_ok=True)
        audio = operation / "audio.wav"
        pcm_limit = 21600 * definition.audio_spec.bytes_per_second + 44
        pcm_budget = min(pcm_limit, int(float(metadata["duration_seconds"]) * definition.audio_spec.bytes_per_second) + 4096)
        margin = self.processing_margin(request, float(metadata["duration_seconds"]))
        await self.reserve_processing(await self.store.run(self.store.job_metadata, job_id), pcm_budget + margin)
        pcm_lease = await self.app.state.workspace_manager.handoff_persistent(root, "job:" + job_id, pcm_budget)
        self.operation_leases[job_id] = pcm_lease
        last_recorded = 0
        async def grow_pcm(required: int) -> None:
            nonlocal last_recorded
            if required - last_recorded >= 1024 * 1024 or required > pcm_lease.size_bytes:
                await self.record_files(job_id, ["operation/audio.wav"])
                last_recorded = required
            if required > pcm_lease.size_bytes:
                desired = min(pcm_limit, max(required, pcm_lease.size_bytes + 1024 * 1024))
                current = await self.store.run(self.store.job_metadata, job_id)
                materialized = current.get("retained_file_bytes", {}).get("operation/audio.wav", 0)
                await self.reserve_processing(current, desired - materialized + margin)
                await pcm_lease.grow_persistent(desired - pcm_lease.size_bytes)
        normalized = await normalize_audio_path_to_wav_async(playback, audio,
            timeout_seconds=self.settings.ffmpeg_timeout_seconds, max_output_bytes=pcm_limit,
            max_duration_seconds=21600, spec=definition.audio_spec, cancel_event=context.cancel_event,
            owner=job_id, resource_lease=pcm_lease, ensure_output_capacity=grow_pcm)
        await pcm_lease.release_unused()
        asset = normalized.asset
        if asset is None:
            asset = await self.store.run(NormalizedAudioAsset.from_path, audio, spec=definition.audio_spec, owner=job_id)
        stable_identity = {"retained_sha256": metadata["sha256"], "retained_version": metadata["version"],
                           "decoder": await decoder_identity(context=context), "pcm": asset.to_metadata()}
        identity_path = root / "checkpoints" / "audio-identity.json"
        if identity_path.exists():
            previous = await self.store.run(lambda: json.loads(identity_path.read_text()))
            if fingerprint(previous) != fingerprint(stable_identity):
                raise AsrError(409, "checkpoint_version_mismatch", "playback audio or PCM decoding identity differs; retry from the beginning", {"can_retry_from_start": True})
        else:
            await self.store.run(atomic_json, identity_path, stable_identity)
        async with self.lock:
            current = await self.store.run(self.store.job_metadata, job_id)
            current["normalized_asset"] = asset.to_metadata()
            await self.store.run(self.store.save_job, current)
        await self.record_files(job_id, ["operation/audio.wav", "checkpoints/audio-identity.json"])
        await self.settle_processing_reservation(job_id, margin)
        self.check_running(job_id)
        return asset

    async def _run_job(self, job_id: str) -> None:
        async with self.lock:
            try:
                starting = (await self.store.run(self.store.job_metadata, job_id))
            except AsrError as exc:
                if exc.code == "job_not_found":
                    return
                raise
            self.active.add(job_id)
        root = self.job_dir(job_id)
        workspace = root / "work"
        try:
            if starting.get("delete_requested") or starting.get("cancel_requested") or starting["status"] == "cancel_requested":
                raise WorkbenchInterrupted()
            async with self.lock:
                self.check_running(job_id)
                job = await self.store.run(self.store.job_metadata, job_id)
                request = TranscriptionRequest(**job["request"])
                self.assert_accepting(request.model)
                definition = self.app.state.manager.runtime_for(request.model or self.app.state.manager.default_model_id).definition
                if definition.revision != job["model_revision"] or (job.get("model_identity") is not None and await self.store.run(self.execution_identity, request) != job["model_identity"]):
                    raise AsrError(409, "resume_version_unavailable", "saved model revision is unavailable; retry as a new task")
                self.attempts[job_id] = await self.store.run(self.store.begin_attempt, job_id,
                    operation_id=(job.get("recovery") or {}).get("operation_id"))
                job = await self.store.run(self.store.job_metadata, job_id)
                job["started_at"] = job["started_at"] or now()
                job["status"] = "preprocessing"
                job["progress"] = {**job.get("progress", {}), "phase": "preprocessing"}
                await self.store.run(self.store.save_job, job, **self.attempt_arguments(job_id))
            prepared_error = self.preparation_errors.pop(job_id, None)
            if prepared_error is not None:
                raise prepared_error
            asset = self.prepared_assets.pop(job_id, None)
            if asset is None:
                asset = await self.prepare_audio(job_id)
            self.check_running(job_id)
            await self.wait_for_video_capacity(job_id)
            workspace.mkdir(exist_ok=True)
            async def stage(phase: str, progress: dict[str, object]) -> None:
                self.check_running(job_id)
                self.progress[job_id] = {**self.progress.get(job_id, {}), **progress, "phase": phase}
                moment = time.monotonic()
                immediate = "completed_chunks" in progress or "current_chunk" in progress or moment - self.progress_saved.get(job_id, 0) >= 1
                if not immediate:
                    return
                async with self.lock:
                    self.check_running(job_id)
                    current = await self.store.run(self.store.job_metadata, job_id)
                    current["status"] = phase
                    current["progress"] = {**current["progress"], **self.progress[job_id]}
                    await self.store.run(self.store.save_job, current, **self.attempt_arguments(job_id))
                    self.progress_saved[job_id] = moment
            async def before_chunk(index: int, total: int) -> None:
                await stage("transcribing", {"current_chunk": index + 1, "total_chunks": total})
            async def after_chunk(index: int, total: int, result: TranscriptionResult) -> None:
                names = await self.store.run(lambda: [str(path.relative_to(root)) for path in (root / "checkpoints").rglob("*.json")])
                await self.record_files(job_id, names)
                async with self.lock:
                    self.check_running(job_id)
                    await self.store.run(self.store.save_chunk, job_id, {"index": index, "text": result.text,
                        "resolution": "pending" if "moss_coverage_pending" in result.warnings else "resolved"}, total,
                        **self.attempt_arguments(job_id))
                    self.progress_saved[job_id] = time.monotonic()
            result = await run_transcription_path(
                asset.path, workspace=workspace, manager=self.app.state.manager, settings=self.settings,
                request=request, stage_callback=stage, before_chunk=before_chunk, after_chunk=after_chunk,
                checkpoint_dir=root / "checkpoints",
                normalized_asset=asset, cancel_event=self.context_for(job).cancel_event,
            )
            self.check_running(job_id)
            self.prepare_result(result, segment_timestamps=definition.capabilities.segment_timestamps)
            ensure_speakers(result)
            result["expected_speakers"] = job.get("expected_speakers")
            if job.get("expected_speakers") is not None and len(speaker_view(dict(result))["speakers"]) != job["expected_speakers"]:
                result["speaker_count_mismatch"] = True
            await self.cleanup_operation(job_id)
            async with self.lock:
                self.check_running(job_id)
                job = await self.store.run(self.store.job, job_id)
                self.result_version(result, job)
                job["result"] = result
                job["error"] = None
                job["gaps"] = []
                job["completeness"] = result.get("completeness", {"complete": True})
                job["retryability"] = {"retryable": False, "reason": "completed"}
                job["next_action"] = None
                job["progress"] = {"phase": "completed", "percent": 100}
                self.terminal(job, "completed")
                await self.store.run(self.store.save_version, job, **self.attempt_arguments(job_id))
            emit("job.completed")
        except WorkbenchInterrupted:
            job = (await self.store.run(self.store.job_metadata, job_id))
            if self.cancelled(job_id):
                await self.finish_cancel(job)
            elif self.is_stopping():
                job["status"] = "queued"
                job["progress"]["phase"] = "queued"
                (await self.store.run(self.store.save_job, job))
            else:
                await self.finish_cancel(job)
        except asyncio.CancelledError:
            job = (await self.store.run(self.store.job_metadata, job_id))
            if self.cancelled(job_id):
                await self.finish_cancel(job)
            elif self.is_stopping():
                job["status"] = "queued"
                job["progress"]["phase"] = "queued"
                (await self.store.run(self.store.save_job, job))
            else:
                await self.finish_cancel(job)
            # Decoder/split cancellation is an operation outcome. The shared
            # scheduler worker remains alive for other owners' queued work.
        except Exception as exc:
            job = (await self.store.run(self.store.job_metadata, job_id))
            if self.cancelled(job_id):
                await self.finish_cancel(job)
            elif self.is_stopping():
                job["status"] = "queued"
                job["progress"]["phase"] = "queued"
                (await self.store.run(self.store.save_job, job))
            else:
                await self.cleanup_operation(job_id)
                finish_cancel = False
                async with self.lock:
                    current = (await self.store.run(self.store.job_metadata, job_id))
                    if self.cancelled(job_id):
                        finish_cancel = True
                    elif self.is_stopping():
                        current["status"] = "queued"
                        current["progress"]["phase"] = "queued"
                        (await self.store.run(self.store.save_job, current))
                    else:
                        error = exc if isinstance(exc, AsrError) else AsrError(500, "transcription_failed", "transcription failed")
                        details = dict(error.details or {})
                        partial = details.pop("partial_result", None)
                        details.pop("partial_text", None)
                        current["error"] = {"code": error.code, "message": error.message, "details": details}
                        if error.code == "worker_circuit_open":
                            from datetime import datetime, timedelta, timezone
                            current["status"] = "queued"
                            current["next_attempt_at"] = (datetime.now(timezone.utc) + timedelta(seconds=max(1, float(details.get("retry_after_seconds", 5))))).isoformat()
                            current.setdefault("recovery", {})["next_attempt_at"] = current["next_attempt_at"]
                            current["progress"] = {**current["progress"], "phase": "retry_wait", "next_attempt_at": current["next_attempt_at"]}
                            current["next_action"] = "wait"
                            await self.store.run(self.store.save_job, current)
                            return
                        from asr_server.lifecycle import is_recoverable_worker_error
                        attempt = self.attempts.get(job_id)
                        if attempt and is_recoverable_worker_error(error):
                            try:
                                await self.store.run(self.store.schedule_infrastructure_retry, job_id,
                                    attempt["attempt_id"], attempt["generation"], current["error"],
                                    delay_seconds=min(60, 5 * 2 ** int((current.get("recovery") or {}).get("infrastructure_retries", 0))),
                                    retention_days=self.settings.workbench_retention_days)
                                emit("job.retry_scheduled", error_code=error.code)
                                return
                            except AsrError as retry_error:
                                if retry_error.code not in {"infrastructure_retry_exhausted", "retry_budget_exhausted"}:
                                    raise
                                details["recovery_stop_reason"] = retry_error.code
                        if partial:
                            current = await self.store.run(self.store.job, job_id)
                            current["error"] = {"code": error.code, "message": error.message, "details": details}
                            self.prepare_result(partial, segment_timestamps=definition.capabilities.segment_timestamps)
                            ensure_speakers(partial)
                            partial["expected_speakers"] = current.get("expected_speakers")
                            self.result_version(partial, current)
                            current["result"] = partial
                            current["completeness"] = partial.get("completeness", {"complete": False})
                            current["gaps"] = partial.get("gaps", details.get("gaps", []))
                            retryable = partial.get("retryability", details.get("retryability", "exhausted"))
                            current["retryability"] = {"retryable": retryable == "retry", "reason": retryable}
                            current["next_action"] = partial.get("next_action", details.get("next_action", "listen_or_change_parameters"))
                        else:
                            current["retryability"] = {"retryable": error.code not in {"checkpoint_version_mismatch", "resume_version_unavailable", "audio_integrity_failed", "infrastructure_retry_exhausted", "recovery_attempt_exhausted"} and not is_recoverable_worker_error(error), "reason": error.code}
                            current["next_action"] = "retry_from_start" if error.code in {"checkpoint_version_mismatch", "resume_version_unavailable"} else "retry" if current["retryability"]["retryable"] else "listen"
                        self.terminal(current, "failed")
                        current["progress"]["phase"] = "failed"
                        await self.store.run(self.store.save_job, current, **self.attempt_arguments(job_id))
                        emit("job.failed", error_code=error.code, diagnostics=safe_exception(exc))
                if finish_cancel:
                    await self.finish_cancel(current)
        finally:
            self.prepared_assets.pop(job_id, None)
            self.preparation_errors.pop(job_id, None)
            try:
                await self.cleanup_operation(job_id)
            except AsrError as exc:
                if exc.code != "cleanup_failed":
                    raise
            finally:
                self.active.discard(job_id)
            try:
                async with self.lock:
                    job = (await self.store.run(self.store.job_metadata, job_id))
                    if job.get("delete_requested") and job_id not in self.video_tasks and not self.playback_consumers.get(job_id, 0):
                        await self.delete_now(job)
                    elif root.exists():
                        reserved = sum(job.get("retained_file_bytes", {}).values()) + int(job.get("video_reserved_bytes", 0))
                        existing = await self.transaction(lambda db: db.execute("SELECT reserved_bytes FROM jobs WHERE id=?", (job_id,)).fetchone()[0])
                        if existing != reserved:
                            await self.store.run(self.store.save_job, job, reserved_bytes=reserved)
                        await self.sync_disk_reservations()
            except AsrError:
                pass
            self.progress.pop(job_id, None)
            self.progress_saved.pop(job_id, None)
            attempt = self.attempts.pop(job_id, None)
            if attempt is not None:
                await self.store.run(self.store.finish_attempt, job_id, attempt["attempt_id"], attempt["generation"])

    @staticmethod
    def result_version(result: dict[str, Any], job: dict[str, Any]) -> None:
        previous = job.get("result") or {}
        version = int(previous.get("version", 0)) + 1
        result["version"] = version
        result["versions"] = [*previous.get("versions", []),
            {"version": version, "created_at": now(), "alignment_status": result["alignment_status"]}]

    def prepare_result(self, result: dict[str, Any], *, segment_timestamps: bool = False) -> None:
        model_segments = bool(result.get("segments")) and segment_timestamps
        segments = result.get("segments") or result.get("chunks") or []
        for index, segment in enumerate(segments):
            segment["id"] = str(segment.get("id", f"segment-{index}"))
            segment["index"] = index
            segment.setdefault("raw_text", segment.get("text", ""))
            segment.setdefault("timestamps", [])
        result["segments"] = segments
        result.setdefault("raw_text", result.get("text", ""))
        result["version"] = 1
        incomplete = any(segment.get("reading_segmentation") == "unmapped_alignment" for segment in segments)
        result["alignment_status"] = "incomplete" if incomplete else "aligned" if result.get("timestamps") else "unavailable"
        if result["alignment_status"] == "unavailable" and model_segments and valid_segment_timing(result):
            result["alignment_status"] = "segment_timed"
            result["timestamp_source"] = "model_segments"
        result["changes"] = result.get("quality", {}).get("changes", [])
        result["versions"] = [{"version": 1, "created_at": now(), "alignment_status": result["alignment_status"]}]

    async def finish_cancel(self, job: dict[str, Any]) -> None:
        video_task = self.video_tasks.get(job["id"])
        if video_task is not None and video_task is not asyncio.current_task():
            await asyncio.shield(video_task)
        await self.cleanup_operation(job["id"])
        async with self.lock:
            try:
                # DELETE may have been requested while file cleanup awaited.
                # Publish a terminal state from current durable intent only.
                current = (await self.store.run(self.store.job_metadata, job["id"]))
            except AsrError as exc:
                if exc.code == "job_not_found":
                    return
                raise
            self.terminal(current, "cancelled")
            current["cancel_requested"] = True
            current["progress"]["phase"] = "cancelled"
            self.prepared_assets.pop(job["id"], None)
            await self.store.run(self.record_files_sync, job["id"], ["source", "audio.mp3"])
            current["retained_file_bytes"] = (await self.store.run(self.store.job_metadata, job["id"]))["retained_file_bytes"]
            await self.store.run(self.store.save_job, current, reserved_bytes=sum(current["retained_file_bytes"].values()))
            await self.sync_disk_reservations()
            emit("job.cancelled")

    async def cancel(self, owner: str, job_id: str) -> dict[str, Any]:
        video_only = False
        async with self.lock:
            job = await self.store.run(self.store.job, job_id, owner)
            if job["status"] in TERMINAL:
                if (job.get("video_asset") or {}).get("state") not in {"queued", "preparing"}:
                    return self.public_job(job)
                video_only = True
                job["video_asset"] = {"state": "failed", "error_code": "video_interrupted"}
            else:
                job["status"] = "cancel_requested"
                job["cancel_requested"] = True
            self.cancel_intents.add(job_id)
            self.context_for(job).cancel_event.set()
            await self.store.run(self.store.save_job, job)
        removed = await self.app.state.job_manager.cancel_external(("video:" if video_only else "") + job_id)
        if not video_only:
            async with self.lock:
                finish_now = removed or job_id not in self.active
            if finish_now:
                await self.finish_cancel(job)
        return self.public_job(await self.store.run(self.store.job, job_id, owner))

    async def delete(self, owner: str, job_id: str) -> bool:
        async with self.lock:
            job = (await self.store.run(self.store.job, job_id, owner))
            job["delete_requested"] = True
            job["cancel_requested"] = True
            job["status"] = "cancel_requested"
            self.cancel_intents.add(job_id)
            self.context_for(job).cancel_event.set()
            (await self.store.run(self.store.save_job, job))
        await self.app.state.job_manager.cancel_external(job_id)
        await self.app.state.job_manager.cancel_external("align:" + job_id)
        await self.app.state.job_manager.cancel_external("video:" + job_id)
        async with self.lock:
            try:
                job = await self.store.run(self.store.job_metadata, job_id, owner)
            except AsrError as exc:
                if exc.code == "job_not_found":
                    return True
                raise
            if job_id in self.active or job_id in self.video_tasks or self.playback_consumers.get(job_id, 0):
                return False
            await self.delete_now(job)
            return True

    async def delete_now(self, job: dict[str, Any]) -> None:
        path = self.job_dir(job["id"])
        try:
            if path.exists():
                await asyncio.to_thread(shutil.rmtree, path)
        except OSError as exc:
            raise AsrError(503, "cleanup_failed", "task files could not be removed; retry deletion") from exc
        lease = self.operation_leases.pop(job["id"], None)
        if lease is not None:
            await lease.release()
        def remove(db: sqlite3.Connection) -> None:
            db.execute("DELETE FROM versions WHERE job_id=?", (job["id"],))
            db.execute("DELETE FROM uploads WHERE job_id=?", (job["id"],))
            db.execute("DELETE FROM jobs WHERE id=?", (job["id"],))
        await self.transaction(remove)
        await self.store.run(self.store.set_physical_usage, job["id"], job["owner_id"], 0)
        await self.sync_disk_reservations()
        self.contexts.pop(job["id"], None)
        self.cancel_intents.discard(job["id"])
        self.progress.pop(job["id"], None)
        self.progress_saved.pop(job["id"], None)
        self.preparation_errors.pop(job["id"], None)
        self.prepared_assets.pop(job["id"], None)

    async def acquire_playback(self, owner: str, job_id: str, *, video: bool = False) -> Path:
        async with self.lock:
            job = await self.store.run(self.store.job_metadata, job_id, owner)
            kind = "video" if video else "audio"
            state = (job.get(kind + "_asset") or {}).get("state", "preparing")
            path = self.job_dir(job_id) / ("video.mp4" if video else "audio.mp3")
            if job.get("delete_requested"):
                raise AsrError(409, "job_deleting", "task deletion is in progress")
            if state != "ready" or not path.exists():
                raise AsrError(409, kind + "_not_ready", "playback media is not ready", {"state": state})
            self.playback_consumers[job_id] = self.playback_consumers.get(job_id, 0) + 1
            return path

    async def release_playback(self, job_id: str) -> None:
        async with self.lock:
            remaining = self.playback_consumers.get(job_id, 0) - 1
            if remaining > 0:
                self.playback_consumers[job_id] = remaining
                return
            self.playback_consumers.pop(job_id, None)
            if job_id not in self.active and job_id not in self.video_tasks:
                job = await self.store.run(self.store.job_metadata, job_id)
                if job.get("delete_requested"):
                    await self.delete_now(job)

    async def retry(self, owner: str, job_id: str, *, from_start: bool = False) -> dict[str, Any]:
        self.assert_accepting()
        async with self.lock:
            job = (await self.store.run(self.store.job, job_id, owner))
            self.assert_accepting(job["request"]["model"])
            if job_id in self.active or job_id in self.video_tasks:
                raise AsrError(409, "job_busy", "wait for the active operation before retrying transcription")
            if job["status"] not in {"failed", "cancelled"}:
                raise AsrError(409, "job_not_retryable", "only failed or cancelled tasks can be retried")
            if not (self.job_dir(job_id) / "audio.mp3").exists() and not (self.job_dir(job_id) / "source").exists():
                raise AsrError(409, "source_unavailable", "original audio is unavailable")
            retryability = job.get("retryability") or {}
            if not from_start and isinstance(retryability, dict) and retryability.get("retryable") is False:
                raise AsrError(409, "job_not_retryable", "saved recovery actions are exhausted; inspect the unresolved intervals",
                    {"next_action": job.get("next_action"), "reason": retryability.get("reason")})
            if not from_start:
                await self.store.run(self.import_legacy_failure, job)
            if from_start:
                checkpoint_root = self.job_dir(job_id) / "checkpoints"
                if checkpoint_root.exists():
                    await self.store.run(shutil.rmtree, checkpoint_root)
                job.pop("normalized_asset", None)
                job["model_identity"] = await self.store.run(self.execution_identity, TranscriptionRequest(**job["request"]))
                job["model_revision"] = self.app.state.manager.runtime_for(job["request"]["model"]).definition.revision
                with self.store.connection() as db:
                    db.execute("DELETE FROM job_chunks WHERE job_id=?", (job_id,))
                job["retained_file_bytes"] = {name: size for name, size in job.get("retained_file_bytes", {}).items() if not name.startswith("checkpoints/")}
            job["status"] = "queued"
            job["queued_at"] = now()
            job["cancel_requested"] = False
            job["error"] = None
            job["completed_at"] = None
            job["expires_at"] = None
            job["progress"] = {"phase": "queued", "percent": 0} if from_start else {**job.get("progress", {}), "phase": "queued"}
            job.pop("next_attempt_at", None)
            if job.get("recovery"):
                # A user action starts one new operation, while the task-wide
                # infrastructure counter and quality ledger remain intact.
                job["recovery"] = {**job["recovery"], "operation_id": identifier("operation"), "operation_retries": 0,
                    "active": False}
                job["recovery"].pop("next_attempt_at", None)
            previous = self.context_for(job)
            context = previous.with_fields(attempt=previous.attempt + 1, cancel_event=threading.Event())
            self.contexts[job_id] = context
            self.cancel_intents.discard(job_id)
            job["execution_context"] = context.to_wire()
            (await self.store.run(self.store.save_job, job))
            await self.enqueue(job_id)
            return self.public_job(job)

    def import_legacy_failure(self, job: dict[str, Any]) -> None:
        """Freeze the original failure before retry clears it; never invent a call."""
        checkpoint = self.job_dir(job["id"]) / "checkpoints"
        path = checkpoint / "recovery-import.json"
        error = job.get("error") or {}
        details = error.get("details") or {}
        if path.exists() or (checkpoint / "recovery-v1" / "state.json").exists():
            return
        if error.get("code") != "quality_incomplete" or details.get("cause") not in {"empty_transcription", "generation_truncated"}:
            return
        try:
            identity = json.loads((checkpoint / "identity.json").read_text())
            if not isinstance(identity, dict) or not isinstance(identity.get("fingerprint"), str):
                raise ValueError("invalid checkpoint identity")
        except (OSError, ValueError, TypeError) as exc:
            raise AsrError(409, "checkpoint_corrupt", "saved checkpoint identity cannot be read") from exc
        index = details.get("chunk_index")
        windows = job.get("progress", {}).get("chunk_windows", [])
        if not isinstance(index, int) or index < 0 or index >= len(windows) or windows[index] != [details.get("start"), details.get("end")]:
            raise AsrError(409, "checkpoint_version_mismatch", "legacy failure does not match its saved chunk plan")
        atomic_json(path, {"base_fingerprint": identity["fingerprint"],
            "failed_chunks": [{"index": index, "cause": details["cause"], "start": details["start"], "end": details["end"]}],
            "source_error_digest": fingerprint(error), "source_job_fingerprint": fingerprint({"request": job["request"],
                "model_identity": job.get("model_identity"), "source_sha256": job.get("source_sha256"),
                "audio_asset": job.get("audio_asset"), "normalized_asset": job.get("normalized_asset"), "chunk_windows": windows}),
            "imported_at": now()})

    async def editable_job(self, owner: str, job_id: str, base_version: int) -> dict[str, Any]:
        job = await self.store.run(self.store.job, job_id, owner)
        result = job.get("result")
        if job.get("delete_requested") or job["status"] != "completed" or result is None:
            raise AsrError(409, "job_not_completed", "only completed transcripts can be edited")
        if result["version"] != base_version:
            raise AsrError(409, "version_conflict", "transcript changed; refresh before editing")
        if result["alignment_status"] == "pending":
            raise AsrError(409, "alignment_pending", "wait for the current edit to finish alignment")
        ensure_speakers(result)
        return job

    async def save_edit(self, job: dict[str, Any], changes: list[dict[str, Any]]) -> None:
        result = job["result"]
        result["version"] += 1
        for change in changes:
            result.setdefault("changes", []).append({**change, "created_at": now(), "version": result["version"]})
        result["versions"].append({"version": result["version"], "created_at": now(), "alignment_status": result["alignment_status"]})
        await self.store.run(self.check_capacity, job["owner_id"], len(json.dumps(result, ensure_ascii=False).encode("utf-8")))
        await self.store.run(self.store.save_version, job)

    def mark_pending_alignment(self, job: dict[str, Any], targets: list[dict[str, Any]]) -> None:
        self.assert_accepting(job["request"]["model"])
        result = job["result"]
        for target in targets:
            target["timestamps"] = []
            target["alignment_status"] = "pending"
        result["timestamps"] = []
        result["alignment_status"] = "pending"
        result.pop("pending_segment_id", None)
        result["pending_segment_ids"] = [target["id"] for target in targets]
        result.pop("alignment_error", None)

    async def save_text_edit(self, job: dict[str, Any], changes: list[dict[str, Any]], align: bool) -> None:
        await self.save_edit(job, changes)
        if align:
            try:
                await self.enqueue_alignment(job["id"])
            except AsrError as exc:
                if exc.code != "job_queue_full":
                    raise

    async def edit(self, owner: str, job_id: str, segment_id: str, text: str | None, base_version: int,
                   *, speaker_id: str | None = None, assign_speaker: bool = False) -> dict[str, Any]:
        async with self.lock:
            job = await self.editable_job(owner, job_id, base_version)
            result = job["result"]
            target = next((s for s in result["segments"] if s["id"] == segment_id), None)
            if target is None:
                raise AsrError(404, "segment_not_found", "segment not found")
            definition = self.app.state.manager.runtime_for(job["request"]["model"]).definition
            changes: list[dict[str, Any]] = []
            align = False
            if assign_speaker:
                if not definition.capabilities.diarization:
                    raise AsrError(422, "capability_not_supported", "model does not support speakers")
                if speaker_id is not None and speaker_id not in result["speakers"]:
                    raise AsrError(404, "speaker_not_found", "speaker not found")
                if speaker_id != target.get("speaker_id") or target.get("speaker_assignment") != "user":
                    changes.append({"method": "speaker_assignment", "segment_id": segment_id, "before": target.get("speaker_id"), "after": speaker_id})
                    target["speaker_id"] = speaker_id
                    target["speaker_assignment"] = "user"
            if text is not None and text.strip() != target["text"]:
                if not text.strip() or len(text) > 20000:
                    raise AsrError(400, "bad_request", "edited text must be nonempty and at most 20000 characters")
                if not (definition.capabilities.forced_alignment or definition.capabilities.segment_timestamps):
                    raise AsrError(422, "capability_not_supported", "model does not support timed text edits")
                changes.append({"method": "user_edit", "segment_id": segment_id, "before": target["text"], "after": text.strip()})
                target["text"] = text.strip()
                target["text_edited"] = True
                result["text"] = "\n\n".join(s["text"] for s in result["segments"] if s["text"])
                align = definition.capabilities.forced_alignment
                if align:
                    self.mark_pending_alignment(job, [target])
            if changes:
                await self.save_text_edit(job, changes, align)
            return self.public_job(job)

    async def assign_pending(self, owner: str, job_id: str, segment_ids: list[str], speaker_id: str,
                             base_version: int) -> dict[str, Any]:
        async with self.lock:
            job = await self.editable_job(owner, job_id, base_version)
            if not self.app.state.manager.runtime_for(job["request"]["model"]).definition.capabilities.diarization:
                raise AsrError(422, "capability_not_supported", "model does not support speakers")
            result = job["result"]
            speakers = result["speakers"]
            if speaker_id not in speakers:
                raise AsrError(404, "speaker_not_found", "speaker not found")
            ids = set(segment_ids)
            targets = [s for s in result["segments"] if s["id"] in ids]
            if len(targets) != len(ids):
                raise AsrError(404, "segment_not_found", "segment not found")
            if any(not pending_speaker(s, speakers) for s in targets):
                raise AsrError(409, "speaker_already_assigned", "select only segments whose speaker is pending")
            for target in targets:
                target["speaker_id"] = speaker_id
                target["speaker_assignment"] = "user"
            await self.save_edit(job, [{"method": "speaker_assignment_batch", "segment_ids": [s["id"] for s in targets],
                                        "before": f"{len(targets)} 段说话人待确认",
                                        "after": speakers[speaker_id]["name"], "speaker_id": speaker_id}])
            return self.public_job(job)

    async def replace_text(self, owner: str, job_id: str, find: str, replacement: str,
                           base_version: int) -> dict[str, Any]:
        async with self.lock:
            job = await self.editable_job(owner, job_id, base_version)
            if not find.strip() or find == replacement:
                raise AsrError(400, "bad_request", "provide nonempty search text and a different replacement")
            result = job["result"]
            segments = result["segments"]
            original_text = result["text"]
            original_segments = "".join(s["text"] for s in segments)
            targets: list[dict[str, Any]] = []
            occurrences = 0
            for entry in segments or [result]:
                original = entry["text"]
                count = original.count(find)
                if not count:
                    continue
                if len(original) + count * (len(replacement) - len(find)) > max(20000, len(original)):
                    raise AsrError(400, "replacement_too_long", "replacement would make a text block too long")
                occurrences += count
                entry.setdefault("raw_text", original)
                entry["text"] = original.replace(find, replacement)
                if segments:
                    entry["text_edited"] = True
                    targets.append(entry)
            if not occurrences:
                return self.public_job(job)

            capabilities = self.app.state.manager.runtime_for(job["request"]["model"]).definition.capabilities
            if segments:
                result.setdefault("raw_text", original_text)
                canonical_matches = original_text.count(find)
                if "".join(original_segments.split()) == "".join(original_text.split()):
                    edited_segments = "\n\n".join(s["text"] for s in segments if s["text"])
                    candidate = original_text.replace(find, replacement) if canonical_matches == occurrences else edited_segments
                    result["text"] = candidate if "".join(candidate.split()) == "".join(edited_segments.split()) else edited_segments
                else:
                    # Some older results include text outside the timed segments.
                    extra_growth = (canonical_matches - occurrences) * (len(replacement) - len(find))
                    if canonical_matches > occurrences and extra_growth > 20000:
                        raise AsrError(400, "replacement_too_long", "replacement would make the transcript too long")
                    result["text"] = original_text.replace(find, replacement)
            elif result.get("timestamps") or result["alignment_status"] == "aligned":
                result["timestamps"] = []
                result["alignment_status"] = "unavailable"
            align = bool(targets and capabilities.forced_alignment)
            if align:
                self.mark_pending_alignment(job, targets)
            await self.save_text_edit(job, [{"method": "text_replace", "before": find, "after": replacement,
                                            "occurrences": occurrences, "segment_count": len(targets)}], align)
            return self.public_job(job)

    async def edit_speaker(self, owner: str, job_id: str, name: str, base_version: int,
                           speaker_id: str | None = None) -> dict[str, Any]:
        async with self.lock:
            job = await self.editable_job(owner, job_id, base_version)
            if not self.app.state.manager.runtime_for(job["request"]["model"]).definition.capabilities.diarization:
                raise AsrError(422, "capability_not_supported", "model does not support speakers")
            name = name.strip()
            if not name or len(name) > 120 or any(ord(c) < 32 for c in name):
                raise AsrError(400, "bad_request", "speaker name must contain 1 to 120 printable characters")
            people = job["result"]["speakers"]
            if speaker_id is None:
                if len(people) >= 256:
                    raise AsrError(400, "bad_request", "speaker limit reached")
                speaker_id = f"P{max([int(k[1:]) for k in people] or [0]) + 1}"
                people[speaker_id] = {"name": name, "scope": "global", "source": "user"}
                change = {"method": "speaker_create", "speaker_id": speaker_id, "after": name}
            else:
                if speaker_id not in people:
                    raise AsrError(404, "speaker_not_found", "speaker not found")
                change = {"method": "speaker_rename", "speaker_id": speaker_id, "before": people[speaker_id]["name"], "after": name}
                if people[speaker_id]["name"] == name:
                    return self.public_job(job)
                people[speaker_id]["name"] = name
            await self.save_edit(job, [change])
            return self.public_job(job)

    async def enqueue_alignment(self, job_id: str) -> None:
        job = await self.store.run(self.store.job, job_id)
        await self.app.state.job_manager.enqueue_external("align:" + job_id, lambda: self.align_edit(job_id), context=self.context_for(job))

    async def align_edit(self, job_id: str) -> None:
        async with self.lock:
            try:
                starting = (await self.store.run(self.store.job, job_id))
            except AsrError as exc:
                if exc.code == "job_not_found":
                    return
                raise
            if starting.get("delete_requested") or starting.get("cancel_requested"):
                return
            self.active.add(job_id)
        try:
            job = (await self.store.run(self.store.job, job_id))
            result = job["result"]
            segment_ids = set(result.get("pending_segment_ids") or [result.get("pending_segment_id")]) - {None}
            if result["alignment_status"] != "pending" or not segment_ids:
                return
            self.check_running(job_id)
            self.attempts[job_id] = await self.store.run(self.store.begin_attempt, job_id, kind="alignment")
            asset = await self.prepare_audio(job_id)
            for target in result["segments"]:
                if target["id"] not in segment_ids:
                    continue
                self.check_running(job_id)
                audio = AudioPath(asset.path, target["start"], target["end"])
                items = await self.app.state.manager.align_text(
                    audio, text=target["text"], model_id=job["request"]["model"],
                    backend="auto", language=job["request"]["language"], timestamps="auto",
                ) if target["text"].strip() else []
                target["timestamps"] = [{**item.to_api(), "start": item.start + target["start"],
                                         "end": item.end + target["start"]} for item in items]
                target["alignment_status"] = "aligned"
                target.pop("reading_segmentation", None)
            await self.cleanup_operation(job_id)
            async with self.lock:
                latest = (await self.store.run(self.store.job, job_id))
                if latest.get("delete_requested") or latest["status"] == "cancel_requested":
                    return
                if latest["result"]["version"] != result["version"]:
                    return
                job = latest
                job["result"] = result
                result["timestamps"] = [item for s in result["segments"] for item in s.get("timestamps", [])]
                result["alignment_status"] = "aligned" if all(
                    not s["text"].strip() or (s.get("timestamps") and s.get("reading_segmentation") != "unmapped_alignment")
                    for s in result["segments"]
                ) else "incomplete"
                result.pop("pending_segment_id", None)
                result.pop("pending_segment_ids", None)
                result["versions"][-1]["alignment_status"] = result["alignment_status"]
                await self.store.run(self.store.save_version, job, **self.attempt_arguments(job_id))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.cleanup_operation(job_id)
            job = (await self.store.run(self.store.job, job_id))
            if self.is_stopping() or job.get("delete_requested") or job.get("cancel_requested"):
                return  # Keep pending alignment for recovery, or honor deletion.
            if job.get("result"):
                result = job["result"]
                result["alignment_status"] = "failed"
                result["alignment_error"] = {"code": exc.code if isinstance(exc, AsrError) else "alignment_failed", "message": str(exc) if isinstance(exc, AsrError) else "alignment failed"}
                result["versions"][-1]["alignment_status"] = "failed"
                await self.store.run(self.store.save_version, job, **self.attempt_arguments(job_id))
        finally:
            attempt = self.attempts.pop(job_id, None)
            if attempt is not None:
                await self.store.run(self.store.finish_attempt, job_id, attempt["attempt_id"], attempt["generation"])
            try:
                await self.cleanup_operation(job_id)
            except AsrError as exc:
                if exc.code != "cleanup_failed":
                    raise
            finally:
                self.active.discard(job_id)
            try:
                job = (await self.store.run(self.store.job, job_id))
                if job.get("delete_requested") and job_id not in self.video_tasks and not self.playback_consumers.get(job_id, 0):
                    await self.delete_now(job)
            except AsrError:
                pass

    def reconcile_capacity(self) -> None:
        """Startup/low-frequency repair only; admission and polling never walk."""
        with self.store.connection() as db:
            uploads = db.execute("SELECT id,owner_id,offset FROM uploads WHERE job_id IS NULL").fetchall()
            jobs = db.execute("SELECT id,owner_id FROM jobs").fetchall()
        live_assets = {str(row["id"]) for row in [*uploads, *jobs]}
        for upload in uploads:
            path = self.upload_path(upload["id"])
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                size = 0
            self.store.set_physical_usage(upload["id"], upload["owner_id"], size)
        for row in jobs:
            # A running operation updates its files at durable chunk boundaries.
            # Do not overwrite its in-flight accounting with a racy scan.
            if row["id"] in self.active or row["id"] in self.video_tasks:
                continue
            root = self.job_dir(row["id"])
            files: dict[str, int] = {}
            seen: set[tuple[int, int]] = set()
            for path in root.rglob("*"):
                try:
                    stat = path.stat()
                    inode = (stat.st_dev, stat.st_ino)
                    if path.is_file() and inode not in seen:
                        files[path.relative_to(root).as_posix()] = stat.st_size
                        seen.add(inode)
                except FileNotFoundError:
                    continue
            temporary, text = self.store.file_categories(files)
            self.store.set_physical_usage(row["id"], row["owner_id"], sum(files.values()), temporary_bytes=temporary, text_file_bytes=text)
            with self.store.connection() as db:
                db.execute("UPDATE jobs SET payload=json_set(payload,'$.retained_file_bytes',json(?)) WHERE id=?", (json.dumps(files), row["id"]))
        with self.store.connection() as db:
            stale = db.execute("SELECT asset_id,owner_id FROM assets").fetchall()
        for asset in stale:
            if asset["asset_id"] not in live_assets:
                self.store.set_physical_usage(asset["asset_id"], asset["owner_id"], 0)

    async def cleanup(self) -> None:
        async with self.lock:
            pending = await self.transaction(lambda db: db.execute("SELECT id FROM jobs WHERE status='cleanup_pending' OR json_extract(payload,'$.delete_requested')=1").fetchall())
            for row in pending:
                key = row["id"]
                if key in self.active or key in self.video_tasks or self.playback_consumers.get(key, 0):
                    continue
                try:
                    task = await self.store.run(self.store.job_metadata, key)
                    if task.get("delete_requested"):
                        await self.delete_now(task)
                    else:
                        await self.cleanup_operation(key, locked=False)
                        task = await self.store.run(self.store.job_metadata, key)
                        self.terminal(task, "cancelled" if task.get("cancel_requested") else "failed")
                        task["progress"] = {"phase": task["status"]}
                        await self.store.run(self.store.save_job, task)
                except AsrError as exc:
                    if exc.code != "cleanup_failed":
                        raise
            expired_ids = await self.transaction(lambda db: db.execute("SELECT id FROM jobs WHERE status IN ('completed','failed','cancelled') AND json_extract(summary,'$.expires_at')<=?", (now(),)).fetchall())
            for row in expired_ids:
                if row["id"] not in self.active and row["id"] not in self.video_tasks and not self.playback_consumers.get(row["id"], 0):
                    await self.delete_now(await self.store.run(self.store.job, row["id"]))
            def cleanup_rows(db: sqlite3.Connection) -> tuple[set[str], set[str]]:
                expired = db.execute("SELECT id,owner_id FROM uploads WHERE job_id IS NULL AND expires_at<=?", (now(),)).fetchall()
                for upload in expired:
                    lock = self.upload_locks.get(upload["id"])
                    if lock is not None and lock.locked():
                        continue
                    self.upload_path(upload["id"]).unlink(missing_ok=True)
                    db.execute("DELETE FROM uploads WHERE id=?", (upload["id"],))
                db.execute("DELETE FROM sessions WHERE expires_at<=?", (now(),))
                uploads = db.execute("SELECT id FROM uploads WHERE job_id IS NULL").fetchall()
                jobs = db.execute("SELECT id FROM jobs").fetchall()
                return {str(row[0]) for row in uploads}, {str(row[0]) for row in jobs}
            upload_ids, job_ids = await self.transaction(cleanup_rows)
            def remove_orphans() -> None:
                uploads_root = self.store.root / "uploads"
                if uploads_root.exists():
                    for path in uploads_root.iterdir():
                        if path.is_file() and path.name not in upload_ids:
                            path.unlink(missing_ok=True)
                jobs_root = self.store.root / "jobs"
                if jobs_root.exists():
                    for path in jobs_root.iterdir():
                        if path.is_dir() and path.name not in job_ids:
                            shutil.rmtree(path, ignore_errors=True)
            await self.store.run(remove_orphans)
            self.reconciling = True
            try:
                await self.store.run(self.reconcile_capacity)
                await self.sync_disk_reservations()
            finally:
                self.reconciling = False
            for upload_id, lock in list(self.upload_locks.items()):
                if upload_id not in upload_ids and not lock.locked():
                    self.upload_locks.pop(upload_id, None)


def timestamp(seconds: float, *, vtt: bool = False) -> str:
    total_ms = max(0, round(seconds * 1000))
    hour, rest = divmod(total_ms, 3600000)
    minute, rest = divmod(rest, 60000)
    second, ms = divmod(rest, 1000)
    return f"{hour:02d}:{minute:02d}:{second:02d}{'.' if vtt else ','}{ms:03d}"


def valid_segment_timing(result: dict[str, Any]) -> bool:
    segments = [s for s in result.get("segments", []) if s.get("text", "").strip()]
    if not segments:
        return False
    try:
        duration = float(result["duration"])
        if not math.isfinite(duration) or duration <= 0:
            return False
        previous_start = -1.0
        for segment in segments:
            if segment.get("timestamp_source") not in {None, "model_segments"}:
                return False  # historical fallback chunks carry vad_chunk_window
            start, end = float(segment["start"]), float(segment["end"])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration and start >= previous_start):
                return False
            previous_start = start
        # A parser dropping some text must not silently produce partial subtitles.
        return "".join("".join(s["text"].split()) for s in segments) == "".join(result["text"].split())
    except (KeyError, TypeError, ValueError):
        return False


def export_result(result: dict[str, Any], format_: str, *, segment_timestamps: bool = False) -> tuple[str, str]:
    if format_ == "txt":
        gaps = result.get("gaps") or []
        incomplete = result.get("completeness", {}).get("complete") is False
        marker = "【不完整转录】\n" if incomplete else ""
        if incomplete:
            marker += "".join(f"缺口 {timestamp(float(gap['start']), vtt=True)}–{timestamp(float(gap['end']), vtt=True)}：{gap.get('reason', 'unresolved')}\n" for gap in gaps)
            marker += "\n"
        return marker + result["text"] + "\n", "text/plain"
    if format_ == "json":
        return json.dumps(result, ensure_ascii=False, indent=2) + "\n", "application/json"
    if format_ not in {"srt", "vtt"}:
        raise AsrError(400, "bad_request", "unsupported export format")
    if result.get("completeness", {}).get("complete") is False or result.get("gaps"):
        raise AsrError(409, "transcript_incomplete", "subtitles are unavailable while transcript intervals remain unresolved")
    model_timing = (segment_timestamps and result.get("alignment_status") in {"unavailable", "segment_timed"}
                    and valid_segment_timing(result))
    if result.get("alignment_status") != "aligned" and not model_timing:
        raise AsrError(409, "alignment_pending", "subtitles are unavailable until this transcript version is aligned")
    cues: list[str] = []
    for segment in result["segments"]:
        if not segment["text"].strip():
            continue
        items = segment.get("timestamps", [])
        if model_timing:
            start, end = segment["start"], segment["end"]
        elif not items:
            raise AsrError(409, "alignment_incomplete", "a transcript segment has no real alignment")
        else:
            start, end = items[0]["start"], items[-1]["end"]
        cues.append(f"{len(cues) + 1}\n{timestamp(start, vtt=format_ == 'vtt')} --> {timestamp(end, vtt=format_ == 'vtt')}\n{segment['text']}")
    text = ("WEBVTT\n\n" if format_ == "vtt" else "") + "\n\n".join(cues) + "\n"
    return text, "text/vtt" if format_ == "vtt" else "application/x-subrip"
