from __future__ import annotations

import asyncio
import copy
import hashlib
import json
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
from asr_server.audio.assets import NormalizedAudioAsset
from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.observability import ExecutionContext, bind_context, current_context, emit, safe_exception
from asr_server.transcription import TranscriptionRequest, run_transcription_path, validate_transcription_request
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
        self.stopping = False
        self.maintenance: asyncio.Task[None] | None = None
        self.active: set[str] = set()
        self.process_lock: BinaryIO | None = None
        self.upload_locks: dict[str, asyncio.Lock] = {}
        self.contexts: dict[str, ExecutionContext] = {}
        self.cancel_intents: set[str] = set()
        self.progress: dict[str, dict[str, Any]] = {}
        self.progress_saved: dict[str, float] = {}
        self.reconciling = False
        self.preparation_errors: dict[str, BaseException] = {}

    async def transaction(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        def execute() -> T:
            with self.store.connection() as db:
                return operation(db)
        return await self.store.run(execute)

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
        self.maintenance = asyncio.create_task(self.maintenance_loop())

    async def shutdown(self) -> None:
        self.stopping = True
        for context in self.contexts.values():
            context.cancel_event.set()
        if self.maintenance is not None:
            self.maintenance.cancel()
            try:
                await self.maintenance
            except asyncio.CancelledError:
                pass

    def close(self) -> None:
        """Release only after the shared queue/model manager has shut down."""
        if self.process_lock is not None:
            import fcntl
            fcntl.flock(self.process_lock.fileno(), fcntl.LOCK_UN)
            self.process_lock.close()
            self.process_lock = None
        self.store.close()

    async def maintenance_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            await self.cleanup()
            await self.fill_queue()

    async def fill_queue(self) -> None:
        for job_id in await self.store.run(self.store.pending_jobs):
            job = await self.store.run(self.store.job, job_id)
            if job["id"] in self.active:
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
            physical = sum(job.get("retained_file_bytes", {}).values())
            reserved = await self.transaction(lambda db: int(db.execute("SELECT reserved_bytes FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]))
            desired = physical + extra_bytes
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
            await self.store.run(self.store.save_job, current, reserved_bytes=retained + margin)
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
            backend="auto", max_new_tokens=None, context="", hotwords=str(parameters.get("hotwords", "")) or None,
            split_strategy="auto", max_chunk_seconds=None, overlap_seconds=None, preserve_segments=True,
            quality_profile=profile, glossary_snapshot=self.snapshot(str(parameters.get("domain", "general")), str(parameters.get("hotwords", ""))),
        )
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
            key = parameters.get("idempotency_key") or upload_id
            if not isinstance(key, str) or len(key) > 200:
                raise AsrError(400, "bad_request", "invalid idempotency key")
            async with self.lock:
                # Revalidate all mutable inputs after the full-file hash.
                latest = await self.store.run(self.upload, owner, upload_id)
                if latest["offset"] != upload["offset"] or latest["job_id"] is not None:
                    raise AsrError(409, "idempotency_conflict", "upload changed during submission")
                current_request = await self.store.run(self.submission_request, parameters)
                if await self.store.run(self.execution_identity, current_request) != identity or asdict(current_request) != asdict(request):
                    raise AsrError(409, "version_conflict", "model or glossary changed during submission; retry")
                fingerprint = hashlib.sha256(json.dumps({"source_sha256": digest, "model_identity": identity,
                    "request": asdict(request)}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
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
                  "progress", "error", "result", "model_revision"}
        payload = {key: value for key, value in job.items() if key in fields}
        if not include_result:
            payload.pop("result", None)
        payload["model"] = job.get("request", {}).get("model")
        payload["queue_position"] = self.app.state.job_manager.external_queue_position(job["id"])
        return payload

    async def enqueue(self, job_id: str) -> None:
        job = await self.store.run(self.store.job, job_id)
        await self.app.state.job_manager.enqueue_external(job_id, lambda: self.run_job(job_id), context=self.context_for(job),
            prepare=lambda: self.prepare_for_scheduler(job_id), dispose=lambda: self.dispose_preparation(job_id))

    async def prepare_for_scheduler(self, job_id: str) -> None:
        added = job_id not in self.active
        if added:
            self.active.add(job_id)
        try:
            await self.prepare_audio(job_id)
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
        await self.store.run((self.job_dir(job_id) / "audio.partial.wav").unlink, missing_ok=True)
        def release_unused() -> None:
            try:
                self.record_files_sync(job_id, ["source", "audio.wav"])
                job = self.store.job_metadata(job_id)
                self.store.save_job(job, reserved_bytes=sum(job.get("retained_file_bytes", {}).values()))
            except AsrError as exc:
                if exc.code != "job_not_found":
                    raise
        await self.store.run(release_unused)
        await self.sync_disk_reservations()

    def cancelled(self, job_id: str) -> bool:
        return job_id in self.cancel_intents

    def check_running(self, job_id: str) -> None:
        context = self.contexts.get(job_id)
        if self.stopping or self.cancelled(job_id) or (context is not None and context.cancel_event.is_set()):
            raise WorkbenchInterrupted()

    def is_stopping(self) -> bool:
        # Re-evaluate after awaited cleanup; shutdown may have started meanwhile.
        return self.stopping

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
        for name in names:
            path = self.job_dir(job_id) / name
            try:
                sizes[name] = path.stat().st_size
            except FileNotFoundError:
                sizes.pop(name, None)
        self.store.set_physical_usage(job_id, current["owner_id"], sum(sizes.values()))
        self.store.save_job(current)

    async def prepare_audio(self, job_id: str) -> NormalizedAudioAsset:
        self.check_running(job_id)
        job = await self.store.run(self.store.job_metadata, job_id)
        request = TranscriptionRequest(**job["request"])
        definition = self.app.state.manager.runtime_for(request.model or self.app.state.manager.default_model_id).definition
        if job.get("model_identity") is not None and await self.store.run(self.execution_identity, request) != job["model_identity"]:
            raise AsrError(409, "resume_version_unavailable", "saved execution identity is unavailable; submit a new task")
        root = self.job_dir(job_id)
        audio = root / "audio.wav"
        source = root / "source"
        if not audio.exists():
            context = self.context_for(job)
            duration = await probe_audio_duration_path_async(source, timeout_seconds=self.settings.ffprobe_timeout_seconds, cancel_event=context.cancel_event)
            if duration is not None and duration > 21600:
                raise AsrError(422, "duration_limit_exceeded", "workbench accepts at most six hours of audio")
            pcm_budget = int((duration if duration is not None else 21600) * definition.audio_spec.bytes_per_second) + 4096
            margin = self.processing_margin(request, duration if duration is not None else 21600)
            # One complete normalized asset plus one bounded worker slice.
            await self.reserve_processing(job, pcm_budget + margin)
            temp_audio = root / "audio.partial.wav"
            lease = await self.app.state.workspace_manager.handoff_persistent(root, "job:" + job_id, pcm_budget)
            try:
                normalized = await normalize_audio_path_to_wav_async(source, temp_audio, timeout_seconds=self.settings.ffmpeg_timeout_seconds,
                    max_output_bytes=pcm_budget, max_duration_seconds=21600, spec=definition.audio_spec,
                    cancel_event=context.cancel_event, owner=job_id, resource_lease=lease)
                await lease.release_unused()
                asset = normalized.asset
                if asset is None:
                    asset = await self.store.run(NormalizedAudioAsset.from_path, temp_audio, spec=definition.audio_spec, owner=job_id)
                await self.store.run(temp_audio.replace, audio)
                asset = asset.relocated(audio)
                async with self.lock:
                    current = await self.store.run(self.store.job_metadata, job_id)
                    current["normalized_asset"] = asset.to_metadata()
                    await self.store.run(self.store.save_job, current)
                await self.store.run(source.unlink, missing_ok=True)
                await self.record_files(job_id, ["source", "audio.wav"])
                await self.settle_processing_reservation(job_id, margin)
            finally:
                await lease.release()
        else:
            if job.get("normalized_asset") is None:
                if await self.store.run((root / "checkpoints").exists):
                    raise AsrError(409, "resume_version_unavailable", "saved audio predates normalized asset identity; submit a new task to retain the old checkpoints")
                asset = await self.store.run(NormalizedAudioAsset.from_path, audio, spec=definition.audio_spec, owner=job_id)
                async with self.lock:
                    current = await self.store.run(self.store.job_metadata, job_id)
                    current["normalized_asset"] = asset.to_metadata()
                    await self.store.run(self.store.save_job, current)
            else:
                asset = await self.store.run(NormalizedAudioAsset.from_metadata, audio, job["normalized_asset"], spec=definition.audio_spec)
            margin = self.processing_margin(request, asset.duration)
            await self.reserve_processing(job, margin)
            await self.settle_processing_reservation(job_id, margin)
            emit("media.reused", output_bytes=asset.size_bytes)
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
            if starting.get("delete_requested") or starting.get("cancel_requested") or starting["status"] == "cancel_requested":
                return
            self.active.add(job_id)
        root = self.job_dir(job_id)
        workspace = root / "work"
        try:
            async with self.lock:
                self.check_running(job_id)
                job = await self.store.run(self.store.job_metadata, job_id)
                request = TranscriptionRequest(**job["request"])
                definition = self.app.state.manager.runtime_for(request.model or self.app.state.manager.default_model_id).definition
                if definition.revision != job["model_revision"] or (job.get("model_identity") is not None and await self.store.run(self.execution_identity, request) != job["model_identity"]):
                    raise AsrError(409, "resume_version_unavailable", "saved model revision is unavailable; retry as a new task")
                job["started_at"] = job["started_at"] or now()
                job["status"] = "preprocessing"
                job["progress"] = {"phase": "preprocessing"}
                await self.store.run(self.store.save_job, job)
            prepared_error = self.preparation_errors.pop(job_id, None)
            if prepared_error is not None:
                raise prepared_error
            asset = await self.prepare_audio(job_id)
            self.check_running(job_id)
            workspace.mkdir(exist_ok=True)
            async def stage(phase: str, progress: dict[str, object]) -> None:
                self.check_running(job_id)
                self.progress[job_id] = {**self.progress.get(job_id, {}), **progress, "phase": phase}
                moment = time.monotonic()
                immediate = "completed_chunks" in progress or moment - self.progress_saved.get(job_id, 0) >= 1
                if not immediate:
                    return
                async with self.lock:
                    self.check_running(job_id)
                    current = await self.store.run(self.store.job_metadata, job_id)
                    current["status"] = phase
                    current["progress"] = {**current["progress"], **self.progress[job_id]}
                    await self.store.run(self.store.save_job, current)
                    self.progress_saved[job_id] = moment
            async def before_chunk(index: int, total: int) -> None:
                await stage("transcribing", {"current_chunk": index + 1, "total_chunks": total})
            async def after_chunk(index: int, total: int, result: TranscriptionResult) -> None:
                await self.record_files(job_id, [f"checkpoints/chunk-{index:05d}.json", "checkpoints/identity.json", "checkpoints/review-budget.json"])
                async with self.lock:
                    self.check_running(job_id)
                    await self.store.run(self.store.save_chunk, job_id, {"index": index, "text": result.text}, total)
                    self.progress_saved[job_id] = time.monotonic()
            result = await run_transcription_path(
                asset.path, workspace=workspace, manager=self.app.state.manager, settings=self.settings,
                request=request, stage_callback=stage, before_chunk=before_chunk, after_chunk=after_chunk,
                checkpoint_dir=root / "checkpoints",
                normalized_asset=asset, cancel_event=self.context_for(job).cancel_event,
            )
            self.check_running(job_id)
            self.prepare_result(result)
            await asyncio.to_thread(shutil.rmtree, workspace, True)
            async with self.lock:
                self.check_running(job_id)
                job = await self.store.run(self.store.job_metadata, job_id)
                job["result"] = result
                job["error"] = None
                job["progress"] = {"phase": "completed", "percent": 100}
                self.terminal(job, "completed")
                await self.store.run(self.store.save_version, job)
            emit("job.completed")
        except WorkbenchInterrupted:
            job = (await self.store.run(self.store.job_metadata, job_id))
            if self.cancelled(job_id):
                await self.finish_cancel(job)
            elif self.stopping:
                job["status"] = "queued"
                job["progress"]["phase"] = "queued"
                (await self.store.run(self.store.save_job, job))
            else:
                await self.finish_cancel(job)
        except asyncio.CancelledError:
            job = (await self.store.run(self.store.job_metadata, job_id))
            if self.cancelled(job_id):
                await self.finish_cancel(job)
            elif self.stopping:
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
            elif self.stopping:
                job["status"] = "queued"
                job["progress"]["phase"] = "queued"
                (await self.store.run(self.store.save_job, job))
            else:
                await asyncio.to_thread(shutil.rmtree, workspace, True)
                await self.store.run((root / "audio.partial.wav").unlink, missing_ok=True)
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
                        current["error"] = {"code": error.code, "message": error.message, "details": error.details}
                        self.terminal(current, "failed")
                        current["progress"]["phase"] = "failed"
                        (await self.store.run(self.store.save_job, current))
                        emit("job.failed", error_code=error.code, diagnostics=safe_exception(exc))
                if finish_cancel:
                    await self.finish_cancel(current)
        finally:
            await asyncio.to_thread(shutil.rmtree, workspace, True)
            self.active.discard(job_id)
            try:
                async with self.lock:
                    job = (await self.store.run(self.store.job_metadata, job_id))
                    if job.get("delete_requested"):
                        await self.delete_now(job)
                    elif root.exists():
                        reserved = sum(job.get("retained_file_bytes", {}).values())
                        (await self.store.run(self.store.save_job, job, reserved_bytes=reserved))
                        await self.sync_disk_reservations()
            except AsrError:
                pass
            self.progress.pop(job_id, None)
            self.progress_saved.pop(job_id, None)

    def prepare_result(self, result: dict[str, Any]) -> None:
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
        result["changes"] = result.get("quality", {}).get("changes", [])
        result["versions"] = [{"version": 1, "created_at": now(), "alignment_status": result["alignment_status"]}]

    async def finish_cancel(self, job: dict[str, Any]) -> None:
        await asyncio.to_thread(shutil.rmtree, self.job_dir(job["id"]) / "work", True)
        await self.store.run((self.job_dir(job["id"]) / "audio.partial.wav").unlink, missing_ok=True)
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
            (await self.store.run(self.store.save_job, current))
            emit("job.cancelled")

    async def cancel(self, owner: str, job_id: str) -> dict[str, Any]:
        async with self.lock:
            job = (await self.store.run(self.store.job, job_id, owner))
            if job["status"] in TERMINAL:
                return self.public_job(job)
            job["status"] = "cancel_requested"
            job["cancel_requested"] = True
            self.cancel_intents.add(job_id)
            self.context_for(job).cancel_event.set()
            (await self.store.run(self.store.save_job, job))
            removed = await self.app.state.job_manager.cancel_external(job_id)
            finish_now = removed or job_id not in self.active
        if finish_now:
            await self.finish_cancel(job)
        return self.public_job((await self.store.run(self.store.job, job_id, owner)))

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
            if job_id in self.active:
                return False
            await self.delete_now(job)
            return True

    async def delete_now(self, job: dict[str, Any]) -> None:
        await asyncio.to_thread(shutil.rmtree, self.job_dir(job["id"]), True)
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

    async def retry(self, owner: str, job_id: str) -> dict[str, Any]:
        async with self.lock:
            job = (await self.store.run(self.store.job, job_id, owner))
            if job["status"] not in {"failed", "cancelled"}:
                raise AsrError(409, "job_not_retryable", "only failed or cancelled tasks can be retried")
            if not (self.job_dir(job_id) / "audio.wav").exists() and not (self.job_dir(job_id) / "source").exists():
                raise AsrError(409, "source_unavailable", "original audio is unavailable")
            job["status"] = "queued"
            job["cancel_requested"] = False
            job["error"] = None
            job["completed_at"] = None
            job["expires_at"] = None
            job["progress"] = {"phase": "queued", "percent": 0}
            previous = self.context_for(job)
            context = previous.with_fields(attempt=previous.attempt + 1, cancel_event=threading.Event())
            self.contexts[job_id] = context
            self.cancel_intents.discard(job_id)
            job["execution_context"] = context.to_wire()
            (await self.store.run(self.store.save_job, job))
            await self.enqueue(job_id)
            return self.public_job(job)

    async def edit(self, owner: str, job_id: str, segment_id: str, text: str, base_version: int) -> dict[str, Any]:
        async with self.lock:
            job = (await self.store.run(self.store.job, job_id, owner))
            result = job.get("result")
            if job["status"] != "completed" or result is None:
                raise AsrError(409, "job_not_completed", "only completed transcripts can be edited")
            definition = self.app.state.manager.runtime_for(job["request"]["model"]).definition
            if not definition.capabilities.forced_alignment:
                raise AsrError(422, "capability_not_supported", "selected model does not support transcript edit alignment")
            if result["version"] != base_version:
                raise AsrError(409, "version_conflict", "transcript changed; refresh before editing")
            if result["alignment_status"] == "pending":
                raise AsrError(409, "alignment_pending", "wait for the current edit to finish alignment")
            segment = next((s for s in result["segments"] if s["id"] == segment_id), None)
            if segment is None:
                raise AsrError(404, "segment_not_found", "segment not found")
            if len(text) > 20000:
                raise AsrError(400, "bad_request", "edited segment is too long")
            result = copy.deepcopy(result)
            target = next(s for s in result["segments"] if s["id"] == segment_id)
            previous_text = target["text"]
            target["text"] = text.strip()
            target["timestamps"] = []
            target["alignment_status"] = "pending"
            result["version"] += 1
            result["alignment_status"] = "pending"
            result["pending_segment_id"] = segment_id
            result["text"] = "\n\n".join(s["text"] for s in result["segments"] if s["text"])
            # Never publish the old timeline alongside new text.
            result["timestamps"] = []
            result.setdefault("changes", []).append({"method": "user_edit", "segment_id": segment_id,
                "before": previous_text, "after": text.strip(), "created_at": now(), "version": result["version"]})
            result["versions"].append({"version": result["version"], "created_at": now(), "alignment_status": "pending"})
            await self.store.run(self.check_capacity, owner, len(json.dumps(result, ensure_ascii=False).encode("utf-8")))
            job["result"] = result
            (await self.store.run(self.store.save_version, job))
            try:
                await self.enqueue_alignment(job_id)
            except AsrError as exc:
                if exc.code != "job_queue_full":
                    raise
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
            segment_id = result.get("pending_segment_id")
            if result["alignment_status"] != "pending" or not segment_id:
                return
            target = next(s for s in result["segments"] if s["id"] == segment_id)
            audio = AudioPath(self.job_dir(job_id) / "audio.wav", target["start"], target["end"])
            items = await self.app.state.manager.align_text(
                audio, text=target["text"], model_id=job["request"]["model"],
                backend="auto", language=job["request"]["language"], timestamps="auto",
            ) if target["text"] else []
            async with self.lock:
                latest = (await self.store.run(self.store.job, job_id))
                if latest.get("delete_requested") or latest["status"] == "cancel_requested":
                    return
                if latest["result"]["version"] != result["version"]:
                    return
                job = latest
                job["result"] = result
                target["timestamps"] = [{**item.to_api(), "start": item.start + target["start"], "end": item.end + target["start"]} for item in items]
                target["alignment_status"] = "aligned"
                target.pop("reading_segmentation", None)
                result["timestamps"] = [item for s in result["segments"] for item in s.get("timestamps", [])]
                result["alignment_status"] = "aligned" if all(
                    not s["text"] or (s.get("timestamps") and s.get("reading_segmentation") != "unmapped_alignment")
                    for s in result["segments"]
                ) else "incomplete"
                result.pop("pending_segment_id", None)
                result["versions"][-1]["alignment_status"] = result["alignment_status"]
                (await self.store.run(self.store.save_version, job))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            job = (await self.store.run(self.store.job, job_id))
            if self.stopping or job.get("delete_requested") or job.get("cancel_requested"):
                return  # Keep pending alignment for recovery, or honor deletion.
            if job.get("result"):
                result = job["result"]
                result["alignment_status"] = "failed"
                result["alignment_error"] = {"code": exc.code if isinstance(exc, AsrError) else "alignment_failed", "message": str(exc) if isinstance(exc, AsrError) else "alignment failed"}
                result["versions"][-1]["alignment_status"] = "failed"
                (await self.store.run(self.store.save_version, job))
        finally:
            self.active.discard(job_id)
            try:
                job = (await self.store.run(self.store.job, job_id))
                if job.get("delete_requested"):
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
            if row["id"] in self.active:
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
            self.store.set_physical_usage(row["id"], row["owner_id"], sum(files.values()))
            with self.store.connection() as db:
                db.execute("UPDATE jobs SET payload=json_set(payload,'$.retained_file_bytes',json(?)) WHERE id=?", (json.dumps(files), row["id"]))
        with self.store.connection() as db:
            stale = db.execute("SELECT asset_id,owner_id FROM assets").fetchall()
        for asset in stale:
            if asset["asset_id"] not in live_assets:
                self.store.set_physical_usage(asset["asset_id"], asset["owner_id"], 0)

    async def cleanup(self) -> None:
        async with self.lock:
            expired_ids = await self.transaction(lambda db: db.execute("SELECT id FROM jobs WHERE status IN ('completed','failed','cancelled') AND json_extract(summary,'$.expires_at')<=?", (now(),)).fetchall())
            for row in expired_ids:
                if row["id"] not in self.active:
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


def export_result(result: dict[str, Any], format_: str) -> tuple[str, str]:
    if format_ == "txt":
        return result["text"] + "\n", "text/plain"
    if format_ == "json":
        return json.dumps(result, ensure_ascii=False, indent=2) + "\n", "application/json"
    if format_ not in {"srt", "vtt"}:
        raise AsrError(400, "bad_request", "unsupported export format")
    if result.get("alignment_status") != "aligned":
        raise AsrError(409, "alignment_pending", "subtitles are unavailable until this transcript version is aligned")
    cues: list[str] = []
    for segment in result["segments"]:
        if not segment["text"].strip():
            continue
        items = segment.get("timestamps", [])
        if not items:
            raise AsrError(409, "alignment_incomplete", "a transcript segment has no real alignment")
        cues.append(f"{len(cues) + 1}\n{timestamp(items[0]['start'], vtt=format_ == 'vtt')} --> {timestamp(items[-1]['end'], vtt=format_ == 'vtt')}\n{segment['text']}")
    text = ("WEBVTT\n\n" if format_ == "vtt" else "") + "\n\n".join(cues) + "\n"
    return text, "text/vtt" if format_ == "vtt" else "application/x-subrip"
