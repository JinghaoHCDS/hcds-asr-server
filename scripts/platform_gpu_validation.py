"""Offline, isolated ASGI validation of installed ASR models; never deploys.

Raw responses may contain transcripts and remain below the caller's output dir.
stdout contains only phase/status summaries. No listener or remote connection is
created; model selection always uses the application's registered definitions.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import secrets
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator
from uuid import uuid4
from unittest.mock import patch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from fastapi import FastAPI

from asr_server.audio.preprocess import probe_audio_duration_path_async
from asr_server.config import Settings
from asr_server.main import create_app
from asr_server.registry import MOSS_MODEL_ID
from asr_server.quality.checkpoints import Checkpoints
from asr_server import observability


TERMINAL = {"completed", "failed", "cancelled", "expired"}


class ValidationFailure(Exception):
    def __init__(self, code: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code, self.details = code, details or {}


def require(condition: object, code: str, **details: Any) -> None:
    if not condition:
        raise ValidationFailure(code, details)


def digest(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hashed.update(block)
    return hashed.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def runtime_snapshot(app: FastAPI) -> dict[str, Any]:
    manager = app.state.manager
    models = []
    for item in manager.list_models():
        runtime = manager.runtime_for(item["id"])
        transport = getattr(runtime.adapter, "_transport", None)
        models.append({"id": item["id"], "engine_id": runtime.definition.engine_id,
                       "status": runtime.status, "active_requests": runtime.active_requests,
                       "worker_pid": transport.pid if transport is not None else None,
                       "worker_alive": transport.is_alive if transport is not None else False})
    return {"models": models, "reserved_temporary_bytes": app.state.workspace_manager._reserved_upload_bytes,
            "scheduled_operations": len(app.state.job_manager.scheduler.operations),
            "gpu_lock_held": manager._gpu_operation_lock.locked(),
            "gpu_snapshot": app.state.gpu_snapshot, "gpu_snapshot_at": app.state.gpu_snapshot_at,
            "owned_child_pids": [child.pid for child in multiprocessing.active_children()]}


class Harness:
    def __init__(self, args: argparse.Namespace, directory: Path) -> None:
        self.args, self.directory = args, directory
        self.password = secrets.token_urlsafe(24)
        self.other_password = secrets.token_urlsafe(24)
        self.sequence = 0
        self.phase = "setup"
        self.volatile_job: str | None = None
        self.report: dict[str, Any] = {
            "schema_version": 1, "mode": "mock" if args.mock else "installed-offline-real-adapters",
            "run_id": directory.name, "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "independent ASGI app/data; no listening socket, online deployment or installation",
            "quality_claim": "runtime behavior only; no human reference accuracy or six-hour quality acceptance",
            "restart_scope": "two app lifespans and new model workers; no forced interruption of active CUDA",
            "phases": [], "snapshots": [],
        }

    def settings(self) -> Settings:
        return Settings(
            adapter="mock" if self.args.mock else "registered", api_token=None,
            enable_qwen_alignment=True, enable_moss=self.args.enable_moss,
            workbench_data_dir=self.directory / "workbench", log_dir=self.directory / "logs",
            idle_unload_seconds=0, min_free_disk_mb=self.args.min_free_disk_mb,
            worker_load_timeout_seconds=self.args.operation_timeout,
            worker_inference_timeout_seconds=self.args.operation_timeout,
            worker_shutdown_timeout_seconds=min(30, self.args.operation_timeout),
            job_shutdown_grace_seconds=min(60, self.args.operation_timeout),
        )

    @asynccontextmanager
    async def application(self, *, initialize: bool) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
        app = create_app(self.settings())
        if initialize:
            app.state.workbench.store.create_user("validation-owner", self.password, True)
            app.state.workbench.store.create_user("validation-other", self.other_password)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://validation.local",
                                         timeout=httpx.Timeout(self.args.operation_timeout)) as client:
                await self.request(client, "POST", "/api/auth/login", expected=200,
                                   json={"username": "validation-owner", "password": self.password})
                yield app, client
        self.report["snapshots"].append({"phase": self.phase, "point": "after_app_shutdown", **runtime_snapshot(app)})

    async def request(self, client: httpx.AsyncClient, method: str, path: str, *, expected: int | tuple[int, ...],
                      label: str = "http", **kwargs: Any) -> httpx.Response:
        self.sequence += 1
        sequence = self.sequence
        started = perf_counter()
        response = await asyncio.wait_for(client.request(method, path, **kwargs), self.args.operation_timeout)
        content: Any
        if "json" in response.headers.get("content-type", ""):
            try:
                content = response.json()
            except ValueError:
                content = {"invalid_json": True}
        else:
            content = {"bytes": len(response.content), "sha256": hashlib.sha256(response.content).hexdigest()}
        # Never persist login cookies, authorization headers or submitted passwords.
        evidence = {"method": method, "path": path, "status_code": response.status_code,
                    "elapsed_seconds": perf_counter() - started, "body": content,
                    "headers": {key: value for key, value in response.headers.items()
                                if key.lower() in {"x-request-id", "etag", "content-type", "content-range", "x-transcript-version"}}}
        write_json(self.directory / "raw" / f"{sequence:05d}-{self.phase}-{label}.json", evidence)
        acceptable = (expected,) if isinstance(expected, int) else expected
        require(response.status_code in acceptable, "unexpected_http_status", method=method, path=path,
                expected=list(acceptable), observed=response.status_code, response=content)
        return response

    async def poll_job(self, client: httpx.AsyncClient, path: str, *, alignment: bool = False) -> tuple[dict[str, Any], list[str]]:
        started = perf_counter()
        states: list[str] = []
        while perf_counter() - started < self.args.operation_timeout:
            response = await client.get(path)
            require(response.status_code == 200, "poll_failed", status=response.status_code)
            job: dict[str, Any] = response.json()
            state = str((job.get("result") or {}).get("alignment_status")) if alignment else str(job["status"])
            if not states or states[-1] != state:
                states.append(state)
                write_json(self.directory / "raw" / f"poll-{self.phase}-{job['id']}-{len(states):02d}.json", job)
            if alignment:
                if state != "pending":
                    require(state == "aligned", "edit_alignment_failed", result=job.get("result"))
                    return job, states
            elif state in TERMINAL:
                require(state == "completed", "job_failed", job=job)
                full_path = path + ("&detail=full" if "?" in path else "?detail=full") if path.startswith("/v1/") else path
                job = (await self.request(client, "GET", full_path, expected=200, label="completed")).json()
                return job, states
            await asyncio.sleep(0.02 if self.args.mock else 0.2)
        raise ValidationFailure("job_deadline_exceeded", {"path": path, "states": states})

    def model(self, app: FastAPI, preferred: str, *, engine: str | None = None, profile: str = "legacy") -> str:
        selected = self.args.model or preferred
        available = {item["id"] for item in app.state.manager.list_models()}
        require(selected in available, "model_not_enabled", model=selected, enabled=sorted(available))
        definition = app.state.manager.runtime_for(selected).definition
        require(engine is None or definition.engine_id == engine, "unexpected_engine", model=selected, expected_engine=engine)
        require(definition.supports_pipeline(profile), "unsupported_validation_profile", model=selected, profile=profile)
        return str(selected)

    async def unload(self, app: FastAPI, client: httpx.AsyncClient, model: str) -> dict[str, Any]:
        runtime = app.state.manager.runtime_for(model)
        require(runtime.active_requests == 0, "unload_requested_while_active", model=model)
        transport = getattr(runtime.adapter, "_transport", None)
        before_pid = transport.pid if transport is not None else None
        before = runtime_snapshot(app)
        require(sum(item["status"] == "loaded" for item in before["models"]) <= 1, "multiple_models_loaded")
        if not self.args.mock:
            require(transport is not None and transport.is_alive and before_pid is not None, "missing_real_model_worker", model=model)
        response = await self.request(client, "DELETE", "/v1/models/" + model, expected=200, label="unload")
        require(runtime.status == "unloaded", "model_not_unloaded", model=model, status=runtime.status)
        require(transport is None or not transport.is_alive, "worker_not_reaped", model=model, pid=before_pid)
        result = {"model": model, "before": before, "before_pid": before_pid, "after_pid": transport.pid if transport is not None else None,
                  "worker_reaped": transport is None or not transport.is_alive, "mock_has_no_worker": self.args.mock,
                  "response": response.json()}
        self.report["snapshots"].append({"phase": self.phase, "point": "after_unload", **runtime_snapshot(app)})
        return result

    async def legacy(self, app: FastAPI, client: httpx.AsyncClient) -> dict[str, Any]:
        model = self.model(app, "qwen3-asr-0.6b", engine="qwen")
        with self.args.dev_audio.open("rb") as source:
            sync = await self.request(client, "POST", "/v1/audio/transcriptions", expected=200, label="sync",
                                      files={"file": (self.args.dev_audio.name, source)},
                                      data={"model": model, "language": "zh", "response_format": "verbose_json"})
        require(sync.json().get("text", "").strip(), "empty_sync_text")
        with self.args.dev_audio.open("rb") as source:
            created = await self.request(client, "POST", "/v1/audio/transcription-jobs", expected=202, label="async",
                                         files={"file": (self.args.dev_audio.name, source)}, data={"model": model, "language": "zh"})
        self.volatile_job = created.json()["id"]
        job, states = await self.poll_job(client, "/v1/jobs/" + self.volatile_job)
        require((job.get("result") or {}).get("text", "").strip(), "empty_async_text")
        return {"model": model, "language": "zh", "sync_chars": len(sync.json()["text"]),
                "async_chars": len(job["result"]["text"]), "states": states,
                "unload": await self.unload(app, client, model)}

    async def upload(self, client: httpx.AsyncClient, source: Path, *, model: str,
                     profile: str, key: str) -> str:
        size = source.stat().st_size
        created = await self.request(client, "POST", "/api/uploads", expected=201, label="upload-create",
                                     json={"filename": source.name, "size_bytes": size})
        upload_id = created.json()["id"]
        offset = 0
        with source.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                response = await self.request(client, "PATCH", "/api/uploads/" + upload_id, expected=200, label="upload-chunk",
                                              content=block, headers={"Content-Type": "application/octet-stream", "Upload-Offset": str(offset)})
                if offset == 0:
                    replay = await self.request(client, "PATCH", "/api/uploads/" + upload_id, expected=200, label="upload-replay",
                                                content=block, headers={"Content-Type": "application/octet-stream", "Upload-Offset": "0"})
                    require(replay.json()["offset"] == response.json()["offset"], "upload_replay_changed_offset")
                offset += len(block)
                require(response.json()["offset"] == offset, "upload_offset_mismatch")
        completed = await self.request(client, "POST", f"/api/uploads/{upload_id}/complete", expected=202, label="upload-complete",
                                       json={"domain": "ai", "model": model, "quality_profile": profile,
                                             "language": "auto", "idempotency_key": key})
        replay = await self.request(client, "POST", f"/api/uploads/{upload_id}/complete", expected=202, label="submit-replay",
                                    json={"domain": "ai", "model": model, "quality_profile": profile,
                                          "language": "auto", "idempotency_key": key})
        require(replay.json()["id"] == completed.json()["id"], "submission_not_idempotent")
        return str(completed.json()["id"])

    async def workbench(self, app: FastAPI, client: httpx.AsyncClient) -> dict[str, Any]:
        model = self.model(app, "qwen3-asr-1.7b", engine="qwen", profile="quality_v1")
        job_id = await self.upload(client, self.args.acceptance_audio, model=model, profile="quality_v1", key="quality-acceptance")
        job, states = await self.poll_job(client, "/api/jobs/" + job_id)
        result = job["result"]
        require(result.get("text", "").strip(), "empty_workbench_text")
        sources = sorted({str(item.get("source")) for item in result.get("timestamps", [])})
        if not self.args.mock:
            require(sources, "missing_timestamp_source")
            require(sources == ["qwen3_forced_aligner"], "unexpected_timestamp_source", sources=sources)
        response = await self.request(client, "GET", "/api/jobs/" + job_id, expected=200, label="etag")
        etag = response.headers.get("etag")
        require(etag, "missing_etag")
        await self.request(client, "GET", "/api/jobs/" + job_id, expected=304, label="etag-unchanged", headers={"If-None-Match": etag})
        ranged = await self.request(client, "GET", f"/api/jobs/{job_id}/audio", expected=206, label="range", headers={"Range": "bytes=0-31"})
        require(len(ranged.content) == 32, "incorrect_range_length")
        exports: dict[str, int] = {}
        for format_ in ("txt", "json", "srt", "vtt"):
            expected = 409 if self.args.mock and format_ in {"srt", "vtt"} and not sources else 200
            exported = await self.request(client, "GET", f"/api/jobs/{job_id}/export?format={format_}&version=1", expected=expected, label="export-" + format_)
            target = self.directory / "exports" / ("quality-v1." + format_)
            target.parent.mkdir(exist_ok=True, mode=0o700)
            target.write_bytes(exported.content)
            target.chmod(0o600)
            exports[format_] = exported.status_code
        segment = next((item for item in result["segments"] if item.get("text", "").strip()), None)
        require(segment is not None, "missing_editable_segment")
        assert segment is not None
        edited = await self.request(client, "PATCH", f"/api/jobs/{job_id}/segments/{segment['id']}", expected=200, label="edit",
                                    json={"text": segment["text"] + "。", "base_version": 1})
        require(edited.json()["result"]["version"] == 2, "edit_version_not_incremented")
        final, edit_states = await self.poll_job(client, "/api/jobs/" + job_id, alignment=True)
        edited_sources = sorted({str(item.get("source")) for item in final["result"].get("timestamps", [])})
        if not self.args.mock:
            require(edited_sources == ["qwen3_forced_aligner"], "unexpected_edit_timestamp_source", sources=edited_sources)
        updated = await self.request(client, "GET", "/api/jobs/" + job_id, expected=200, label="etag-changed", headers={"If-None-Match": etag})
        require(updated.headers.get("etag") != etag, "etag_did_not_change_after_edit")
        await self.request(client, "GET", f"/api/jobs/{job_id}/export?format=srt&version=2", expected=200, label="edited-srt")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://validation.local") as other:
            await self.request(other, "GET", "/api/auth/me", expected=401, label="bearer-isolation", headers={"Authorization": "Bearer validation-not-a-workbench-session"})
            await self.request(other, "POST", "/api/auth/login", expected=200, label="other-login",
                               json={"username": "validation-other", "password": self.other_password})
            for suffix in ("", "/audio", "/export?format=json"):
                await self.request(other, "GET", f"/api/jobs/{job_id}{suffix}", expected=404, label="owner-isolation")
            await self.request(other, "DELETE", "/api/jobs/" + job_id, expected=404, label="owner-delete-isolation")
        await self.request(client, "GET", "/v1/jobs/" + job_id, expected=404, label="legacy-isolation")
        root = app.state.workbench.job_dir(job_id)
        checkpoint_count = len(list((root / "checkpoints").glob("chunk-*.json")))
        await self.request(client, "DELETE", "/api/jobs/" + job_id, expected=204, label="delete")
        await self.request(client, "GET", "/api/jobs/" + job_id, expected=404, label="deleted-missing")
        require(not root.exists(), "deleted_assets_remain")
        return {"model": model, "states": states, "text_chars": len(result["text"]), "timestamp_sources": sources,
                "timestamp_count": len(result.get("timestamps", [])), "checkpoint_count": checkpoint_count,
                "exports": exports, "edit_states": edit_states, "edited_version": final["result"]["version"],
                "edited_timestamp_sources": edited_sources,
                "edited_alignment": final["result"]["alignment_status"], "owner_isolation": True, "deleted": True,
                "unload": await self.unload(app, client, model)}

    async def moss(self, app: FastAPI, client: httpx.AsyncClient) -> dict[str, Any]:
        model = self.model(app, MOSS_MODEL_ID, engine="moss")
        parameters = {"model": model, "language": "auto", "response_format": "verbose_json", "preserve_segments": "true"}
        if self.args.moss_max_new_tokens is not None:
            parameters["max_new_tokens"] = str(self.args.moss_max_new_tokens)
        with self.args.acceptance_audio.open("rb") as source:
            created = await self.request(client, "POST", "/v1/audio/transcription-jobs", expected=202, label="moss-create",
                                         files={"file": (self.args.acceptance_audio.name, source)},
                                         data=parameters)
        job, states = await self.poll_job(client, "/v1/jobs/" + created.json()["id"])
        result = job["result"]
        require(result.get("text", "").strip() and result.get("segments"), "missing_moss_text_or_segments")
        scope = result.get("execution", {}).get("speaker_scope")
        require(scope in {"global", "chunk", "mixed"}, "missing_speaker_scope", execution=result.get("execution"))
        require(any(segment.get("speaker") for segment in result["segments"]), "missing_moss_speaker_labels")
        return {"model": model, "states": states, "text_chars": len(result["text"]), "segments": len(result["segments"]),
                "speaker_scope": scope, "execution": result.get("execution"), "generation": result.get("generation"),
                "requested_max_new_tokens": self.args.moss_max_new_tokens,
                "unload": await self.unload(app, client, model)}

    async def lifecycle(self, app: FastAPI, client: httpx.AsyncClient) -> dict[str, Any]:
        available = {item["id"] for item in app.state.manager.list_models()}
        models = ([self.model(app, "qwen3-asr-0.6b")] if self.args.model else
                  [model for model in ("qwen3-asr-0.6b", "qwen3-asr-1.7b", MOSS_MODEL_ID) if model in available])
        observations = []
        previous: str | None = None
        previous_pid: int | None = None
        for model in models:
            await self.request(client, "POST", f"/v1/models/{model}/load", expected=200, label="load", json={"backend": "transformers"})
            snapshot = runtime_snapshot(app)
            require(sum(item["status"] == "loaded" for item in snapshot["models"]) == 1, "multiple_loaded_models")
            require(not snapshot["gpu_lock_held"] and all(item["active_requests"] == 0 for item in snapshot["models"]),
                    "load_returned_with_active_gpu_ownership")
            current = next(item for item in snapshot["models"] if item["id"] == model)
            require(current["status"] == "loaded", "requested_model_not_loaded", model=model)
            if not self.args.mock:
                require(current["worker_alive"] and current["worker_pid"] is not None, "loaded_worker_missing", model=model)
            evicted = None
            if previous is not None:
                old = next(item for item in snapshot["models"] if item["id"] == previous)
                require(old["status"] == "unloaded" and not old["worker_alive"] and old["worker_pid"] is None,
                        "automatic_switch_did_not_reap_previous_worker", previous_model=previous, snapshot=old)
                evicted = {"model": previous, "before_pid": previous_pid, "after_pid": old["worker_pid"],
                           "status": old["status"], "worker_reaped": not old["worker_alive"]}
            observations.append({"model": model, "loaded": snapshot, "automatically_evicted": evicted})
            previous, previous_pid = model, current["worker_pid"]
        require(previous is not None, "no_enabled_lifecycle_models")
        assert previous is not None
        return {"models": observations, "automatic_switch_count": max(0, len(models) - 1),
                "intermediate_manual_unloads": 0, "final_unload": await self.unload(app, client, previous)}

    async def prepare_restart(self, app: FastAPI, client: httpx.AsyncClient) -> dict[str, Any]:
        model = self.model(app, "qwen3-asr-1.7b", engine="qwen")
        entered, release = asyncio.Event(), asyncio.Event()

        async def barrier() -> None:
            entered.set()
            await release.wait()

        operation = await app.state.job_manager.scheduler.submit("validation-restart-barrier", barrier, kind="validation-control")
        await asyncio.wait_for(entered.wait(), 5)
        try:
            job_id = await self.upload(client, self.args.dev_audio, model=model, profile="legacy", key="restart-durable")
            before = (await self.request(client, "GET", "/api/jobs/" + job_id, expected=200, label="restart-queued")).json()
            require(before["status"] == "queued", "restart_fixture_not_queued")
            with self.args.dev_audio.open("rb") as source:
                memory = await self.request(client, "POST", "/v1/audio/transcription-jobs", expected=202, label="restart-volatile",
                                            files={"file": (self.args.dev_audio.name, source)}, data={"model": model})
            memory_id = memory.json()["id"]
            await self.request(client, "DELETE", "/v1/jobs/" + memory_id, expected=200, label="stop-volatile")
            # Match graceful queue disposal while preserving the durable owner's
            # queued intent. No model call is interrupted or worker force-killed.
            await app.state.workbench.shutdown()
            require(await app.state.job_manager.cancel_external(job_id), "restart_queue_not_removed")
            return {"model": model, "durable_job_id": job_id, "volatile_job_id": memory_id,
                    "fixture": "graceful stop with durable queued work", "checkpoint_resume_tested": False}
        finally:
            release.set()
            await asyncio.wait_for(asyncio.shield(operation.future), 5)

    async def prepare_inflight_restart(self, app: FastAPI, client: httpx.AsyncClient) -> dict[str, Any]:
        """Stop immediately after a real, unmodified checkpoint atomic commit.

        A local harness hook chooses a deterministic safe boundary; it changes
        neither checkpoint contents nor inference. The owning service's normal
        shutdown/cancel signal handles the rest. No CUDA call is interrupted.
        """
        model = self.model(app, "qwen3-asr-1.7b", engine="qwen", profile="quality_v1")
        ready = asyncio.Event()
        once = threading.Event()
        loop = asyncio.get_running_loop()
        original_save = Checkpoints.save

        def save_checkpoint(owner: Checkpoints, index: int, result: Any, reviews: Any, extra_seconds: float) -> None:
            original_save(owner, index, result, reviews, extra_seconds)
            if owner.directory is not None and not once.is_set():
                once.set()
                stopped = asyncio.run_coroutine_threadsafe(app.state.workbench.shutdown(), loop)
                stopped.result(timeout=min(60, self.args.operation_timeout))
                loop.call_soon_threadsafe(ready.set)

        with patch.object(Checkpoints, "save", save_checkpoint):
            job_id = await self.upload(client, self.args.acceptance_audio, model=model, profile="quality_v1", key="restart-inflight")
            operation = app.state.job_manager.scheduler.operations.get(job_id)
            await asyncio.wait_for(ready.wait(), self.args.operation_timeout)
            if operation is not None:
                await asyncio.wait_for(asyncio.shield(operation.future), self.args.operation_timeout)
        job = (await self.request(client, "GET", "/api/jobs/" + job_id, expected=200, label="safe-stopped")).json()
        require(job["status"] == "queued", "inflight_job_did_not_preserve_queued_intent", status=job["status"])
        total = job.get("progress", {}).get("total_chunks", 0)
        root = app.state.workbench.job_dir(job_id)
        checkpoints = sorted((root / "checkpoints").glob("chunk-*.json"))
        require(checkpoints and int(total) > len(checkpoints), "restart_requires_remaining_chunks", total=total, saved=len(checkpoints))
        saved = {path.name: {"sha256": digest(path), "text": json.loads(path.read_text())["result"]["text"]} for path in checkpoints}
        asset = root / "audio.wav"
        stat = asset.stat()
        before_events = await self.events_for(job_id)
        return {"model": model, "durable_job_id": job_id, "volatile_job_id": None,
                "fixture": "normal service shutdown after checkpoint commit; local deterministic boundary hook",
                "checkpoint_resume_tested": True, "planned_chunks": total, "saved_checkpoints": saved,
                "asset_before": {"sha256": digest(asset), "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns, "bytes": stat.st_size},
                "events_before": before_events, "unload_before_restart": await self.unload(app, client, model)}

    async def events_for(self, job_id: str) -> dict[str, int]:
        runtime = getattr(observability, "_runtime", None)
        if runtime is not None:
            await asyncio.to_thread(runtime.handler.pending.join)
        counts: dict[str, int] = {}
        for path in sorted((self.directory / "logs").glob("events.jsonl*"))[-10:]:
            with path.open("rb") as stream:
                while line := stream.readline(8193):
                    if not line.endswith(b"\n"):
                        continue
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if event.get("job_id") == job_id:
                        name = str(event.get("event"))
                        counts[name] = counts.get(name, 0) + 1
        return counts

    async def finish_restart(self, app: FastAPI, client: httpx.AsyncClient, prepared: dict[str, Any]) -> dict[str, Any]:
        if prepared.get("volatile_job_id"):
            await self.request(client, "GET", "/v1/jobs/" + prepared["volatile_job_id"], expected=404, label="volatile-not-restored")
        job, states = await self.poll_job(client, "/api/jobs/" + prepared["durable_job_id"])
        require(job["result"].get("text", "").strip(), "restored_job_empty")
        result = {**prepared, "restored_states": states, "restored_text_chars": len(job["result"]["text"]),
                  "snapshot": runtime_snapshot(app), "unload": await self.unload(app, client, prepared["model"])}
        if prepared.get("checkpoint_resume_tested"):
            root = app.state.workbench.job_dir(prepared["durable_job_id"])
            asset = root / "audio.wav"
            stat = asset.stat()
            asset_after = {"sha256": digest(asset), "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns, "bytes": stat.st_size}
            require(asset_after == prepared["asset_before"], "normalized_asset_was_rewritten_on_resume")
            chunks = {int(chunk["index"]): chunk["text"] for chunk in job["result"]["chunks"]}
            for name, saved in prepared["saved_checkpoints"].items():
                require(digest(root / "checkpoints" / name) == saved["sha256"], "saved_checkpoint_was_rewritten", checkpoint=name)
                index = int(name.removeprefix("chunk-").removesuffix(".json"))
                require(chunks.get(index) == saved["text"], "saved_chunk_text_was_lost", chunk=index)
            events = await self.events_for(prepared["durable_job_id"])
            before = prepared["events_before"]
            hits = events.get("quality.checkpoint_hit", 0) - before.get("quality.checkpoint_hit", 0)
            decodes = events.get("media.normalized", 0) - before.get("media.normalized", 0)
            require(hits >= len(prepared["saved_checkpoints"]), "checkpoint_hit_not_observed", hits=hits)
            require(decodes == 0, "audio_was_normalized_again_on_resume", decodes=decodes)
            result.update(asset_after=asset_after, checkpoint_hits_after_restart=hits,
                          normalized_again_after_restart=decodes, saved_chunk_text_preserved=True, events_after=events)
        await self.request(client, "DELETE", "/api/jobs/" + prepared["durable_job_id"], expected=204, label="restart-delete")
        return result

    async def phase_run(self, name: str, operation: Any) -> Any:
        self.phase = name
        started = perf_counter()
        record: dict[str, Any] = {"phase": name, "status": "running"}
        self.report["phases"].append(record)
        write_json(self.directory / "summary.json", self.report)
        print(json.dumps({"phase": name, "status": "started"}), flush=True)
        try:
            async with asyncio.timeout(self.args.phase_timeout):
                result = await operation()
            record.update(status="passed", elapsed_seconds=perf_counter() - started, result=result)
            print(json.dumps({"phase": name, "status": "passed", "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
            return result
        except BaseException as exc:
            record.update(status="failed", elapsed_seconds=perf_counter() - started,
                          error_type=type(exc).__name__, error_code=getattr(exc, "code", "validation_exception"))
            if isinstance(exc, ValidationFailure):
                write_json(self.directory / "raw" / (name + "-failure.json"), {"code": exc.code, "details": exc.details})
            print(json.dumps({"phase": name, "status": "failed", "error_type": type(exc).__name__, "error_code": record["error_code"]}), flush=True)
            raise
        finally:
            write_json(self.directory / "summary.json", self.report)

    async def run(self) -> dict[str, Any]:
        self.report["versions"] = {}
        for package in ("torch", "transformers", "fastapi", "starlette", "moss-transcribe-diarize"):
            try:
                self.report["versions"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self.report["versions"][package] = None
        self.report["audio"] = {}
        for label, path in (("development", self.args.dev_audio), ("acceptance", self.args.acceptance_audio)):
            self.report["audio"][label] = {"name": path.name, "bytes": path.stat().st_size, "sha256": await asyncio.to_thread(digest, path),
                "duration_seconds": await probe_audio_duration_path_async(path, timeout_seconds=30)}
        if not self.args.mock:
            require(self.report["audio"]["development"]["sha256"] != self.report["audio"]["acceptance"]["sha256"], "development_acceptance_audio_identical")
            require(os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1", "offline_environment_required")
        phases = {self.args.phase} if self.args.phase != "all" else {"legacy", "workbench", "moss", "restart"}
        prepared = None
        async with self.application(initialize=True) as (app, client):
            self.report["model_definitions"] = app.state.manager.list_models()
            if "legacy" in phases:
                await self.phase_run("legacy", lambda: self.legacy(app, client))
            if "workbench" in phases:
                await self.phase_run("workbench", lambda: self.workbench(app, client))
            if "moss" in phases:
                if self.args.enable_moss:
                    await self.phase_run("moss", lambda: self.moss(app, client))
                elif self.args.phase == "moss":
                    raise ValidationFailure("moss_not_enabled")
                else:
                    self.report["phases"].append({"phase": "moss", "status": "skipped", "reason": "not explicitly enabled"})
            if "lifecycle" in phases:
                await self.phase_run("lifecycle", lambda: self.lifecycle(app, client))
            if "restart" in phases:
                prepared = await self.phase_run("restart-prepare", lambda: self.prepare_inflight_restart(app, client)
                                                if self.args.restart_inflight else self.prepare_restart(app, client))
        if prepared is not None:
            async with self.application(initialize=False) as (app, client):
                await self.phase_run("restart-restore", lambda: self.finish_restart(app, client, prepared))
        self.report["status"] = "passed"
        write_json(self.directory / "summary.json", self.report)
        return self.report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-audio", type=Path, required=True)
    parser.add_argument("--acceptance-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "legacy", "workbench", "moss", "restart", "lifecycle"), default="all")
    parser.add_argument("--model")
    parser.add_argument("--enable-moss", action="store_true")
    parser.add_argument("--moss-max-new-tokens", type=int, help="explicit API budget; omission preserves the model policy")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--restart-inflight", action="store_true", help="stop after one atomic quality checkpoint, then verify resume")
    parser.add_argument("--operation-timeout", type=float, default=1800)
    parser.add_argument("--phase-timeout", type=float, default=3600)
    parser.add_argument("--min-free-disk-mb", type=int, default=5120)
    args = parser.parse_args()
    for path in (args.dev_audio, args.acceptance_audio):
        if not path.is_file():
            parser.error("both input audio files must exist")
    if any(not math.isfinite(value) or value <= 0 for value in (args.operation_timeout, args.phase_timeout)):
        parser.error("timeouts must be positive finite numbers")
    if args.min_free_disk_mb < 0:
        parser.error("minimum free disk must be nonnegative")
    if args.moss_max_new_tokens is not None and not 1 <= args.moss_max_new_tokens <= 65536:
        parser.error("MOSS token budget must be between 1 and 65536")
    if args.restart_inflight and args.phase not in {"all", "restart"}:
        parser.error("--restart-inflight requires --phase restart or all")
    directory = args.output_dir.resolve() / ("run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])
    directory.mkdir(parents=True, mode=0o700)
    os.environ["ASR_TEMP_DIR"] = str(directory / "temporary")
    harness = Harness(args, directory)
    try:
        report = asyncio.run(harness.run())
    except BaseException as exc:
        harness.report["status"] = "failed"
        harness.report["error_type"] = type(exc).__name__
        harness.report["error_code"] = getattr(exc, "code", "validation_exception")
        write_json(directory / "summary.json", harness.report)
        print(json.dumps({"run_id": directory.name, "status": "failed", "error_type": type(exc).__name__,
                          "error_code": harness.report["error_code"]}), flush=True)
        raise SystemExit(1) from None
    print(json.dumps({"run_id": directory.name, "status": report["status"], "mode": report["mode"]}), flush=True)


if __name__ == "__main__":
    main()
