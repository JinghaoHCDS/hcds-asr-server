from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4
import socket
import importlib
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import Body, FastAPI, File, Form, UploadFile, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from asr_server import __version__
from asr_server.api.schemas import LoadRequest, UnloadRequest
from asr_server.adapters.base import AsrAdapter
from asr_server.audio.preprocess import probe_audio_duration_path_async
from asr_server.audio.workspace import WorkspaceManager
from asr_server.auth import BearerAuthMiddleware, GatewayApp
from asr_server.admission import AdmissionMiddleware, AdmissionRoute, RequestContextMiddleware
from asr_server.observability import configure_logging, current_context, emit, safe_exception
from asr_server.config import Settings, load_settings
from asr_server.errors import AsrError, asr_error_handler, validation_error_handler
from asr_server.jobs import JobManager
from asr_server.lifecycle import ModelLifecycleManager
from asr_server.registry import Backend, ModelDefinition, default_models
from asr_server.engines import EngineRegistration, create_registered_adapter
from asr_server.transcription import (
    SYNC_JOB_THRESHOLD_SECONDS,
    TranscriptionRequest,
    run_transcription_path,
    validate_transcription_request,
)


DOCS_DESCRIPTION_PATH = Path(__file__).resolve().parent.parent / "docs" / "docs-endpoint-capabilities.md"
MCP_TUTORIAL_PATH = Path(__file__).resolve().parent.parent / "docs" / "mcp-bridge.md"


def gpu_health() -> dict[str, object]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return {"available": False, "name": None, "vram_total_mb": None}
    if getattr(torch.version, "cuda", None) is None or not torch.cuda.is_available():
        return {"available": False, "name": None, "vram_total_mb": None}
    properties = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "vram_total_mb": int(properties.total_memory / 1024 / 1024),
    }


def docs_description() -> str:
    try:
        return DOCS_DESCRIPTION_PATH.read_text(encoding="utf-8")
    except OSError:
        return (
            "WSL ASR Server API. Runtime model capabilities must be discovered "
            "from GET /v1/models."
        )


def mcp_tutorial() -> str:
    try:
        return MCP_TUTORIAL_PATH.read_text(encoding="utf-8")
    except OSError:
        return "# MCP setup tutorial\n\nThe MCP tutorial file is unavailable in this deployment.\n"


def create_app(settings: Settings | None = None, adapter_delay_seconds: float = 0.0, *,
               models: Mapping[str, ModelDefinition] | None = None,
               engines: Mapping[str, EngineRegistration] | None = None) -> FastAPI:
    app_settings = settings or load_settings()
    definitions = dict(models) if models is not None else default_models(
        app_settings.default_model, enable_moss=app_settings.enable_moss,
        enable_qwen_alignment=app_settings.enable_qwen_alignment,
    )

    def adapter_factory(model_id: str) -> AsrAdapter:
        return create_registered_adapter(definitions[model_id], app_settings, engines=engines,
                                         adapter_delay_seconds=adapter_delay_seconds)

    @asynccontextmanager
    async def lifespan(lifespan_app: FastAPI) -> AsyncIterator[None]:
        job_manager: JobManager = lifespan_app.state.job_manager
        manager: ModelLifecycleManager = lifespan_app.state.manager
        logging_runtime = configure_logging(level=app_settings.log_level, log_format=app_settings.log_format, log_dir=app_settings.log_dir)
        async def sample_gpu() -> None:
            while True:
                if app_settings.adapter != "mock":
                    try:
                        lifespan_app.state.gpu_snapshot = await asyncio.to_thread(gpu_health)
                    except Exception as exc:
                        emit("health.gpu_snapshot_failed", diagnostics=safe_exception(exc))
                lifespan_app.state.gpu_snapshot_at = datetime.now(timezone.utc).isoformat()
                await asyncio.sleep(30)
        gpu_sampler = asyncio.create_task(sample_gpu())
        await job_manager.start()
        try:
            if app_settings.workbench_enabled:
                await lifespan_app.state.workbench.start()
            yield
        finally:
            if app_settings.workbench_enabled:
                await lifespan_app.state.workbench.shutdown()
            await job_manager.shutdown()
            await manager.shutdown()
            if app_settings.workbench_enabled:
                lifespan_app.state.workbench.close()
            gpu_sampler.cancel()
            try:
                await gpu_sampler
            except asyncio.CancelledError:
                pass
            logging_runtime.close()

    app = GatewayApp(
        title="WSL ASR Server",
        version=__version__,
        description=docs_description(),
        lifespan=lifespan,
    )
    app.router.route_class = AdmissionRoute
    app.add_middleware(AdmissionMiddleware, settings=app_settings)
    app.add_exception_handler(AsrError, cast(Any, asr_error_handler))
    app.add_exception_handler(RequestValidationError, cast(Any, validation_error_handler))
    if app_settings.api_token:
        app.add_middleware(BearerAuthMiddleware, token=app_settings.api_token)
    app.add_middleware(RequestContextMiddleware)
    app.state.manager = ModelLifecycleManager(
        definitions,
        adapter_factory,
        idle_unload_seconds=app_settings.idle_unload_seconds,
    )
    app.state.gpu_snapshot = {"available": False, "name": None, "vram_total_mb": None}
    app.state.gpu_snapshot_at = None
    app.state.settings = app_settings
    app.state.workspace_manager = WorkspaceManager(app_settings)
    app.state.job_manager = JobManager(app.state.manager, app_settings)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "host": socket.gethostname(),
            "gpu": app.state.gpu_snapshot,
            "gpu_snapshot_at": app.state.gpu_snapshot_at,
        }

    @app.get("/docs/mcp", response_class=PlainTextResponse, tags=["documentation"])
    async def mcp_setup_tutorial() -> PlainTextResponse:
        """Return the Mac-side stdio MCP setup tutorial as Markdown."""
        return PlainTextResponse(mcp_tutorial(), media_type="text/markdown")

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        manager: ModelLifecycleManager = app.state.manager
        return {"models": manager.list_models()}

    @app.get("/v1/models/{model_id}/status")
    async def model_status(model_id: str) -> dict[str, object]:
        manager: ModelLifecycleManager = app.state.manager
        return manager.runtime_for(model_id).status_summary()

    @app.post("/v1/models/{model_id}/load")
    async def load_model(
        model_id: str,
        request: Annotated[LoadRequest, Body()] = LoadRequest(),
    ) -> dict[str, object]:
        manager: ModelLifecycleManager = app.state.manager
        # Validate capability before spending a queue slot.
        manager.resolve_backend(manager.runtime_for(model_id), request.backend)
        await app.state.job_manager.start()
        return cast(dict[str, object], await app.state.job_manager.scheduler.wait(
            "load_" + uuid4().hex,
            lambda: manager.load_model(model_id, backend=request.backend, device=request.device, dtype=request.dtype),
            kind="load",
        ))

    @app.delete("/v1/models/{model_id}")
    async def unload_model(
        model_id: str,
        request: Annotated[UnloadRequest, Body()] = UnloadRequest(),
    ) -> dict[str, object]:
        del request.mode, request.reject_new_requests
        manager: ModelLifecycleManager = app.state.manager
        return await manager.unload_model(model_id, cuda_empty_cache=request.cuda_empty_cache)

    @app.delete("/v1/models")
    async def unload_models(request: Annotated[UnloadRequest, Body()] = UnloadRequest()) -> dict[str, object]:
        del request.mode, request.reject_new_requests
        manager: ModelLifecycleManager = app.state.manager
        return await manager.unload_all(cuda_empty_cache=request.cuda_empty_cache)

    @app.post("/v1/audio/transcriptions", response_model=None)
    async def transcribe(
        raw_request: Request,
        file: Annotated[UploadFile, File()],
        model: Annotated[str | None, Form()] = None,
        language: Annotated[str, Form()] = "auto",
        response_format: Annotated[str | None, Form()] = None,
        timestamps: Annotated[str | None, Form()] = None,
        backend: Annotated[Backend, Form()] = "auto",
        temperature: Annotated[float | None, Form()] = None,
        max_new_tokens: Annotated[int | None, Form(ge=1)] = None,
        context: Annotated[str, Form()] = "",
        hotwords: Annotated[str | None, Form()] = None,
        split_strategy: Annotated[str | None, Form()] = None,
        max_chunk_seconds: Annotated[float | None, Form()] = None,
        overlap_seconds: Annotated[float | None, Form()] = None,
        preserve_segments: Annotated[bool | None, Form()] = None,
        speaker_resolution: Annotated[str | None, Form()] = None,
        purpose: Annotated[str, Form()] = "transcription",
        pause_threshold_seconds: Annotated[float, Form()] = 0.8,
        idempotency_key: Annotated[str | None, Form()] = None,
        quality_profile: Annotated[str, Form()] = "legacy",
    ) -> dict[str, object] | PlainTextResponse | JSONResponse:
        del temperature
        manager: ModelLifecycleManager = app.state.manager
        request = _transcription_request(
            model=model,
            language=language,
            response_format=response_format,
            timestamps=timestamps,
            backend=backend,
            max_new_tokens=max_new_tokens,
            context=context,
            hotwords=hotwords,
            split_strategy=split_strategy,
            max_chunk_seconds=max_chunk_seconds,
            overlap_seconds=overlap_seconds,
            preserve_segments=preserve_segments,
            speaker_resolution=speaker_resolution,
            purpose=purpose,
            pause_threshold_seconds=pause_threshold_seconds,
            idempotency_key=idempotency_key,
            quality_profile=quality_profile,
        )
        validated = validate_transcription_request(manager, request)
        workspace = await app.state.workspace_manager.store_upload(file)
        transferred = False
        try:
            duration_seconds = await probe_audio_duration_path_async(
                workspace.upload_path,
                timeout_seconds=app_settings.ffprobe_timeout_seconds,
            )
            if duration_seconds is None:
                # Let the path preprocessing stage produce the canonical decode
                # error rather than decoding the entire upload twice.
                duration_seconds = 0.0
            if duration_seconds > SYNC_JOB_THRESHOLD_SECONDS:
                job_manager: JobManager = app.state.job_manager
                payload = await job_manager.create_job(
                    workspace=workspace,
                    filename=file.filename,
                    content_type=file.content_type,
                    request=request,
                )
                transferred = True
                return JSONResponse(status_code=202, content=payload)
            await app.state.job_manager.start()
            execution = current_context()
            async def run_sync() -> dict[str, object]:
                return await run_transcription_path(
                    workspace.upload_path, workspace=workspace.root, manager=manager,
                    settings=app_settings, request=request, validated=validated,
                    cancel_event=execution.cancel_event, owner_workspace=workspace,
                )
            pending = asyncio.create_task(app.state.job_manager.scheduler.wait(
                "sync_" + uuid4().hex, run_sync, on_cancel=execution.cancel_event.set,
            ))
            async def disconnect() -> None:
                while not pending.done():
                    if await raw_request.is_disconnected():
                        pending.cancel()
                        return
                    await asyncio.sleep(0.1)
            watcher = asyncio.create_task(disconnect())
            try:
                result = cast(dict[str, object], await pending)
            finally:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass
        finally:
            if not transferred:
                await workspace.cleanup()
        if request.response_format == "text":
            return PlainTextResponse(cast(str, result["text"]))
        return result

    @app.post("/v1/audio/transcription-jobs", status_code=202)
    async def create_transcription_job(
        file: Annotated[UploadFile, File()],
        model: Annotated[str | None, Form()] = None,
        language: Annotated[str, Form()] = "auto",
        response_format: Annotated[str | None, Form()] = None,
        timestamps: Annotated[str | None, Form()] = None,
        backend: Annotated[Backend, Form()] = "auto",
        temperature: Annotated[float | None, Form()] = None,
        max_new_tokens: Annotated[int | None, Form(ge=1)] = None,
        context: Annotated[str, Form()] = "",
        hotwords: Annotated[str | None, Form()] = None,
        split_strategy: Annotated[str | None, Form()] = None,
        max_chunk_seconds: Annotated[float | None, Form()] = None,
        overlap_seconds: Annotated[float | None, Form()] = None,
        preserve_segments: Annotated[bool | None, Form()] = None,
        speaker_resolution: Annotated[str | None, Form()] = None,
        purpose: Annotated[str, Form()] = "transcription",
        pause_threshold_seconds: Annotated[float, Form()] = 0.8,
        idempotency_key: Annotated[str | None, Form()] = None,
        quality_profile: Annotated[str, Form()] = "legacy",
    ) -> dict[str, object]:
        del temperature
        request = _transcription_request(
            model=model,
            language=language,
            response_format=response_format,
            timestamps=timestamps,
            backend=backend,
            max_new_tokens=max_new_tokens,
            context=context,
            hotwords=hotwords,
            split_strategy=split_strategy,
            max_chunk_seconds=max_chunk_seconds,
            overlap_seconds=overlap_seconds,
            preserve_segments=preserve_segments,
            speaker_resolution=speaker_resolution,
            purpose=purpose,
            pause_threshold_seconds=pause_threshold_seconds,
            idempotency_key=idempotency_key,
            quality_profile=quality_profile,
        )
        manager: ModelLifecycleManager = app.state.manager
        validate_transcription_request(manager, request)
        workspace = await app.state.workspace_manager.store_upload(file)
        job_manager: JobManager = app.state.job_manager
        try:
            return await job_manager.create_job(
                workspace=workspace,
                filename=file.filename,
                content_type=file.content_type,
                request=request,
            )
        except BaseException:
            await workspace.cleanup()
            raise

    @app.get("/v1/jobs/{job_id}")
    async def get_transcription_job(job_id: str, detail: str = "compact") -> dict[str, object]:
        job_manager: JobManager = app.state.job_manager
        return await job_manager.get_job(job_id, detail=detail)

    @app.delete("/v1/jobs/{job_id}")
    async def cancel_transcription_job(job_id: str) -> dict[str, object]:
        job_manager: JobManager = app.state.job_manager
        return await job_manager.cancel_job(job_id)

    @app.post("/v1/audio/alignments")
    async def alignments() -> dict[str, object]:
        raise AsrError(
            422,
            "capability_not_supported",
            "forced alignment is not enabled in this server",
        )

    if app_settings.workbench_enabled:
        from asr_server.workbench.router import install_workbench

        app.state.workbench = install_workbench(app, app_settings)
        static_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
        if static_dir.is_dir():
            app.mount("/", StaticFiles(directory=static_dir, html=True), name="workbench")
    return app


def _transcription_request(
    *,
    model: str | None,
    language: str,
    response_format: str | None,
    timestamps: str | None,
    backend: Backend,
    max_new_tokens: int | None,
    context: str,
    hotwords: str | None,
    split_strategy: str | None,
    max_chunk_seconds: float | None,
    overlap_seconds: float | None,
    preserve_segments: bool | None,
    speaker_resolution: str | None,
    purpose: str,
    pause_threshold_seconds: float,
    idempotency_key: str | None,
    quality_profile: str = "legacy",
) -> TranscriptionRequest:
    profile_defaults = {
        "transcription": ("json", "none", False, "off"),
        "oral_analysis": ("verbose_json", "auto", True, "off"),
        "podcast": ("verbose_json", "none", True, "auto"),
    }
    defaults = profile_defaults.get(purpose, profile_defaults["transcription"])
    return TranscriptionRequest(
        model=model,
        language=language,
        response_format=response_format or defaults[0],
        timestamps=timestamps or defaults[1],
        backend=backend,
        max_new_tokens=max_new_tokens,
        context=context,
        hotwords=hotwords,
        split_strategy=split_strategy or "auto",
        max_chunk_seconds=max_chunk_seconds,
        overlap_seconds=overlap_seconds,
        preserve_segments=defaults[2] if preserve_segments is None else preserve_segments,
        speaker_resolution=speaker_resolution or defaults[3],
        purpose=purpose,
        pause_threshold_seconds=pause_threshold_seconds,
        idempotency_key=idempotency_key,
        quality_profile=quality_profile,
    )


app = create_app()
