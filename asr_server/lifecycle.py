from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from time import monotonic, perf_counter
from typing import Any, Protocol, TypeVar

from asr_server.adapters.base import AudioInput, AsrAdapter, TimestampItem, TranscriptionResult, TranscriptionTimings
from asr_server.errors import AsrError
from asr_server.observability import emit, safe_exception
from asr_server.registry import Backend, ModelDefinition, ModelStatus
from asr_server.workers.transport import ProcessRpcTransport


ChunkProgressCallback = Callable[[int, int], Awaitable[None]]
ChunkResultCallback = Callable[[int, int, TranscriptionResult], Awaitable[None]]
T = TypeVar("T")
logger = logging.getLogger(__name__)


class AdaptiveTranscriptionSequence(Protocol):
    def __len__(self) -> int: ...

    def audio_for(self, index: int) -> AudioInput: ...

    def accept(self, index: int, result: TranscriptionResult) -> TranscriptionResult: ...


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class LoadConfig:
    model_id: str
    backend: str
    device: str
    dtype: str
    revision: str


@dataclass
class ModelRuntime:
    definition: ModelDefinition
    adapter: AsrAdapter
    status: ModelStatus = "unloaded"
    active_requests: int = 0
    rejecting_new_requests: bool = False
    backend: str | None = None
    device: str | None = None
    dtype: str | None = None
    load_config: LoadConfig | None = None
    loaded_at: str | None = None
    last_used_at: str | None = None
    vram_allocated_mb: int | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    request_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    idle_unload_task: asyncio.Task[None] | None = None
    consecutive_worker_failures: int = 0
    circuit_until: float = 0.0
    last_worker_error: str | None = None

    def model_summary(self) -> dict[str, object]:
        return {
            **self.definition.discovery(),
            "id": self.definition.id,
            "provider": self.definition.provider,
            "status": self.status,
            "default": self.definition.default,
            "revision": self.definition.revision,
            "capabilities": self.definition.capabilities.to_api(),
        }

    def status_summary(self) -> dict[str, object]:
        return {
            "id": self.definition.id,
            "status": self.status,
            "active_requests": self.active_requests,
            "rejecting_new_requests": self.rejecting_new_requests,
            "backend": self.backend,
            "device": self.device,
            "dtype": self.dtype,
            "load_config": None
            if self.load_config is None
            else {
                "model_id": self.load_config.model_id,
                "backend": self.load_config.backend,
                "device": self.load_config.device,
                "dtype": self.load_config.dtype,
                "revision": self.load_config.revision,
            },
            "loaded_at": self.loaded_at,
            "last_used_at": self.last_used_at,
            "vram_allocated_mb": self.vram_allocated_mb,
        }


class ModelLifecycleManager:
    def __init__(
        self,
        models: dict[str, ModelDefinition],
        adapter_factory: Callable[[str], AsrAdapter],
        idle_unload_seconds: float = 180.0,
        circuit_failure_limit: int = 3,
        circuit_cooldown_seconds: float = 60.0,
    ) -> None:
        self._runtimes = {
            model_id: ModelRuntime(definition=model, adapter=adapter_factory(model_id))
            for model_id, model in models.items()
        }
        self._default_model_id = next(model.id for model in models.values() if model.default)
        self._idle_unload_seconds = idle_unload_seconds
        # This is intentionally global rather than per ModelRuntime. The target
        # deployment has one GPU and must never load or run two model workers at
        # the same time without an explicitly validated memory policy.
        self._gpu_operation_lock = asyncio.Lock()
        self._circuit_failure_limit = circuit_failure_limit
        self._circuit_cooldown_seconds = circuit_cooldown_seconds

    def inference_readiness(self, model_id: str | None = None) -> dict[str, object]:
        runtime = self.runtime_for(model_id or self.default_model_id)
        quarantine = ProcessRpcTransport.quarantine_status()
        retry_after = max(0.0, runtime.circuit_until - monotonic())
        reasons = []
        if quarantine["isolated"]:
            reasons.append("worker_isolated")
        if retry_after > 0:
            reasons.append("worker_circuit_open")
        return {"ready": not reasons, "reasons": reasons, "retry_after_seconds": retry_after}

    def worker_diagnostics(self) -> dict[str, object]:
        return {"quarantine": ProcessRpcTransport.quarantine_status(), "models": [
            {"id": runtime.definition.id, "consecutive_failures": runtime.consecutive_worker_failures,
             "last_error": runtime.last_worker_error,
             "retry_after_seconds": max(0.0, runtime.circuit_until - monotonic()),
             "recovery_probe_available": bool(runtime.circuit_until and runtime.circuit_until <= monotonic())}
            for runtime in self._runtimes.values()]}

    def assert_inference_available(self, model_id: str | None = None) -> None:
        ProcessRpcTransport.assert_gpu_safe()
        state = self.inference_readiness(model_id)
        if not state["ready"]:
            raise AsrError(503, "worker_circuit_open", "model worker is cooling down after repeated failures",
                           {"retryable": True, "retry_after_seconds": state["retry_after_seconds"]})

    def _record_worker_failure(self, runtime: ModelRuntime, error: BaseException) -> None:
        if not isinstance(error, AsrError) or not error.details.get("worker_fatal"):
            return
        runtime.consecutive_worker_failures += 1
        runtime.last_worker_error = error.code
        if runtime.consecutive_worker_failures >= self._circuit_failure_limit:
            runtime.circuit_until = monotonic() + self._circuit_cooldown_seconds
            emit("worker.circuit_opened", model_id=runtime.definition.id,
                 failures=runtime.consecutive_worker_failures,
                 cooldown_seconds=self._circuit_cooldown_seconds)

    def _record_worker_success(self, runtime: ModelRuntime) -> None:
        if runtime.consecutive_worker_failures:
            emit("worker.circuit_closed", model_id=runtime.definition.id,
                 failures=runtime.consecutive_worker_failures)
        runtime.consecutive_worker_failures = 0
        runtime.circuit_until = 0.0
        runtime.last_worker_error = None

    @property
    def default_model_id(self) -> str:
        return self._default_model_id

    def list_models(self) -> list[dict[str, object]]:
        return [runtime.model_summary() for runtime in self._runtimes.values()]

    def model_for_purpose(self, purpose: str) -> str:
        if purpose == "podcast":
            candidate = next(
                (runtime.definition.id for runtime in self._runtimes.values() if runtime.definition.capabilities.diarization),
                None,
            )
            if candidate is None:
                raise AsrError(
                    422,
                    "capability_not_supported",
                    "purpose=podcast requires an enabled diarization model",
                    {"required_capability": "diarization"},
                )
            return candidate
        return self.default_model_id

    def runtime_for(self, model_id: str) -> ModelRuntime:
        try:
            return self._runtimes[model_id]
        except KeyError as exc:
            raise AsrError(404, "model_not_found", f"unknown model: {model_id}") from exc

    async def shutdown(self) -> None:
        async with self._gpu_operation_lock:
            for runtime in self._runtimes.values():
                self._cancel_idle_unload(runtime)
                async with runtime.lock:
                    if runtime.active_requests > 0:
                        continue
                    if runtime.status == "unloaded" and runtime.backend is None:
                        continue
                    try:
                        await self._unload_now(runtime, cuda_empty_cache=True)
                    except Exception as exc:
                        emit("model.shutdown_unload_failed", level=logging.WARNING,
                             model_id=runtime.definition.id, diagnostics=safe_exception(exc))
                        runtime.status = "error"
                        runtime.rejecting_new_requests = False

    async def abort_all_workers(self) -> None:
        """Force-stop model workers after the service shutdown grace expires."""
        for runtime in self._runtimes.values():
            try:
                await _call_adapter(runtime.adapter.abort)
            except Exception as exc:
                emit("model.abort_failed", level=logging.WARNING,
                     model_id=runtime.definition.id, diagnostics=safe_exception(exc))
            async with runtime.lock:
                runtime.status = "error"
                runtime.backend = None
                runtime.device = None
                runtime.dtype = None
                runtime.load_config = None
                runtime.loaded_at = None
                runtime.rejecting_new_requests = True

    def resolve_backend(self, runtime: ModelRuntime, backend: Backend) -> str:
        if backend == "auto":
            if runtime.backend is not None:
                return runtime.backend
            return runtime.definition.capabilities.backends[0]
        if backend not in runtime.definition.capabilities.backends:
            raise AsrError(
                422,
                "capability_not_supported",
                f"{runtime.definition.id} does not support backend: {backend}",
                {"backend": backend},
            )
        return backend

    async def load_model(
        self,
        model_id: str,
        *,
        backend: Backend = "auto",
        device: str = "cuda",
        dtype: str = "auto",
    ) -> dict[str, object]:
        runtime = self.runtime_for(model_id)
        resolved_backend = self.resolve_backend(runtime, backend)
        requested_config = LoadConfig(model_id, resolved_backend, device, dtype, runtime.definition.revision)
        async with self._gpu_operation_lock:
            self.assert_inference_available(runtime.definition.id)
            await self._unload_other_runtimes(runtime)
            async with runtime.request_lock:
                async with runtime.lock:
                    self._cancel_idle_unload(runtime)
                    if runtime.status in ("unloading", "unloading_scheduled"):
                        raise AsrError(409, "model_unloading_scheduled", f"model is unloading: {model_id}")
                    if (
                        runtime.status == "loaded"
                        and runtime.backend == resolved_backend
                        and runtime.load_config == requested_config
                    ):
                        return {"id": model_id, "status": runtime.status, "message": "model already loaded"}
                    if runtime.status == "loaded":
                        await self._unload_now(runtime, cuda_empty_cache=True)
                    runtime.status = "loading"
                    emit("model.loading", model_id=model_id, engine_id=runtime.definition.engine_id, model_revision=runtime.definition.revision)
                    try:
                        await _call_adapter(lambda: runtime.adapter.load(resolved_backend, device, dtype))
                    except Exception as exc:
                        self._record_worker_failure(runtime, exc)
                        runtime.status = "error"
                        runtime.rejecting_new_requests = False
                        raise
                    runtime.status = "loaded"
                    runtime.rejecting_new_requests = False
                    runtime.backend = resolved_backend
                    runtime.device = device
                    runtime.dtype = dtype
                    runtime.load_config = requested_config
                    runtime.loaded_at = utc_now_iso()
                    emit("model.loaded", model_id=model_id, engine_id=runtime.definition.engine_id)
                    return {"id": model_id, "status": runtime.status, "message": "model loaded"}

    async def unload_model(
        self,
        model_id: str,
        *,
        cuda_empty_cache: bool = True,
    ) -> dict[str, object]:
        runtime = self.runtime_for(model_id)
        async with runtime.lock:
            self._cancel_idle_unload(runtime)
            if runtime.active_requests > 0:
                runtime.status = "unloading_scheduled"
                runtime.rejecting_new_requests = True
                return {
                    "id": model_id,
                    "status": runtime.status,
                    "active_requests": runtime.active_requests,
                    "rejecting_new_requests": runtime.rejecting_new_requests,
                }
        async with self._gpu_operation_lock:
            async with runtime.lock:
                if runtime.active_requests > 0:
                    runtime.status = "unloading_scheduled"
                    runtime.rejecting_new_requests = True
                    return {
                        "id": model_id,
                        "status": runtime.status,
                        "active_requests": runtime.active_requests,
                        "rejecting_new_requests": runtime.rejecting_new_requests,
                    }
                await self._unload_now(runtime, cuda_empty_cache=cuda_empty_cache)
            return {
                "id": model_id,
                "status": runtime.status,
                "active_requests": runtime.active_requests,
                "rejecting_new_requests": runtime.rejecting_new_requests,
            }

    async def unload_all(self, *, cuda_empty_cache: bool = True) -> dict[str, object]:
        models = []
        for model_id in self._runtimes:
            models.append(await self.unload_model(model_id, cuda_empty_cache=cuda_empty_cache))
        return {"status": "accepted", "models": models}

    async def transcribe(
        self,
        audio: bytes,
        *,
        model_id: str | None,
        backend: Backend,
        language: str,
        timestamps: str,
        context: str = "",
        max_new_tokens: int | None = None,
        batch_size: int = 1,
    ) -> tuple[str, TranscriptionResult]:
        resolved_backend, results, timings = await self.transcribe_chunks(
            [audio],
            model_id=model_id,
            backend=backend,
            language=language,
            timestamps=timestamps,
            context=context,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
        return resolved_backend, replace(results[0], timings=timings)

    async def align_text(
        self,
        audio: AudioInput,
        *,
        text: str,
        model_id: str | None = None,
        backend: Backend = "auto",
        language: str = "auto",
        timestamps: str = "auto",
    ) -> list[TimestampItem]:
        """Align supplied text under the same GPU lease as transcription.

        The adapter must align the supplied text, never silently transcribe a
        replacement. Returned offsets are relative to the supplied audio view.
        """
        async def operation(runtime: ModelRuntime, selected_model_id: str, resolved_backend: str) -> list[TranscriptionResult]:
            del selected_model_id, resolved_backend
            align = getattr(runtime.adapter, "align_text", None)
            if not callable(align):
                raise AsrError(422, "capability_not_supported", "adapter does not support supplied-text alignment")
            result = await _call_adapter(lambda: align(audio, text=text, language=language, timestamps=timestamps))
            if not isinstance(result, list) or not all(isinstance(item, TimestampItem) for item in result):
                raise AsrError(422, "alignment_failed", "adapter returned invalid supplied-text alignment")
            return [TranscriptionResult(text, 0.0, language, [], timestamps=result)]

        _backend, results, _timings = await self._transcribe_managed(
            model_id=model_id, backend=backend, timestamps=timestamps, operation=operation,
        )
        return results[0].timestamps

    async def transcribe_chunks(
        self,
        audio_chunks: list[AudioInput],
        *,
        model_id: str | None,
        backend: Backend,
        language: str,
        timestamps: str,
        context: str,
        max_new_tokens: int | None,
        batch_size: int,
        before_chunk: ChunkProgressCallback | None = None,
        after_chunk: ChunkResultCallback | None = None,
    ) -> tuple[str, list[TranscriptionResult], TranscriptionTimings]:
        if not audio_chunks:
            raise AsrError(400, "bad_request", "audio request did not contain any chunks")
        return await self._transcribe_managed(
            model_id=model_id,
            backend=backend,
            timestamps=timestamps,
            operation=lambda runtime, selected_model_id, resolved_backend: self._transcribe_chunk_results(
                runtime,
                audio_chunks,
                model_id=selected_model_id,
                backend=resolved_backend,
                language=language,
                context=context,
                timestamps=timestamps,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                before_chunk=before_chunk,
                after_chunk=after_chunk,
            ),
        )

    async def transcribe_sequence(
        self,
        sequence: AdaptiveTranscriptionSequence,
        *,
        model_id: str | None,
        backend: Backend,
        language: str,
        timestamps: str,
        context: str,
        max_new_tokens: int | None,
        before_chunk: ChunkProgressCallback | None = None,
        after_chunk: ChunkResultCallback | None = None,
    ) -> tuple[str, list[TranscriptionResult], TranscriptionTimings]:
        """Run an adaptive sequence as one model request and one GPU lease."""

        if len(sequence) == 0:
            raise AsrError(400, "bad_request", "audio request did not contain any chunks")
        return await self._transcribe_managed(
            model_id=model_id,
            backend=backend,
            timestamps=timestamps,
            operation=lambda runtime, selected_model_id, resolved_backend: self._transcribe_sequence_results(
                runtime,
                sequence,
                model_id=selected_model_id,
                backend=resolved_backend,
                language=language,
                context=context,
                timestamps=timestamps,
                max_new_tokens=max_new_tokens,
                before_chunk=before_chunk,
                after_chunk=after_chunk,
            ),
        )

    async def _transcribe_managed(
        self,
        *,
        model_id: str | None,
        backend: Backend,
        timestamps: str,
        operation: Callable[[ModelRuntime, str, str], Awaitable[list[TranscriptionResult]]],
    ) -> tuple[str, list[TranscriptionResult], TranscriptionTimings]:
        request_started = perf_counter()
        selected_model_id = model_id or self.default_model_id
        runtime = self.runtime_for(selected_model_id)
        if timestamps != "none" and not runtime.definition.capabilities.timestamps:
            raise AsrError(
                422,
                "capability_not_supported",
                f"{selected_model_id} does not support timestamps in this server",
                {"timestamps": timestamps},
            )
        async with runtime.lock:
            if runtime.status == "unloading_scheduled" or runtime.rejecting_new_requests:
                raise AsrError(
                    409,
                    "model_unloading_scheduled",
                    f"model is scheduled for unloading: {runtime.definition.id}",
                )
        async with self._gpu_operation_lock:
            self.assert_inference_available(runtime.definition.id)
            await self._unload_other_runtimes(runtime)
            async with runtime.request_lock:
                resolved_backend, load_ms = await self._begin_request(runtime, backend=backend)
                try:
                    results = await operation(runtime, selected_model_id, resolved_backend)
                    self._record_worker_success(runtime)
                    runtime.last_used_at = utc_now_iso()
                    timings = TranscriptionTimings(
                        total_ms=(perf_counter() - request_started) * 1000,
                        load_ms=load_ms,
                        decode_ms=sum(result.timings.decode_ms for result in results),
                        inference_ms=sum(result.timings.inference_ms for result in results),
                        postprocess_ms=sum(result.timings.postprocess_ms for result in results),
                    )
                    return resolved_backend, results, timings
                except AsrError as exc:
                    self._record_worker_failure(runtime, exc)
                    if exc.details.get("worker_fatal"):
                        async with runtime.lock:
                            runtime.status = "error"
                            runtime.backend = None
                            runtime.device = None
                            runtime.dtype = None
                            runtime.load_config = None
                            runtime.loaded_at = None
                    raise
                finally:
                    await self._finish_request(runtime)

    async def _transcribe_sequence_results(
        self,
        runtime: ModelRuntime,
        sequence: AdaptiveTranscriptionSequence,
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        timestamps: str,
        max_new_tokens: int | None,
        before_chunk: ChunkProgressCallback | None,
        after_chunk: ChunkResultCallback | None,
    ) -> list[TranscriptionResult]:
        results: list[TranscriptionResult] = []
        total_count = len(sequence)
        for index in range(total_count):
            if before_chunk is not None:
                await before_chunk(index, total_count)
            chunk_started = perf_counter()
            audio_input = sequence.audio_for(index)
            try:
                raw_result = await _run_inference(
                    lambda: _call_adapter(
                        lambda: _adapter_transcribe(
                            runtime.adapter,
                            audio_input,
                            model_id=model_id,
                            backend=backend,
                            language=language,
                            context=context,
                            timestamps=timestamps,
                            max_new_tokens=max_new_tokens,
                        )
                    )
                )
            except AsrError:
                raise
            except Exception as exc:
                raise AsrError(
                    503,
                    "inference_failed",
                    "sequence transcription failed",
                    {"chunk_index": index, "error_type": type(exc).__name__},
                ) from exc
            result = sequence.accept(index, raw_result)
            if result.timings.total_ms == 0:
                result = replace(
                    result,
                    timings=replace(result.timings, total_ms=(perf_counter() - chunk_started) * 1000),
                )
            results.append(result)
            if after_chunk is not None:
                await after_chunk(index, total_count, result)
        return results

    async def _transcribe_chunk_results(
        self,
        runtime: ModelRuntime,
        audio_chunks: list[AudioInput],
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        timestamps: str,
        max_new_tokens: int | None,
        batch_size: int,
        before_chunk: ChunkProgressCallback | None,
        after_chunk: ChunkResultCallback | None,
    ) -> list[TranscriptionResult]:
        if len(audio_chunks) == 1:
            if before_chunk is not None:
                await before_chunk(0, 1)
            chunk_started = perf_counter()
            result = await _run_inference(
                lambda: _call_adapter(
                    lambda: _adapter_transcribe(
                        runtime.adapter,
                        audio_chunks[0],
                        model_id=model_id,
                        backend=backend,
                        language=language,
                        context=context,
                        timestamps=timestamps,
                        max_new_tokens=max_new_tokens,
                    )
                )
            )
            if result.timings.total_ms == 0:
                result = replace(
                    result,
                    timings=replace(result.timings, total_ms=(perf_counter() - chunk_started) * 1000),
                )
            if after_chunk is not None:
                await after_chunk(0, 1, result)
            return [result]

        results = []
        effective_batch_size = 1 if timestamps != "none" else max(batch_size, 1)
        completed_count = 0
        total_count = len(audio_chunks)
        for batch in _chunked(audio_chunks, effective_batch_size):
            batch_start_index = completed_count
            if before_chunk is not None:
                for offset in range(len(batch)):
                    await before_chunk(batch_start_index + offset, total_count)
            batch_started = perf_counter()
            try:
                if timestamps != "none":
                    batch_results = [
                        await _run_inference(
                            lambda: _call_adapter(
                                lambda: _adapter_transcribe(
                                    runtime.adapter,
                                    batch[0],
                                    model_id=model_id,
                                    backend=backend,
                                    language=language,
                                    context=context,
                                    timestamps=timestamps,
                                    max_new_tokens=max_new_tokens,
                                )
                            )
                        )
                    ]
                else:
                    batch_results = await _run_inference(
                        lambda: _call_adapter(
                            lambda: _adapter_transcribe_batch(
                                runtime.adapter,
                                batch,
                                model_id=model_id,
                                backend=backend,
                                language=language,
                                context=context,
                                timestamps=timestamps,
                                max_new_tokens=max_new_tokens,
                            )
                        )
                    )
            except AsrError:
                raise
            except Exception as exc:
                raise AsrError(
                    503,
                    "inference_failed",
                    "batch transcription failed",
                    {"error_type": type(exc).__name__},
                ) from exc
            if len(batch_results) != len(batch):
                raise AsrError(
                    503,
                    "inference_failed",
                    "batch transcription returned an unexpected result count",
                    {"expected": len(batch), "actual": len(batch_results)},
                )
            batch_ms = (perf_counter() - batch_started) * 1000
            per_chunk_ms = batch_ms / len(batch)
            for offset, result in enumerate(batch_results):
                if result.timings.total_ms == 0:
                    result = replace(result, timings=replace(result.timings, total_ms=per_chunk_ms))
                results.append(result)
                completed_count += 1
                if after_chunk is not None:
                    await after_chunk(batch_start_index + offset, total_count, result)
        return results

    async def _begin_request(
        self,
        runtime: ModelRuntime,
        *,
        backend: Backend,
    ) -> tuple[str, float]:
        resolved_backend = self.resolve_backend(runtime, backend)
        load_ms = 0.0
        async with runtime.lock:
            self._cancel_idle_unload(runtime)
            if runtime.status == "unloading_scheduled" or runtime.rejecting_new_requests:
                raise AsrError(
                    409,
                    "model_unloading_scheduled",
                    f"model is scheduled for unloading: {runtime.definition.id}",
                )
            if runtime.status == "loading":
                raise AsrError(409, "model_loading", f"model is loading: {runtime.definition.id}")
            if runtime.status == "unloading":
                raise AsrError(
                    409,
                    "model_unloading_scheduled",
                    f"model is unloading: {runtime.definition.id}",
                )
            if runtime.status in ("unloaded", "error") or runtime.backend != resolved_backend:
                runtime.status = "loading"
                load_started = perf_counter()
                emit("model.loading", model_id=runtime.definition.id, engine_id=runtime.definition.engine_id, model_revision=runtime.definition.revision)
                try:
                    await _call_adapter(
                        lambda: runtime.adapter.load(resolved_backend, "cuda", "auto")
                    )
                except Exception as exc:
                    self._record_worker_failure(runtime, exc)
                    runtime.status = "error"
                    runtime.rejecting_new_requests = False
                    raise
                load_ms = (perf_counter() - load_started) * 1000
                runtime.status = "loaded"
                runtime.backend = resolved_backend
                runtime.device = "cuda"
                runtime.dtype = "auto"
                runtime.load_config = LoadConfig(
                    runtime.definition.id,
                    resolved_backend,
                    "cuda",
                    "auto",
                    runtime.definition.revision,
                )
                runtime.loaded_at = utc_now_iso()
                emit("model.loaded", model_id=runtime.definition.id, duration_ms=load_ms)
                runtime.rejecting_new_requests = False
            runtime.active_requests += 1
            return resolved_backend, load_ms

    async def _finish_request(self, runtime: ModelRuntime) -> None:
        async with runtime.lock:
            runtime.active_requests = max(runtime.active_requests - 1, 0)
            should_unload = runtime.status == "unloading_scheduled" and runtime.active_requests == 0
            if should_unload:
                await self._unload_now(runtime, cuda_empty_cache=True)
            elif runtime.status == "loaded" and runtime.active_requests == 0:
                self._schedule_idle_unload(runtime)

    async def _unload_now(
        self,
        runtime: ModelRuntime,
        *,
        cuda_empty_cache: bool,
        cancel_idle_task: bool = True,
    ) -> None:
        if cancel_idle_task:
            self._cancel_idle_unload(runtime)
        runtime.status = "unloading"
        emit("model.unloading", model_id=runtime.definition.id)
        await _call_adapter(lambda: runtime.adapter.unload(cuda_empty_cache))
        runtime.status = "unloaded"
        emit("model.unloaded", model_id=runtime.definition.id)
        runtime.rejecting_new_requests = False
        runtime.backend = None
        runtime.device = None
        runtime.dtype = None
        runtime.load_config = None
        runtime.loaded_at = None
        runtime.vram_allocated_mb = None

    def _schedule_idle_unload(self, runtime: ModelRuntime) -> None:
        if self._idle_unload_seconds <= 0:
            return
        self._cancel_idle_unload(runtime)
        runtime.idle_unload_task = asyncio.create_task(self._idle_unload_after_delay(runtime))
        runtime.idle_unload_task.add_done_callback(_consume_task_exception)

    def _cancel_idle_unload(self, runtime: ModelRuntime) -> None:
        task = runtime.idle_unload_task
        if task is None:
            return
        if task is not asyncio.current_task() and not task.done():
            task.cancel()
        runtime.idle_unload_task = None

    async def _idle_unload_after_delay(self, runtime: ModelRuntime) -> None:
        try:
            await asyncio.sleep(self._idle_unload_seconds)
            async with self._gpu_operation_lock:
                async with runtime.lock:
                    if runtime.status != "loaded" or runtime.active_requests > 0 or runtime.rejecting_new_requests:
                        return
                    try:
                        await self._unload_now(runtime, cuda_empty_cache=True, cancel_idle_task=False)
                    except Exception:
                        runtime.status = "error"
                        runtime.rejecting_new_requests = False
                        raise
                    finally:
                        runtime.idle_unload_task = None
        except asyncio.CancelledError:
            return

    async def _unload_other_runtimes(self, target: ModelRuntime) -> None:
        """Unload an idle GPU owner before another model may load or infer.

        The caller must hold ``_gpu_operation_lock``. This keeps model switching
        serial while retaining the PRD behaviour where an unload request can
        mark an active model as ``unloading_scheduled`` without waiting.
        """
        for runtime in self._runtimes.values():
            if runtime is target:
                continue
            async with runtime.lock:
                self._cancel_idle_unload(runtime)
                if runtime.active_requests > 0:
                    raise AsrError(
                        503,
                        "gpu_unavailable",
                        f"GPU is still in use by model: {runtime.definition.id}",
                        {"model": runtime.definition.id},
                    )
                if runtime.status == "unloaded" and runtime.backend is None:
                    continue
                logger.info(
                    "switching GPU owner from model=%s to model=%s",
                    runtime.definition.id,
                    target.definition.id,
                )
                try:
                    await self._unload_now(runtime, cuda_empty_cache=True)
                except Exception:
                    runtime.status = "error"
                    runtime.rejecting_new_requests = False
                    raise


def _chunked(items: list[AudioInput], batch_size: int) -> list[list[AudioInput]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def is_recoverable_worker_error(error: BaseException) -> bool:
    """One task layer owns retries and must persist its attempt before replay."""
    return (isinstance(error, AsrError)
            and error.details.get("worker_reaped") is not False
            and error.code in {"worker_unavailable", "gpu_unavailable", "inference_failed"})


async def _run_inference(operation: Callable[[], Awaitable[T]]) -> T:
    # No implicit adapter/lifecycle retry: durable workbench recovery records
    # every replay, including its quality audio cost, before calling again.
    return await operation()


async def _call_adapter(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    operation = asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(factory())))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        # Cancelling asyncio.to_thread does not stop the underlying thread. Keep
        # ownership of the GPU request until the adapter operation really ends;
        # a transport deadline is responsible for terminating a hung worker.
        try:
            await operation
        finally:
            raise


def _adapter_transcribe(
    adapter: AsrAdapter,
    audio: AudioInput,
    *,
    model_id: str,
    backend: str,
    language: str,
    context: str,
    timestamps: str,
    max_new_tokens: int | None,
) -> Coroutine[Any, Any, TranscriptionResult]:
    kwargs: dict[str, Any] = {
        "model_id": model_id,
        "backend": backend,
        "language": language,
        "context": context,
        "max_new_tokens": max_new_tokens,
    }
    if "timestamps" in inspect.signature(adapter.transcribe).parameters:
        kwargs["timestamps"] = timestamps
    return adapter.transcribe(audio, **kwargs)


def _adapter_transcribe_batch(
    adapter: AsrAdapter,
    audio_chunks: list[AudioInput],
    *,
    model_id: str,
    backend: str,
    language: str,
    context: str,
    timestamps: str,
    max_new_tokens: int | None,
) -> Coroutine[Any, Any, list[TranscriptionResult]]:
    kwargs: dict[str, Any] = {
        "model_id": model_id,
        "backend": backend,
        "language": language,
        "context": context,
        "max_new_tokens": max_new_tokens,
    }
    if "timestamps" in inspect.signature(adapter.transcribe_batch).parameters:
        kwargs["timestamps"] = timestamps
    return adapter.transcribe_batch(audio_chunks, **kwargs)


def _consume_task_exception(task: asyncio.Task[None]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass
