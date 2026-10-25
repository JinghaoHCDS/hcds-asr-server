from __future__ import annotations

import gc
import importlib
import logging
import os
import signal
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from time import perf_counter
from typing import Any

from asr_server.adapters.base import (
    AudioInput,
    ContextApplication,
    GenerationMetrics,
    TimestampItem,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionTimings,
)
from asr_server.errors import AsrError
from asr_server.observability import PROTOCOL_VERSION, safe_exception
from asr_server.registry import ALIGNER_SOURCE, MODEL_SOURCES
from asr_server.workers.audio import audio_duration_seconds, materialized_audio_path
from asr_server.workers.transport import ProcessRpcTransport
from asr_server.workers.codec import (
    optional_int as _optional_int, result_to_payload as _result_to_payload,
    result_from_payload, error_to_payload as _error_to_payload, validate_request_envelope,
)


MODEL_REPOS = {model_id: source[0] for model_id, source in MODEL_SOURCES.items() if model_id.startswith("qwen3-asr-")}
MODEL_REVISIONS = {model_id: source[1] for model_id, source in MODEL_SOURCES.items() if model_id in MODEL_REPOS}
ALIGNER_REPO, ALIGNER_REVISION = ALIGNER_SOURCE
logger = logging.getLogger(__name__)


def _qwen_worker_main(conn: Connection, model_id: str, enable_alignment: bool,
                      repository: str | None = None, revision: str | None = None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    backend = _QwenWorkerBackend(model_id, enable_alignment=enable_alignment, repository=repository, revision=revision)
    try:
        while True:
            request = conn.recv()
            request_id = request.get("id")
            op = request.get("op")
            try:
                validate_request_envelope(request)
                if op == "ping":
                    conn.send({"id": request_id, "ok": True, "result": {"pid": os.getpid(), "protocol_version": PROTOCOL_VERSION}})
                    continue
                if op == "load":
                    _run_async(
                        backend.load(
                            str(request["backend"]),
                            str(request["device"]),
                            str(request["dtype"]),
                            max_new_tokens=_optional_int(request.get("max_new_tokens")),
                        )
                    )
                    conn.send({"id": request_id, "ok": True, "result": None})
                    continue
                if op == "transcribe":
                    result = _run_async(
                        backend.transcribe(
                            request["audio"],
                            model_id=model_id,
                            backend=backend.loaded_backend or "transformers",
                            language=str(request["language"]),
                            context=str(request["context"]),
                            timestamps=str(request.get("timestamps", "none")),
                            max_new_tokens=_optional_int(request.get("max_new_tokens")),
                        )
                    )
                    conn.send({"id": request_id, "ok": True, "result": _result_to_payload(result)})
                    continue
                if op == "align_text":
                    items = _run_async(backend.align_text(
                        request["audio"], text=str(request["text"]), language=str(request["language"]),
                        timestamps=str(request.get("timestamps", "auto")),
                    ))
                    conn.send({"id": request_id, "ok": True, "result": [item.to_api() for item in items]})
                    continue
                if op == "transcribe_batch":
                    results = _run_async(
                        backend.transcribe_batch(
                            list(request["audio_chunks"]),
                            model_id=model_id,
                            backend=backend.loaded_backend or "transformers",
                            language=str(request["language"]),
                            context=str(request["context"]),
                            timestamps=str(request.get("timestamps", "none")),
                            max_new_tokens=_optional_int(request.get("max_new_tokens")),
                        )
                    )
                    conn.send({"id": request_id, "ok": True, "result": [_result_to_payload(item) for item in results]})
                    continue
                if op == "shutdown":
                    _run_async(backend.unload(cuda_empty_cache=bool(request.get("cuda_empty_cache", True))))
                    conn.send({"id": request_id, "ok": True, "result": None})
                    return
                raise AsrError(400, "bad_request", f"unknown Qwen worker operation: {op}")
            except AsrError as exc:
                conn.send({"id": request_id, "ok": False, "error": _error_to_payload(exc), "diagnostics": safe_exception(exc)})
            except Exception as exc:
                error = AsrError(
                    503,
                    "inference_failed",
                    "Qwen worker operation failed",
                    {"operation": str(op), "error_type": type(exc).__name__},
                )
                conn.send({"id": request_id, "ok": False, "error": _error_to_payload(error), "diagnostics": safe_exception(exc)})
    except EOFError:
        return
    finally:
        try:
            _run_async(backend.unload(cuda_empty_cache=True))
        finally:
            conn.close()


def _run_async(coro: Any) -> Any:
    import asyncio

    return asyncio.run(coro)


def _result_from_payload(payload: Any) -> TranscriptionResult:
    return result_from_payload(payload, default_timestamp_source="qwen3_forced_aligner")


class QwenAsrAdapter:
    """Qwen3-ASR adapter with GPU work isolated in a child process."""

    def __init__(
        self,
        model_id: str,
        *,
        enable_alignment: bool = False,
        repository: str | None = None,
        revision: str | None = None,
        startup_timeout_seconds: float = 30.0,
        load_timeout_seconds: float = 900.0,
        inference_timeout_seconds: float = 3600.0,
        shutdown_timeout_seconds: float = 10.0,
    ) -> None:
        self.model_id = model_id
        self.loaded_backend: str | None = None
        self._load_timeout_seconds = load_timeout_seconds
        self._inference_timeout_seconds = inference_timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._transport = ProcessRpcTransport(
            name=f"Qwen {model_id}",
            target=_qwen_worker_main,
            target_args=(model_id, enable_alignment, repository, revision),
            startup_timeout_seconds=startup_timeout_seconds,
        )

    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        self._transport.start()
        self._request(
            "load",
            timeout_seconds=self._load_timeout_seconds,
            backend=backend,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )
        self.loaded_backend = backend

    async def unload(self, cuda_empty_cache: bool) -> None:
        self.loaded_backend = None
        self._transport.shutdown(
            timeout_seconds=self._shutdown_timeout_seconds,
            cuda_empty_cache=cuda_empty_cache,
        )

    async def abort(self) -> None:
        self.loaded_backend = None
        self._transport.close(force=True)

    async def align_text(
        self, audio: AudioInput, *, text: str, language: str = "auto", timestamps: str = "auto",
    ) -> list[TimestampItem]:
        payload = self._request(
            "align_text", timeout_seconds=self._inference_timeout_seconds,
            audio=audio, text=text, language=language, timestamps=timestamps,
        )
        return [TimestampItem(**item) for item in payload]

    async def transcribe(
        self,
        audio: AudioInput,
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        timestamps: str = "none",
        max_new_tokens: int | None = None,
    ) -> TranscriptionResult:
        del model_id, backend
        payload = self._request(
            "transcribe",
            timeout_seconds=self._inference_timeout_seconds,
            audio=audio,
            language=language,
            context=context,
            timestamps=timestamps,
            max_new_tokens=max_new_tokens,
        )
        return _result_from_payload(payload)

    async def transcribe_batch(
        self,
        audio_chunks: list[AudioInput],
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        timestamps: str = "none",
        max_new_tokens: int | None = None,
    ) -> list[TranscriptionResult]:
        del model_id, backend
        payload = self._request(
            "transcribe_batch",
            timeout_seconds=self._inference_timeout_seconds,
            audio_chunks=audio_chunks,
            language=language,
            context=context,
            timestamps=timestamps,
            max_new_tokens=max_new_tokens,
        )
        return [_result_from_payload(item) for item in payload]

    def _request(self, op: str, *, timeout_seconds: float, **payload: object) -> Any:
        return self._transport.request(op, timeout_seconds=timeout_seconds, **payload)


class _QwenWorkerBackend:
    """In-process Qwen3-ASR backend used only inside the worker process."""

    def __init__(self, model_id: str, *, enable_alignment: bool = False,
                 repository: str | None = None, revision: str | None = None) -> None:
        self.model_id = model_id
        self._repository = repository or MODEL_REPOS.get(model_id)
        self._revision = revision or MODEL_REVISIONS.get(model_id)
        self.loaded_backend: str | None = None
        self._model: Any | None = None
        self._enable_alignment = enable_alignment
        self._aligner_processor: Any | None = None
        self._aligner_model: Any | None = None

    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        if device not in {"cuda", "cuda:0"}:
            raise AsrError(422, "capability_not_supported", "Qwen HF native adapter requires device=cuda or cuda:0")
        if backend != "transformers":
            raise AsrError(422, "capability_not_supported", f"unsupported Qwen backend: {backend}")
        torch = self._assert_cuda_torch()
        torch_dtype = self._resolve_torch_dtype(torch, dtype)
        processor_cls, model_cls = self._import_hf_native_transformers()
        repo_id, revision = self._repository, self._revision
        if not repo_id or not revision:
            raise AsrError(404, "model_not_found", "Qwen model source is not configured")
        if self._model is not None:
            await self.unload(cuda_empty_cache=True)
        try:
            processor = processor_cls.from_pretrained(repo_id, revision=revision)
            model = model_cls.from_pretrained(repo_id, revision=revision, dtype=torch_dtype).to(device).eval()
            self._model = _HfNativeQwenModel(
                processor=processor,
                model=model,
                torch=torch,
                max_new_tokens=max_new_tokens,
            )
        except AsrError:
            raise
        except Exception as exc:
            raise self._map_qwen_exception(exc, phase="load", model_id=repo_id) from exc
        self.loaded_backend = backend

    async def unload(self, cuda_empty_cache: bool) -> None:
        model = self._model
        self._model = None
        aligner_model = self._aligner_model
        self._aligner_model = None
        self._aligner_processor = None
        self.loaded_backend = None
        if model is not None:
            self._move_model_to_cpu(model)
            self._close_model(model)
            del model
        if aligner_model is not None:
            self._move_model_to_cpu(aligner_model)
            self._close_model(aligner_model)
            del aligner_model
        gc.collect()
        if cuda_empty_cache:
            self._release_cuda_cache()

    async def align_text(
        self, audio: AudioInput, *, text: str, language: str = "auto", timestamps: str = "auto",
    ) -> list[TimestampItem]:
        if not text.strip():
            return []
        if language == "auto":
            # The aligner expects a language, not its ASR auto-detection
            # sentinel. An edited Chinese-first document can be inferred from
            # its supplied script without generating replacement text.
            language = "Chinese" if any("\u4e00" <= char <= "\u9fff" for char in text) else "English"
        with materialized_audio_path(audio, suffix=".wav") as audio_path:
            items, _warnings = self._align(str(audio_path), text, language, requested_granularity=timestamps)
        return items

    async def transcribe(
        self,
        audio: AudioInput,
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        timestamps: str = "none",
        max_new_tokens: int | None = None,
    ) -> TranscriptionResult:
        del model_id, backend
        model = self._model
        if model is None:
            raise AsrError(409, "model_loading", "Qwen model is not loaded")
        with materialized_audio_path(audio, suffix=".wav") as audio_path:
            qwen_language = None if language == "auto" else language
            inference_started = perf_counter()
            transcribe_kwargs: dict[str, Any] = {"audio": str(audio_path), "language": qwen_language}
            if context:
                transcribe_kwargs["context"] = context
            if max_new_tokens is not None:
                transcribe_kwargs["max_new_tokens"] = max_new_tokens
            try:
                results = self._transcribe_model(model, transcribe_kwargs)
            except AsrError:
                raise
            except Exception as exc:
                raise self._map_qwen_exception(exc, phase="inference", model_id=self.model_id) from exc
            inference_ms = (perf_counter() - inference_started) * 1000
            postprocess_started = perf_counter()
            first = self._first_result(results)
            text = self._result_text(first)
            if not text.strip():
                raise AsrError(422, "empty_transcription", "Qwen returned an empty transcription result")
            detected_language = self._result_language(first) or ("zh" if language == "auto" else language)
            timestamp_items: list[TimestampItem] = []
            alignment_warnings: list[str] = []
            if timestamps != "none":
                timestamp_items, alignment_warnings = self._align(
                    str(audio_path),
                    text,
                    detected_language,
                    requested_granularity=timestamps,
                )
            postprocess_ms = (perf_counter() - postprocess_started) * 1000
            return TranscriptionResult(
                text=text,
                duration=audio_duration_seconds(audio),
                language=detected_language,
                warnings=[*self._result_warnings(first), *alignment_warnings],
                timestamps=timestamp_items,
                timings=TranscriptionTimings(
                    inference_ms=inference_ms,
                    postprocess_ms=postprocess_ms,
                ),
                generation=self._result_generation(first),
                context=ContextApplication(
                    requested=bool(context),
                    applied=bool(context),
                    method="qwen3_asr_system_prompt" if context else None,
                    input_chars=len(context),
                    used_chars=len(context),
                ),
            )
    async def transcribe_batch(
        self,
        audio_chunks: list[AudioInput],
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        timestamps: str = "none",
        max_new_tokens: int | None = None,
    ) -> list[TranscriptionResult]:
        del model_id, backend
        model = self._model
        if model is None:
            raise AsrError(409, "model_loading", "Qwen model is not loaded")
        if timestamps != "none":
            raise AsrError(422, "capability_not_supported", "timestamp alignment requires single-chunk adapter calls")
        # The HF helper does not provide a proven GPU-safe batch path here.
        # Process one descriptor at a time so only one bounded chunk is materialized.
        results = []
        inference_started = perf_counter()
        try:
            for audio in audio_chunks:
                with materialized_audio_path(audio, suffix=".wav") as audio_path:
                    qwen_language = None if language == "auto" else language
                    results.append(
                        self._transcribe_model(
                            model,
                            {
                                "audio": str(audio_path),
                                "language": qwen_language,
                                "context": context,
                                "max_new_tokens": max_new_tokens,
                            },
                        )[0]
                    )
        except AsrError:
            raise
        except Exception as exc:
            raise self._map_qwen_exception(exc, phase="inference", model_id=self.model_id) from exc
        inference_ms = (perf_counter() - inference_started) * 1000
        if len(results) != len(audio_chunks):
            raise AsrError(
                503,
                "inference_failed",
                "Qwen batch transcription returned an unexpected result count",
                {"expected": len(audio_chunks), "actual": len(results)},
            )
        postprocess_started = perf_counter()
        transcriptions = [
                TranscriptionResult(
                    text=self._non_empty_text(result),
                    duration=audio_duration_seconds(audio),
                    language=self._result_language(result) or ("zh" if language == "auto" else language),
                    warnings=self._result_warnings(result),
                    generation=self._result_generation(result),
                    context=ContextApplication(
                        requested=bool(context),
                        applied=bool(context),
                        method="qwen3_asr_system_prompt" if context else None,
                        input_chars=len(context),
                        used_chars=len(context),
                    ),
                    timings=TranscriptionTimings(
                        inference_ms=inference_ms / len(audio_chunks) if audio_chunks else 0.0,
                        postprocess_ms=0.0,
                    ),
                )
                for audio, result in zip(audio_chunks, results, strict=True)
        ]
        postprocess_ms = (perf_counter() - postprocess_started) * 1000
        if transcriptions:
            per_chunk_postprocess_ms = postprocess_ms / len(transcriptions)
            transcriptions = [
                    TranscriptionResult(
                        text=result.text,
                        duration=result.duration,
                        language=result.language,
                        warnings=result.warnings,
                        timestamps=result.timestamps,
                        generation=result.generation,
                        context=result.context,
                        timings=TranscriptionTimings(
                            inference_ms=result.timings.inference_ms,
                            postprocess_ms=per_chunk_postprocess_ms,
                        ),
                    )
                    for result in transcriptions
            ]
        return transcriptions

    def _import_hf_native_transformers(self) -> tuple[Any, Any]:
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise AsrError(
                503,
                "gpu_unavailable",
                "transformers is not installed; install HF native dependencies in WSL after CUDA torch validation",
            ) from exc
        processor_cls = getattr(transformers, "AutoProcessor", None)
        model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
        if processor_cls is None or model_cls is None:
            raise AsrError(
                503,
                "model_dependency_unavailable",
                "installed transformers does not provide Qwen3-ASR HF native classes; install a newer release or transformers main",
                {
                    "missing": [
                        name
                        for name, value in (
                            ("AutoProcessor", processor_cls),
                            ("AutoModelForMultimodalLM", model_cls),
                        )
                        if value is None
                    ]
                },
            )
        return processor_cls, model_cls

    def _assert_cuda_torch(self) -> Any:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise AsrError(503, "gpu_unavailable", "torch is not installed") from exc
        if torch.version.cuda is None:
            raise AsrError(503, "gpu_unavailable", "installed torch is CPU-only")
        if not torch.cuda.is_available():
            raise AsrError(503, "gpu_unavailable", "torch cannot access CUDA")
        return torch

    def _resolve_torch_dtype(self, torch: Any, dtype: str) -> Any:
        normalized = dtype.lower()
        if normalized == "auto" or normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float16", "fp16"}:
            return torch.float16
        raise AsrError(
            422,
            "capability_not_supported",
            "Qwen HF native adapter supports dtype=auto, bfloat16/bf16, or float16/fp16",
            {"dtype": dtype},
        )

    def _transcribe_model(self, model: Any, transcribe_kwargs: dict[str, Any]) -> Any:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError:
            return model.transcribe(**transcribe_kwargs)
        inference_mode = getattr(torch, "inference_mode", None)
        if not callable(inference_mode):
            return model.transcribe(**transcribe_kwargs)
        try:
            inference_context = inference_mode()
        except Exception as exc:
            logger.warning("torch.inference_mode setup failed during Qwen transcribe: %s", exc)
            return model.transcribe(**transcribe_kwargs)
        with inference_context:
            return model.transcribe(**transcribe_kwargs)

    def _close_model(self, model: Any) -> None:
        for method_name in ("close", "shutdown", "destroy", "cleanup"):
            cleanup = getattr(model, method_name, None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as exc:
                    logger.warning("Qwen model cleanup method %s failed during unload: %s", method_name, exc)

    def _move_model_to_cpu(self, model: Any) -> None:
        candidates = [
            model,
            getattr(model, "model", None),
            getattr(model, "forced_aligner", None),
            getattr(getattr(model, "forced_aligner", None), "model", None),
        ]
        seen: set[int] = set()
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            to_device = getattr(candidate, "to", None)
            if callable(to_device):
                try:
                    to_device("cpu")
                except Exception as exc:
                    logger.warning("moving Qwen model component to CPU failed during unload: %s", exc)

    def _release_cuda_cache(self) -> None:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError:
            return
        cuda = torch.cuda
        if not self._cuda_is_available(cuda):
            return
        self._run_cuda_cleanup("synchronize", cuda.synchronize)
        self._run_cuda_cleanup("empty_cache", cuda.empty_cache)
        ipc_collect = getattr(cuda, "ipc_collect", None)
        if callable(ipc_collect):
            self._run_cuda_cleanup("ipc_collect", ipc_collect)

    def _cuda_is_available(self, cuda: Any) -> bool:
        try:
            return bool(cuda.is_available())
        except Exception as exc:
            logger.warning("torch.cuda.is_available failed during unload: %s", exc)
            return False

    def _run_cuda_cleanup(self, name: str, cleanup: Callable[[], object]) -> None:
        try:
            cleanup()
        except Exception as exc:
            logger.warning("torch.cuda.%s failed during unload: %s", name, exc)

    def _result_text(self, result: Any) -> str:
        text = getattr(result, "text", "")
        if isinstance(text, str):
            return text
        return str(text)

    def _non_empty_text(self, result: Any) -> str:
        text = self._result_text(result)
        if not text.strip():
            raise AsrError(422, "empty_transcription", "Qwen returned an empty transcription result")
        return text

    def _result_language(self, result: Any) -> str:
        language = getattr(result, "language", "")
        if isinstance(language, str):
            return language
        return str(language)

    def _result_warnings(self, result: Any) -> list[str]:
        warnings = getattr(result, "warnings", [])
        if not isinstance(warnings, list):
            return [str(warnings)]
        return [str(warning) for warning in warnings]

    def _result_generation(self, result: Any) -> GenerationMetrics:
        generation = getattr(result, "generation", None)
        return generation if isinstance(generation, GenerationMetrics) else GenerationMetrics()

    def _align(
        self,
        audio_path: str,
        transcript: str,
        language: str,
        *,
        requested_granularity: str,
    ) -> tuple[list[TimestampItem], list[str]]:
        if not self._enable_alignment:
            raise AsrError(
                422,
                "capability_not_supported",
                "Qwen forced alignment is not enabled in this deployment",
            )
        self._ensure_aligner_loaded()
        processor = self._aligner_processor
        model = self._aligner_model
        if processor is None or model is None:
            raise AsrError(503, "model_dependency_unavailable", "Qwen forced aligner failed to load")
        try:
            inputs, word_lists = processor.prepare_forced_aligner_inputs(
                audio=audio_path,
                transcript=transcript,
                language=language,
            )
            inputs = inputs.to(model.device, model.dtype)
            inference_mode = getattr(self._assert_cuda_torch(), "inference_mode")
            with inference_mode():
                outputs = model(**inputs)
            aligned_batches = processor.decode_forced_alignment(
                logits=outputs.logits,
                input_ids=inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=model.config.timestamp_token_id,
            )
            aligned = aligned_batches[0]
        except AsrError:
            raise
        except Exception as exc:
            raise AsrError(
                422,
                "alignment_failed",
                "Qwen forced alignment failed",
                {"error_type": type(exc).__name__, "message": str(exc)[-500:]},
            ) from exc
        if not aligned:
            raise AsrError(422, "alignment_failed", "Qwen forced aligner returned no timestamp items")
        actual_granularity = "char" if _is_character_alignment(aligned) else "word"
        warnings = []
        if requested_granularity not in {"auto", actual_granularity}:
            warnings.append(
                f"alignment_granularity_adjusted:requested={requested_granularity}:actual={actual_granularity}"
            )
        items: list[TimestampItem] = []
        previous_end = 0.0
        for raw in aligned:
            text = str(raw.get("text", "")).strip()
            start = float(raw["start_time"])
            end = float(raw["end_time"])
            if not text or start < 0 or end < start or start + 0.001 < previous_end:
                raise AsrError(
                    422,
                    "alignment_failed",
                    "Qwen forced aligner returned invalid or non-monotonic timestamps",
                )
            items.append(TimestampItem(text=text, start=start, end=end, granularity=actual_granularity))
            previous_end = end
        return items, warnings

    def _ensure_aligner_loaded(self) -> None:
        if self._aligner_model is not None and self._aligner_processor is not None:
            return
        try:
            transformers = importlib.import_module("transformers")
            processor_cls = getattr(transformers, "AutoProcessor")
            model_cls = getattr(transformers, "AutoModelForTokenClassification")
            processor = processor_cls.from_pretrained(ALIGNER_REPO, revision=ALIGNER_REVISION)
            asr_wrapper = self._model
            if asr_wrapper is None:
                raise RuntimeError("ASR model is not loaded")
            device = asr_wrapper.model.device
            dtype = asr_wrapper.model.dtype
            model = model_cls.from_pretrained(ALIGNER_REPO, revision=ALIGNER_REVISION, dtype=dtype).to(device).eval()
        except Exception as exc:
            self._release_cuda_cache()
            raise self._map_qwen_exception(exc, phase="alignment_load", model_id=ALIGNER_REPO) from exc
        self._aligner_processor = processor
        self._aligner_model = model

    def _first_result(self, results: Any) -> Any:
        try:
            first = results[0]
        except (IndexError, KeyError, TypeError) as exc:
            raise AsrError(422, "empty_transcription", "Qwen returned no transcription results") from exc
        return first

    def _map_qwen_exception(self, exc: Exception, *, phase: str, model_id: str) -> AsrError:
        message = str(exc)
        lowered = message.lower()
        details = {
            "phase": phase,
            "model": model_id,
            "error_type": type(exc).__name__,
            "message": message[-500:],
        }
        if "out of memory" in lowered or "cuda oom" in lowered:
            return AsrError(503, "gpu_unavailable", "CUDA out of memory during Qwen ASR", details)
        if phase in {"load", "alignment_load"} and any(
            marker in lowered
            for marker in ("download", "connection", "repository", "resolve", "not found", "401", "403", "404")
        ):
            return AsrError(503, "model_download_failed", "Qwen model download or resolution failed", details)
        return AsrError(503, "inference_failed", f"Qwen {phase} failed", details)


@dataclass(frozen=True)
class _QwenTranscriptionItem:
    text: str
    language: str
    warnings: list[str]
    generation: GenerationMetrics


def _qwen_prefill_chat_template(processor: Any) -> str:
    """Preserve native audio rendering while retaining the language prefill.

    The pinned HF checkpoints' templates omit assistant messages, whereas
    apply_transcription_request uses one for its normalized language prefix.
    Render that turn exactly once without mutating the processor or cache.
    """
    template = getattr(processor, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise RuntimeError("Qwen processor requires a text chat template for explicit language transcription")
    return (
        "{%- set asr_prefill_messages = messages -%}"
        "{%- set messages = messages | rejectattr('role', 'equalto', 'assistant') | list -%}"
        + template
        + "{%- set messages = asr_prefill_messages -%}"
        "{%- for m in messages if m.role == 'assistant' -%}"
        "{{- '<|im_start|>assistant\\n' -}}"
        "{%- if m.content is string -%}{{- m.content -}}"
        "{%- else -%}{%- for c in m.content if c.type == 'text' -%}{{- c.text -}}{%- endfor -%}"
        "{%- endif -%}{%- endfor -%}"
    )


class _HfNativeQwenModel:
    """Small adapter around the Qwen3-ASR Transformers native helper API."""

    def __init__(self, *, processor: Any, model: Any, torch: Any, max_new_tokens: int | None) -> None:
        self.processor = processor
        self.model = model
        self.torch = torch
        self.max_new_tokens = max_new_tokens

    def transcribe(self, **kwargs: object) -> list[_QwenTranscriptionItem]:
        audio = kwargs.get("audio")
        if not isinstance(audio, str):
            raise TypeError("HF native Qwen transcription expects one audio path")
        language_value = kwargs.get("language")
        language = language_value if isinstance(language_value, str) else None
        request_max_new_tokens = kwargs.get("max_new_tokens")
        max_new_tokens = request_max_new_tokens if isinstance(request_max_new_tokens, int) else self.max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = 512
        context = kwargs.get("context")
        warnings: list[str] = []

        apply_request = getattr(self.processor, "apply_transcription_request", None)
        if not callable(apply_request):
            raise RuntimeError("processor.apply_transcription_request is unavailable")
        request_kwargs: dict[str, object] = {"audio": audio, "language": language}
        if isinstance(context, str) and context:
            try:
                supports_prompt = "prompt" in inspect.signature(apply_request).parameters
            except (TypeError, ValueError):
                supports_prompt = False
            if not supports_prompt:
                raise AsrError(
                    422,
                    "context_not_supported",
                    "installed transformers Qwen helper does not support prompt; install the pinned runtime version",
                )
            request_kwargs["prompt"] = context
        if language is None:
            messages: list[dict[str, object]] = []
            if isinstance(context, str) and context:
                messages.append(
                    {"role": "system", "content": [{"type": "text", "text": context}]}
                )
            messages.append(
                {"role": "user", "content": [{"type": "audio", "audio": audio}]}
            )
            apply_chat_template = getattr(self.processor, "apply_chat_template", None)
            if not callable(apply_chat_template):
                raise RuntimeError("processor.apply_chat_template is unavailable for automatic language detection")
            inputs = apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            )
            warnings.append("qwen_auto_language_uses_chat_template_generation_prompt")
        else:
            request_kwargs["chat_template"] = _qwen_prefill_chat_template(self.processor)
            inputs = apply_request(**request_kwargs)
        inputs = inputs.to(self.model.device, self.model.dtype)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        prompt_tokens = int(inputs["input_ids"].shape[1])
        generated_tokens = _sequence_length(generated_ids)
        reached_limit = generated_tokens >= max_new_tokens
        ended_with_eos = _ends_with_eos(generated_ids, self.model)
        if reached_limit and not ended_with_eos:
            raise AsrError(
                422,
                "generation_truncated",
                "Qwen generation reached max_new_tokens before EOS; transcript may be incomplete",
                {"generated_tokens": generated_tokens, "max_new_tokens": max_new_tokens},
            )
        generation = GenerationMetrics(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            max_new_tokens=max_new_tokens,
            truncated=False,
            finish_reason="eos" if ended_with_eos else "completed",
        )
        parsed_items = self.processor.decode(generated_ids, return_format="parsed")
        return [self._item_from_parsed(item, warnings=warnings, generation=generation) for item in parsed_items]

    def _item_from_parsed(
        self,
        parsed: object,
        *,
        warnings: list[str],
        generation: GenerationMetrics,
    ) -> _QwenTranscriptionItem:
        if isinstance(parsed, dict):
            text = parsed.get("transcription", parsed.get("text", ""))
            language = parsed.get("language", "")
            return _QwenTranscriptionItem(
                text=text if isinstance(text, str) else str(text),
                language=language if isinstance(language, str) else "" if language is None else str(language),
                warnings=warnings,
                generation=generation,
            )
        return _QwenTranscriptionItem(text=str(parsed), language="", warnings=warnings, generation=generation)


def _sequence_length(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    try:
        return len(value[0])
    except (IndexError, KeyError, TypeError):
        try:
            return len(value)
        except TypeError:
            return 0


def _ends_with_eos(generated_ids: Any, model: Any) -> bool:
    if _sequence_length(generated_ids) == 0:
        return False
    config = getattr(model, "generation_config", None)
    eos = getattr(config, "eos_token_id", None)
    if eos is None:
        eos = getattr(getattr(model, "config", None), "eos_token_id", None)
    eos_ids = {int(item) for item in eos} if isinstance(eos, list | tuple | set) else {int(eos)} if eos is not None else set()
    if not eos_ids:
        return False
    try:
        last = generated_ids[0, -1]
    except (IndexError, KeyError, TypeError):
        try:
            last = generated_ids[0][-1]
        except (IndexError, KeyError, TypeError):
            return False
    item = getattr(last, "item", None)
    value = item() if callable(item) else last
    return int(value) in eos_ids


def _is_character_alignment(items: list[dict[str, object]]) -> bool:
    texts = [str(item.get("text", "")) for item in items]
    cjk = sum(any(_is_cjk(character) for character in text) for text in texts)
    return bool(texts) and cjk >= max(1, len(texts) // 2)


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )
