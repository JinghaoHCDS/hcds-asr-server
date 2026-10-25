from __future__ import annotations

import gc
import importlib
import logging
import os
import signal
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from time import perf_counter
from typing import Any

from asr_server.adapters.base import (
    AudioInput,
    ContextApplication,
    GenerationMetrics,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionTimings,
    TimestampItem,
)
from asr_server.audio.metadata import inspect_audio_path
from asr_server.audio.splitter import _energy_intervals_path
from asr_server.errors import AsrError
from asr_server.observability import PROTOCOL_VERSION, safe_exception
from asr_server.registry import MOSS_MODEL_ID, MODEL_SOURCES
from asr_server.workers.transport import ProcessRpcTransport
from asr_server.workers.codec import (
    optional_int as _optional_int, result_to_payload as _result_to_payload,
    result_from_payload as _result_from_payload, error_to_payload as _error_to_payload, validate_request_envelope,
)
from asr_server.workers.audio import audio_duration_seconds, materialized_audio_path


MODEL_REPOS = {MOSS_MODEL_ID: MODEL_SOURCES[MOSS_MODEL_ID][0]}
MODEL_REVISIONS = {MOSS_MODEL_ID: MODEL_SOURCES[MOSS_MODEL_ID][1]}
DEFAULT_MOSS_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
logger = logging.getLogger(__name__)


def _moss_worker_main(conn: Connection, model_id: str, repository: str | None = None,
                      revision: str | None = None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    backend = _MossWorkerBackend(model_id, repository=repository, revision=revision)
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
                raise AsrError(400, "bad_request", f"unknown MOSS worker operation: {op}")
            except AsrError as exc:
                conn.send({"id": request_id, "ok": False, "error": _error_to_payload(exc), "diagnostics": safe_exception(exc)})
            except Exception as exc:
                error = AsrError(
                    503,
                    "inference_failed",
                    "MOSS worker operation failed",
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


class MossTranscribeDiarizeAdapter:
    """MOSS-Transcribe-Diarize adapter with GPU work isolated in a child process."""

    def __init__(
        self,
        model_id: str,
        *,
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
            name=f"MOSS {model_id}",
            target=_moss_worker_main,
            target_args=(model_id, repository, revision),
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


@dataclass(frozen=True)
class _MossHelpers:
    default_prompt: str
    build_transcription_messages: Any
    generate_transcription: Any
    parse_transcript: Any


class _MossWorkerBackend:
    """In-process MOSS backend used only inside the worker process."""

    def __init__(self, model_id: str, *, repository: str | None = None, revision: str | None = None) -> None:
        self.model_id = model_id
        self._repository = repository or MODEL_REPOS.get(model_id)
        self._revision = revision or MODEL_REVISIONS.get(model_id)
        self.loaded_backend: str | None = None
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None
        self._dtype: Any | None = None
        self._helpers: _MossHelpers | None = None
        self._max_new_tokens: int | None = None

    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        if not self._repository or not self._revision:
            raise AsrError(404, "model_not_found", f"unknown MOSS model: {self.model_id}")
        if device not in {"cuda", "cuda:0"}:
            raise AsrError(422, "capability_not_supported", "MOSS adapter requires device=cuda or cuda:0")
        if backend != "transformers":
            raise AsrError(422, "capability_not_supported", f"unsupported MOSS backend: {backend}")
        torch = self._assert_cuda_torch()
        torch_dtype = self._resolve_torch_dtype(torch, dtype)
        torch_device = torch.device(device)
        processor_cls, model_cls = self._import_transformers()
        helpers = self._import_moss_helpers()
        repo_id, revision = self._repository, self._revision
        if self._model is not None:
            await self.unload(cuda_empty_cache=True)
        try:
            model = (
                # Official Transformers recipe recommends SDPA/FlashAttention;
                # eager attention has quadratic memory cost on long recordings.
                # This pinned remote-code revision declares _supports_sdpa.
                model_cls.from_pretrained(repo_id, revision=revision, trust_remote_code=True, dtype="auto",
                                          attn_implementation="sdpa")
                .to(dtype=torch_dtype)
                .to(torch_device)
                .eval()
            )
            processor = processor_cls.from_pretrained(repo_id, revision=revision, trust_remote_code=True)
            self._model = model
            self._processor = processor
            self._torch = torch
            self._device = torch_device
            self._dtype = torch_dtype
            self._helpers = helpers
            self._max_new_tokens = max_new_tokens
        except AsrError:
            raise
        except Exception as exc:
            raise self._map_moss_exception(exc, phase="load", model_id=repo_id) from exc
        self.loaded_backend = backend

    async def unload(self, cuda_empty_cache: bool) -> None:
        model = self._model
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None
        self._dtype = None
        self._helpers = None
        self._max_new_tokens = None
        self.loaded_backend = None
        if model is not None:
            self._move_model_to_cpu(model)
            self._close_model(model)
            del model
        gc.collect()
        if cuda_empty_cache:
            self._release_cuda_cache()

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
        del model_id, backend, timestamps
        model = self._model
        processor = self._processor
        helpers = self._helpers
        if model is None or processor is None or helpers is None:
            raise AsrError(409, "model_loading", "MOSS model is not loaded")
        with materialized_audio_path(audio, suffix=".wav") as audio_path:
            effective_max_new_tokens = max_new_tokens or self._max_new_tokens or 2048
            self._reset_peak_memory_stats()
            inference_started = perf_counter()
            try:
                result = helpers.generate_transcription(
                    model,
                    processor,
                    helpers.build_transcription_messages(
                        str(audio_path),
                        prompt=self._prompt_for_request(context, language, helpers.default_prompt),
                    ),
                    max_new_tokens=effective_max_new_tokens,
                    do_sample=False,
                    device=self._device,
                    dtype=self._dtype,
                )
            except Exception as exc:
                self._release_cuda_cache()
                raise self._map_moss_exception(exc, phase="inference", model_id=self.model_id) from exc
            inference_ms = (perf_counter() - inference_started) * 1000
            try:
                prompt_tokens = self._result_int(result, "prompt_len")
                generated_tokens = self._result_int(result, "generated_tokens")
                peak_vram_allocated_mb = self._peak_vram_allocated_mb()
                raw_text = self._result_text(result)
            finally:
                del result
                # Keep the loaded model, but return unused KV/output buffers to
                # CUDA between long-form invocations and after controlled 422s.
                self._release_cuda_cache()
            if generated_tokens is not None and generated_tokens >= effective_max_new_tokens:
                raise AsrError(
                    422,
                    "generation_truncated",
                    "MOSS generation reached max_new_tokens; the transcript may be incomplete",
                    {
                        "generated_tokens": generated_tokens,
                        "max_new_tokens": effective_max_new_tokens,
                        "recommended_action": "retry with split_strategy=fixed or a larger validated model limit",
                    },
                )
            postprocess_started = perf_counter()
            if not raw_text.strip():
                raise AsrError(422, "empty_transcription", "MOSS returned an empty transcription result")
            segments, parse_warnings = self._parse_segments(raw_text, helpers, audio_duration_seconds(audio))
            text = "\n".join(segment.text for segment in segments) if segments else raw_text
            duration = audio_duration_seconds(audio)
            coverage_end = max((segment.end for segment in segments), default=None)
            coverage_ratio = min(coverage_end / duration, 1.0) if coverage_end is not None and duration > 0 else None
            speech_intervals = _energy_intervals_path(
                audio_path,
                inspect_audio_path(audio_path),
                padding_seconds=0.0,
            )
            audible_end, uncovered_speech = self._validate_speech_coverage(
                duration=duration,
                coverage_end=coverage_end,
                segments=segments,
                speech_intervals=speech_intervals,
            )
            postprocess_ms = (perf_counter() - postprocess_started) * 1000
            return TranscriptionResult(
                text=text,
                duration=duration,
                language=language,
                warnings=parse_warnings,
                segments=segments,
                timings=TranscriptionTimings(
                    inference_ms=inference_ms,
                    postprocess_ms=postprocess_ms,
                ),
                generation=GenerationMetrics(
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated_tokens,
                    max_new_tokens=effective_max_new_tokens,
                    peak_vram_allocated_mb=peak_vram_allocated_mb,
                    segment_coverage_end_seconds=coverage_end,
                    segment_coverage_ratio=coverage_ratio,
                    audible_speech_end_seconds=audible_end,
                    uncovered_speech_intervals=uncovered_speech,
                ),
                context=ContextApplication(
                    requested=bool(context),
                    applied=bool(context),
                    method="moss_transcription_prompt" if context else None,
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
        results = []
        for audio in audio_chunks:
            results.append(
                await self.transcribe(
                    audio,
                    model_id=model_id,
                    backend=backend,
                    language=language,
                    context=context,
                    timestamps=timestamps,
                    max_new_tokens=max_new_tokens,
                )
            )
        return results

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
            "MOSS adapter supports dtype=auto, bfloat16/bf16, or float16/fp16",
            {"dtype": dtype},
        )

    def _import_transformers(self) -> tuple[Any, Any]:
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as exc:
            raise AsrError(
                503,
                "model_dependency_unavailable",
                "transformers is not installed; install MOSS dependencies in WSL after CUDA torch validation",
            ) from exc
        processor_cls = getattr(transformers, "AutoProcessor", None)
        model_cls = getattr(transformers, "AutoModelForCausalLM", None)
        if processor_cls is None or model_cls is None:
            raise AsrError(
                503,
                "model_dependency_unavailable",
                "installed transformers does not provide MOSS loading classes",
                {
                    "missing": [
                        name
                        for name, value in (
                            ("AutoProcessor", processor_cls),
                            ("AutoModelForCausalLM", model_cls),
                        )
                        if value is None
                    ]
                },
            )
        return processor_cls, model_cls

    def _import_moss_helpers(self) -> _MossHelpers:
        try:
            moss_package = importlib.import_module("moss_transcribe_diarize")
            inference_utils = importlib.import_module("moss_transcribe_diarize.inference_utils")
        except ModuleNotFoundError as exc:
            raise AsrError(
                503,
                "model_dependency_unavailable",
                "moss-transcribe-diarize is not installed in this WSL environment",
            ) from exc
        parse_transcript = getattr(moss_package, "parse_transcript", None)
        build_messages = getattr(inference_utils, "build_transcription_messages", None)
        generate_transcription = getattr(inference_utils, "generate_transcription", None)
        if not callable(parse_transcript) or not callable(build_messages) or not callable(generate_transcription):
            raise AsrError(
                503,
                "model_dependency_unavailable",
                "installed moss-transcribe-diarize package is missing required inference helpers",
                {
                    "missing": [
                        name
                        for name, value in (
                            ("parse_transcript", parse_transcript),
                            ("build_transcription_messages", build_messages),
                            ("generate_transcription", generate_transcription),
                        )
                        if not callable(value)
                    ]
                },
            )
        return _MossHelpers(
            default_prompt=str(getattr(inference_utils, "DEFAULT_PROMPT", DEFAULT_MOSS_PROMPT)),
            build_transcription_messages=build_messages,
            generate_transcription=generate_transcription,
            parse_transcript=parse_transcript,
        )

    def _prompt_for_request(self, context: str, language: str, default_prompt: str) -> str:
        language_instruction = {
            "zh": "请仅使用中文转写。",
            "en": "Transcribe in English only.",
        }.get(language, "")
        stripped = context.strip()
        return "\n".join(part for part in (default_prompt, language_instruction, stripped) if part)

    def _result_text(self, result: object) -> str:
        if isinstance(result, dict):
            text = result.get("text", "")
            return text if isinstance(text, str) else str(text)
        text = getattr(result, "text", "")
        return text if isinstance(text, str) else str(text)

    def _result_int(self, result: object, key: str) -> int | None:
        value = result.get(key) if isinstance(result, dict) else getattr(result, key, None)
        return value if isinstance(value, int) else None

    def _validate_tail_coverage(self, *, duration: float, coverage_end: float | None) -> None:
        self._validate_speech_coverage(
            duration=duration,
            coverage_end=coverage_end,
            segments=[],
            speech_intervals=[],
        )

    def _validate_speech_coverage(
        self,
        *,
        duration: float,
        coverage_end: float | None,
        segments: list[TranscriptionSegment],
        speech_intervals: list[tuple[float, float]],
    ) -> tuple[float | None, int]:
        if coverage_end is None or duration <= 0:
            return (speech_intervals[-1][1] if speech_intervals else None), len(speech_intervals)
        audible_end = speech_intervals[-1][1] if speech_intervals else duration
        uncovered_tail = max(audible_end - coverage_end, 0.0)
        allowed_tail = max(60.0, duration * 0.02)
        uncovered_intervals = (
            [
                (start, end)
                for start, end in speech_intervals
                if not any(segment.start < end and segment.end > start for segment in segments)
            ]
            if segments
            else []
        )
        if uncovered_tail > allowed_tail or uncovered_intervals:
            raise AsrError(
                422,
                "incomplete_transcript",
                "MOSS output did not cover candidate audible intervals",
                {
                    "audio_duration_seconds": duration,
                    "audible_speech_end_seconds": audible_end,
                    "last_segment_end_seconds": coverage_end,
                    "uncovered_tail_seconds": uncovered_tail,
                    "allowed_tail_seconds": allowed_tail,
                    "uncovered_speech_intervals": [
                        {"start": start, "end": end} for start, end in uncovered_intervals[:20]
                    ],
                    "uncovered_speech_interval_count": len(uncovered_intervals),
                    "coverage_evidence": "energy_activity_not_speech_classification",
                    "coverage_is_not_accuracy": True,
                    "recommended_action": "retry with split_strategy=fixed and a validated chunk duration",
                },
            )
        return audible_end, 0

    def _reset_peak_memory_stats(self) -> None:
        cuda = getattr(self._torch, "cuda", None)
        reset = getattr(cuda, "reset_peak_memory_stats", None)
        if callable(reset):
            try:
                reset(self._device)
            except Exception as exc:
                logger.warning("resetting MOSS peak CUDA memory stats failed: %s", exc)

    def _peak_vram_allocated_mb(self) -> float | None:
        cuda = getattr(self._torch, "cuda", None)
        peak = getattr(cuda, "max_memory_allocated", None)
        if not callable(peak):
            return None
        try:
            return float(peak(self._device)) / 1024 / 1024
        except Exception as exc:
            logger.warning("reading MOSS peak CUDA memory stats failed: %s", exc)
            return None

    def _parse_segments(
        self,
        text: str,
        helpers: _MossHelpers,
        chunk_duration: float,
    ) -> tuple[list[TranscriptionSegment], list[str]]:
        try:
            parsed = helpers.parse_transcript(text)
        except Exception as exc:
            return [], [f"moss_segment_parser_failed:{type(exc).__name__}"]
        if not isinstance(parsed, list | tuple):
            return [], ["moss_segment_parser_invalid_structure"]
        segments: list[TranscriptionSegment] = []
        warnings: list[str] = []
        for raw_segment in parsed:
            try:
                segment = self._segment_from_parsed(raw_segment)
            except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                warnings.append("moss_segment_invalid_structure")
                continue
            normalized, warning = self._normalize_segment(segment, chunk_duration)
            if warning is not None:
                warnings.append(warning)
            if normalized is not None:
                segments.append(normalized)
        segments.sort(key=lambda item: (item.start, item.end))
        return segments, list(dict.fromkeys(warnings))

    def _normalize_segment(
        self,
        segment: TranscriptionSegment,
        chunk_duration: float,
    ) -> tuple[TranscriptionSegment | None, str | None]:
        import math

        if not math.isfinite(segment.start) or not math.isfinite(segment.end):
            return None, "moss_segment_non_finite"
        if segment.start < 0 or segment.end < segment.start:
            return None, "moss_segment_invalid_range"
        epsilon = 0.05
        if segment.start > chunk_duration + epsilon or segment.end > chunk_duration + epsilon:
            return None, "moss_segment_out_of_chunk_bounds"
        start = min(max(segment.start, 0.0), chunk_duration)
        end = min(max(segment.end, start), chunk_duration)
        text = segment.text.strip()
        if not text:
            return None, "moss_segment_empty_text"
        warning = "moss_segment_clamped_to_chunk" if (start, end) != (segment.start, segment.end) else None
        return TranscriptionSegment(start=start, end=end, speaker=segment.speaker, text=text), warning

    def _segment_from_parsed(self, segment: object) -> TranscriptionSegment:
        if isinstance(segment, dict):
            return TranscriptionSegment(
                start=float(segment["start"]),
                end=float(segment["end"]),
                speaker=str(segment["speaker"]) if segment.get("speaker") is not None else None,
                text=str(segment["text"]),
            )
        return TranscriptionSegment(
            start=float(getattr(segment, "start")),
            end=float(getattr(segment, "end")),
            speaker=str(getattr(segment, "speaker")) if getattr(segment, "speaker", None) is not None else None,
            text=str(getattr(segment, "text")),
        )

    def _close_model(self, model: Any) -> None:
        for method_name in ("close", "shutdown", "destroy", "cleanup"):
            cleanup = getattr(model, method_name, None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as exc:
                    logger.warning("MOSS model cleanup method %s failed during unload: %s", method_name, exc)

    def _move_model_to_cpu(self, model: Any) -> None:
        to_device = getattr(model, "to", None)
        if callable(to_device):
            try:
                to_device("cpu")
            except Exception as exc:
                logger.warning("moving MOSS model to CPU failed during unload: %s", exc)

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
            logger.warning("torch.cuda.is_available failed during MOSS unload: %s", exc)
            return False

    def _run_cuda_cleanup(self, name: str, cleanup: Callable[[], object]) -> None:
        try:
            cleanup()
        except Exception as exc:
            logger.warning("torch.cuda.%s failed during MOSS unload: %s", name, exc)

    def _map_moss_exception(self, exc: Exception, *, phase: str, model_id: str) -> AsrError:
        message = str(exc)
        lowered = message.lower()
        details = {
            "phase": phase,
            "model": model_id,
            "error_type": type(exc).__name__,
            "message": message[-500:],
        }
        if phase == "inference" and any(
            marker in lowered
            for marker in (
                "max_position_embeddings",
                "context length",
                "sequence length",
                "input is too long",
                "max_length",
            )
        ):
            return AsrError(
                422,
                "context_length_exceeded",
                "MOSS native long-form input exceeds the model context limit",
                details,
            )
        if "out of memory" in lowered or "cuda oom" in lowered:
            return AsrError(503, "gpu_unavailable", "CUDA out of memory during MOSS ASR", details)
        if phase == "load" and any(
            marker in lowered
            for marker in ("download", "connection", "repository", "resolve", "not found", "401", "403", "404")
        ):
            return AsrError(503, "model_download_failed", "MOSS model download or resolution failed", details)
        return AsrError(503, "inference_failed", f"MOSS {phase} failed", details)
