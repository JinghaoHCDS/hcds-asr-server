"""Versioned local IPC envelope and shared transcription-result codec."""
from __future__ import annotations

from typing import Any

from asr_server.adapters.base import (
    ContextApplication, GenerationMetrics, TimestampItem, TranscriptionResult,
    TranscriptionSegment, TranscriptionTimings,
)
from asr_server.errors import AsrError
from asr_server.observability import PROTOCOL_VERSION, ExecutionContext


def request_envelope(request_id: int, operation: str, payload: dict[str, object], context: ExecutionContext) -> dict[str, object]:
    return {**payload, "id": request_id, "op": operation, "protocol_version": PROTOCOL_VERSION,
            "execution_context": context.to_wire()}


def validate_request_envelope(request: dict[str, Any]) -> None:
    if request.get("protocol_version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise AsrError(503, "worker_unavailable", "incompatible worker protocol version")
    if not isinstance(request.get("id"), int) or not isinstance(request.get("op"), str):
        raise AsrError(503, "worker_unavailable", "malformed worker request envelope")


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected optional int value, got {type(value).__name__}")


def result_to_payload(result: TranscriptionResult) -> dict[str, object]:
    return {
        "text": result.text,
        "duration": result.duration,
        "language": result.language,
        "warnings": result.warnings,
        "segments": [segment.to_api() for segment in result.segments],
        "timestamps": [item.to_api() for item in result.timestamps],
        "timings": {
            "total_ms": result.timings.total_ms,
            "load_ms": result.timings.load_ms,
            "decode_ms": result.timings.decode_ms,
            "inference_ms": result.timings.inference_ms,
            "postprocess_ms": result.timings.postprocess_ms,
        },
        "generation": result.generation.to_api(),
        "context": result.context.to_api(),
    }


def result_from_payload(payload: Any, *, default_timestamp_source: str = "unknown") -> TranscriptionResult:
    timings = payload.get("timings", {})
    generation = payload.get("generation", {})
    context = payload.get("context", {})
    return TranscriptionResult(
        text=str(payload["text"]),
        duration=float(payload["duration"]),
        language=str(payload["language"]),
        warnings=[str(warning) for warning in payload.get("warnings", [])],
        segments=[
            TranscriptionSegment(
                start=float(segment["start"]),
                end=float(segment["end"]),
                speaker=str(segment["speaker"]) if segment.get("speaker") is not None else None,
                text=str(segment["text"]),
                source_speaker=(
                    str(segment["source_speaker"])
                    if segment.get("source_speaker") is not None
                    else None
                ),
                speaker_resolution=(
                    str(segment["speaker_resolution"])
                    if segment.get("speaker_resolution") is not None
                    else None
                ),
            )
            for segment in payload.get("segments", [])
        ],
        timestamps=[
            TimestampItem(
                text=str(item["text"]),
                start=float(item["start"]),
                end=float(item["end"]),
                granularity=str(item["granularity"]),
                source=str(item.get("source", default_timestamp_source)),
                status=str(item.get("status", "aligned")),
            )
            for item in payload.get("timestamps", [])
        ],
        timings=TranscriptionTimings(
            total_ms=float(timings.get("total_ms", 0.0)),
            load_ms=float(timings.get("load_ms", 0.0)),
            decode_ms=float(timings.get("decode_ms", 0.0)),
            inference_ms=float(timings.get("inference_ms", 0.0)),
            postprocess_ms=float(timings.get("postprocess_ms", 0.0)),
        ),
        generation=GenerationMetrics(
            prompt_tokens=optional_int(generation.get("prompt_tokens")),
            generated_tokens=optional_int(generation.get("generated_tokens")),
            max_new_tokens=optional_int(generation.get("max_new_tokens")),
            peak_vram_allocated_mb=optional_float(generation.get("peak_vram_allocated_mb")),
            segment_coverage_end_seconds=optional_float(generation.get("segment_coverage_end_seconds")),
            segment_coverage_ratio=optional_float(generation.get("segment_coverage_ratio")),
            audible_speech_end_seconds=optional_float(generation.get("audible_speech_end_seconds")),
            uncovered_speech_intervals=optional_int(generation.get("uncovered_speech_intervals")),
            truncated=bool(generation.get("truncated", False)),
            finish_reason=str(generation["finish_reason"]) if generation.get("finish_reason") is not None else None,
        ),
        context=ContextApplication(
            requested=bool(context.get("requested", False)),
            applied=bool(context.get("applied", False)),
            method=str(context["method"]) if context.get("method") is not None else None,
            input_chars=int(context.get("input_chars", 0)),
            used_chars=int(context.get("used_chars", 0)),
            hotword_count=int(context.get("hotword_count", 0)),
            truncated=bool(context.get("truncated", False)),
            ignored_reason=str(context["ignored_reason"]) if context.get("ignored_reason") is not None else None,
        ),
    )


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"expected optional float value, got {type(value).__name__}")


def error_to_payload(error: AsrError) -> dict[str, object]:
    return {
        "status_code": error.status_code,
        "code": error.code,
        "message": error.message,
        "details": error.details,
    }


