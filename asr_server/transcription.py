from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from asr_server.adapters.base import AudioInput, AudioPath, TranscriptionResult
from asr_server.audio.merger import ChunkLike, merge_transcription_results
from asr_server.audio.timing import timing_analysis
from asr_server.audio.metadata import inspect_audio, inspect_audio_path
from asr_server.audio.preprocess import NormalizedAudioPath, normalize_audio_path_to_wav_async, normalize_audio_to_wav
from asr_server.audio.assets import NormalizedAudioAsset
from asr_server.observability import current_context, emit
from asr_server.audio.splitter import PathSplitResult, SplitCancelled, SplitResult, split_audio, split_audio_path
from asr_server.config import Settings
from asr_server.context import PreparedContext, prepare_context
from asr_server.audio.workspace import AudioWorkspace, validate_workspace_limits
from asr_server.errors import AsrError
from asr_server.diarization.anchor_replay import AnchorReplaySequence
from asr_server.execution import ExecutionPlan, ModelExecutionPolicy
from asr_server.lifecycle import ModelLifecycleManager
from asr_server.registry import Backend


SYNC_JOB_THRESHOLD_SECONDS = 600.0
# Real four-speaker podcast validation showed that a dense 1,740-second body
# can exhaust MOSS's useful output window after roughly 1,324 seconds even
# without reaching the nominal audio-duration limit. Keep a content-density
# margin while leaving the separate 60-second anchor replay budget intact.
MOSS_ANCHOR_REPLAY_BODY_CHUNK_SECONDS = 1_200.0
MOSS_ANCHOR_REPLAY_MIN_MAX_NEW_TOKENS = 24_000

StageCallback = Callable[[str, dict[str, object]], Awaitable[None]]
BeforeChunkCallback = Callable[[int, int], Awaitable[None]]
AfterChunkCallback = Callable[[int, int, TranscriptionResult], Awaitable[None]]


@dataclass(frozen=True)
class GenerationConfig:
    language: str
    max_new_tokens: int | None
    context: str
    hotwords: str | None


@dataclass(frozen=True)
class RequestConfig:
    response_format: str
    timestamps: str
    split_strategy: str
    max_chunk_seconds: float | None
    overlap_seconds: float | None
    preserve_segments: bool
    speaker_resolution: str


@dataclass(frozen=True)
class TranscriptionRequest:
    model: str | None
    language: str
    response_format: str
    timestamps: str
    backend: Backend
    max_new_tokens: int | None
    context: str
    hotwords: str | None
    split_strategy: str
    max_chunk_seconds: float | None
    overlap_seconds: float | None
    preserve_segments: bool
    speaker_resolution: str = "off"
    purpose: str = "transcription"
    pause_threshold_seconds: float = 0.8
    idempotency_key: str | None = None
    quality_profile: str = "legacy"
    glossary_snapshot: dict[str, object] | None = None
    quality_chunk_seconds: float | None = None
    quality_review_enabled: bool = True

    @property
    def generation_config(self) -> GenerationConfig:
        return GenerationConfig(self.language, self.max_new_tokens, self.context, self.hotwords)

    @property
    def request_config(self) -> RequestConfig:
        return RequestConfig(
            self.response_format,
            self.timestamps,
            self.split_strategy,
            self.max_chunk_seconds,
            self.overlap_seconds,
            self.preserve_segments,
            self.speaker_resolution,
        )


@dataclass(frozen=True)
class ValidatedTranscription:
    selected_model: str
    resolved_backend: str
    adapter_context: str
    max_new_tokens: int | None
    execution_policy: ModelExecutionPolicy
    supports_diarization: bool
    context: PreparedContext


def validate_transcription_request(
    manager: ModelLifecycleManager,
    request: TranscriptionRequest,
) -> ValidatedTranscription:
    if request.quality_profile not in {"legacy", "quality_v1"}:
        raise AsrError(400, "bad_request", "unsupported quality_profile")
    if request.response_format not in {"json", "text", "verbose_json"}:
        raise AsrError(400, "bad_request", f"unsupported response_format: {request.response_format}")
    if request.timestamps not in {"none", "auto", "word", "char"}:
        raise AsrError(400, "bad_request", f"unsupported timestamps value: {request.timestamps}")
    if request.speaker_resolution not in {"off", "auto", "required"}:
        raise AsrError(400, "bad_request", f"unsupported speaker_resolution: {request.speaker_resolution}")
    if request.purpose not in {"transcription", "oral_analysis", "podcast"}:
        raise AsrError(400, "bad_request", f"unsupported purpose: {request.purpose}")
    if request.pause_threshold_seconds < 0.2 or request.pause_threshold_seconds > 10.0:
        raise AsrError(400, "bad_request", "pause_threshold_seconds must be between 0.2 and 10")
    for name, value in (("max_chunk_seconds", request.max_chunk_seconds), ("overlap_seconds", request.overlap_seconds), ("pause_threshold_seconds", request.pause_threshold_seconds)):
        if value is not None and not math.isfinite(value):
            raise AsrError(400, "bad_request", f"{name} must be finite")
    selected_model = request.model or manager.model_for_purpose(request.purpose)
    runtime = manager.runtime_for(selected_model)
    if runtime.rejecting_new_requests or runtime.status in {"unloading", "unloading_scheduled"}:
        raise AsrError(409, "model_unloading_scheduled", "model is scheduled for unloading")
    resolved_backend = manager.resolve_backend(runtime, request.backend)
    if not runtime.definition.capabilities.transcription:
        raise AsrError(422, "capability_not_supported", "model does not support transcription")
    if (request.context or request.hotwords) and not runtime.definition.capabilities.context_prompt:
        raise AsrError(422, "capability_not_supported", "model does not support context prompting")
    if not runtime.definition.supports_pipeline(request.quality_profile):
        raise AsrError(422, "capability_not_supported", "model does not support the requested pipeline")
    if request.quality_profile == "quality_v1":
        if not runtime.definition.capabilities.forced_alignment:
            raise AsrError(422, "capability_not_supported", "quality_v1 requires enabled Qwen forced alignment")
        if request.quality_chunk_seconds is not None and request.quality_chunk_seconds not in {30, 60, 90, 120, 180}:
            raise AsrError(400, "bad_request", "quality_chunk_seconds must be 30, 60, 90, 120 or 180")
        if request.overlap_seconds is not None and request.overlap_seconds not in {0, 1}:
            raise AsrError(400, "bad_request", "quality_v1 context padding must be 0 or 1 second")
    if request.language not in runtime.definition.capabilities.languages:
        raise AsrError(
            422,
            "capability_not_supported",
            f"{selected_model} does not support language: {request.language}",
            {"language": request.language},
        )
    if request.timestamps != "none" and request.timestamps not in runtime.definition.capabilities.timestamps:
        if request.timestamps == "auto" and runtime.definition.capabilities.timestamps:
            pass
        else:
            raise AsrError(
                422,
                "capability_not_supported",
                f"{selected_model} does not support requested timestamps in this server",
                {"timestamps": request.timestamps, "supported": runtime.definition.capabilities.timestamps},
            )
    if request.purpose == "podcast" and not runtime.definition.capabilities.diarization:
        raise AsrError(
            422,
            "capability_not_supported",
            f"{selected_model} cannot satisfy purpose=podcast because it lacks diarization",
            {"required_capability": "diarization"},
        )
    if request.speaker_resolution != "off" and not runtime.definition.capabilities.diarization:
        raise AsrError(
            422,
            "capability_not_supported",
            f"{selected_model} does not support speaker resolution",
            {"speaker_resolution": request.speaker_resolution},
        )
    if request.speaker_resolution != "off" and request.response_format != "verbose_json":
        raise AsrError(
            422,
            "capability_not_supported",
            "speaker resolution requires response_format=verbose_json",
            {"response_format": request.response_format, "recommended_response_format": "verbose_json"},
        )
    prepared_context = prepare_context(request.context, request.hotwords)
    return ValidatedTranscription(
        selected_model=selected_model,
        resolved_backend=resolved_backend,
        adapter_context=prepared_context.prompt,
        max_new_tokens=runtime.definition.execution_policy.validate_max_new_tokens(request.max_new_tokens),
        execution_policy=runtime.definition.execution_policy,
        supports_diarization=runtime.definition.capabilities.diarization,
        context=prepared_context,
    )


async def run_transcription(
    audio: bytes,
    *,
    manager: ModelLifecycleManager,
    settings: Settings,
    request: TranscriptionRequest,
    validated: ValidatedTranscription | None = None,
    stage_callback: StageCallback | None = None,
    before_chunk: BeforeChunkCallback | None = None,
    after_chunk: AfterChunkCallback | None = None,
) -> dict[str, object]:
    checked = validated or validate_transcription_request(manager, request)
    cancel_event = current_context().cancel_event
    if request.quality_profile == "quality_v1":
        import tempfile

        with tempfile.TemporaryDirectory(prefix="asr_quality_") as temporary:
            root = Path(temporary)
            source = root / "upload"
            await asyncio.to_thread(source.write_bytes, audio)
            return await run_transcription_path(
                source, workspace=root, manager=manager, settings=settings, request=request,
                validated=checked, stage_callback=stage_callback,
                before_chunk=before_chunk, after_chunk=after_chunk,
            )
    if request.speaker_resolution != "off":
        raise AsrError(
            422,
            "capability_not_supported",
            "speaker resolution requires the path-based transcription pipeline",
        )
    if stage_callback is not None:
        await stage_callback("preprocessing", {"percent": 1.0})
    normalized = await asyncio.to_thread(normalize_audio_to_wav, audio)
    metadata = await asyncio.to_thread(inspect_audio, normalized.audio)
    plan = checked.execution_policy.plan(
        requested_split_strategy=request.split_strategy,
        audio_duration_seconds=metadata.duration_seconds,
        max_chunk_seconds=request.max_chunk_seconds,
        overlap_seconds=request.overlap_seconds,
    )
    if stage_callback is not None:
        await stage_callback("splitting", {"percent": 3.0})
    split = await asyncio.to_thread(
        lambda: split_audio(
            normalized.audio,
            split_strategy=plan.split_strategy,
            max_chunk_seconds=plan.max_chunk_seconds,
            overlap_seconds=request.overlap_seconds,
            hard_chunk_seconds=plan.hard_chunk_seconds,
        )
    )
    emit("split.completed", strategy=split.strategy, chunk_count=len(split.chunks))
    split = replace(
        split,
        requested_strategy=request.split_strategy,
        warnings=list(dict.fromkeys([*split.warnings, *plan.warnings])),
    )
    resolved_max_new_tokens = _resolved_max_new_tokens(checked, split)
    speaker_scope = _speaker_scope(checked, plan, len(split.chunks))
    if stage_callback is not None:
        await stage_callback(
            "loading_model",
            {
                "percent": 5.0,
                "split": _split_summary(split, plan, speaker_scope),
                "total_chunks": len(split.chunks),
                "completed_chunks": 0,
                "chunk_windows": [(chunk.start, chunk.end) for chunk in split.chunks],
            },
        )
    async def guarded_before_chunk(chunk_index: int, total_chunks: int) -> None:
        if cancel_event.is_set():
            raise asyncio.CancelledError
        emit("chunk.started", chunk_index=chunk_index, total_chunks=total_chunks)
        if before_chunk is not None:
            await before_chunk(chunk_index, total_chunks)

    async def guarded_after_chunk(chunk_index: int, total_chunks: int, result: TranscriptionResult) -> None:
        emit("chunk.completed", chunk_index=chunk_index, total_chunks=total_chunks,
             duration_ms=result.timings.inference_ms, generated_tokens=result.generation.generated_tokens)
        if after_chunk is not None:
            await after_chunk(chunk_index, total_chunks, result)

    effective_before_chunk: BeforeChunkCallback | None = guarded_before_chunk
    effective_after_chunk: AfterChunkCallback | None = guarded_after_chunk
    if plan.execution_mode == "native_long_form":
        effective_after_chunk = None

        async def native_before_chunk(_chunk_index: int, _total_chunks: int) -> None:
            if cancel_event.is_set():
                raise asyncio.CancelledError
            if stage_callback is not None:
                await stage_callback("transcribing", {"percent": 5.0})

        effective_before_chunk = native_before_chunk
    resolved_backend, chunk_results, timings = await manager.transcribe_chunks(
        [chunk.audio for chunk in split.chunks],
        model_id=checked.selected_model,
        backend=request.backend,
        language=request.language,
        timestamps=request.timestamps,
        context=checked.adapter_context,
        max_new_tokens=resolved_max_new_tokens,
        batch_size=settings.qwen_batch_size,
        before_chunk=effective_before_chunk,
        after_chunk=effective_after_chunk,
    )
    timings = replace(
        timings,
        total_ms=timings.total_ms + normalized.decode_ms,
        decode_ms=timings.decode_ms + normalized.decode_ms,
    )
    if stage_callback is not None:
        await stage_callback("merging", {"percent": 99.0})
    result = await asyncio.to_thread(
        merge_transcription_results,
        split.chunks,
        chunk_results,
        source_duration=split.metadata.duration_seconds,
        preserve_segments=request.preserve_segments,
        timings=timings,
        speaker_scope="global" if speaker_scope == "global" else "chunk",
    )
    warnings = _warnings(
        result.warnings,
        [*split.warnings, *checked.context.warnings],
        resolved_max_new_tokens,
        request.max_new_tokens,
    )
    timing = timing_analysis(
        result.timestamps,
        transcript=result.text,
        duration=result.duration,
        chunks=split.chunks,
        speech_intervals=getattr(split, "speech_intervals", ()),
        vad_based_chunks=split.strategy in {"energy", "silero"},
        pause_threshold_seconds=request.pause_threshold_seconds,
    )
    return transcription_payload(
        id_=f"tr_{uuid4().hex}",
        selected_model=checked.selected_model,
        resolved_backend=resolved_backend,
        split=split,
        text=result.text,
        language=result.language,
        duration=result.duration,
        chunks=result.chunks,
        segments=result.segments if request.response_format == "verbose_json" else [],
        timestamps=result.timestamps,
        timings=result.timings.to_api(),
        warnings=warnings,
        execution=plan.to_api(speaker_scope=speaker_scope),
        generation=_generation_payload(split, chunk_results, resolved_max_new_tokens),
        diarization=None,
        context=_context_payload(checked.context, chunk_results),
        timing=timing,
        purpose=request.purpose,
    )


async def run_transcription_path(
    upload_path: Path,
    *,
    workspace: Path,
    manager: ModelLifecycleManager,
    settings: Settings,
    request: TranscriptionRequest,
    validated: ValidatedTranscription | None = None,
    stage_callback: StageCallback | None = None,
    before_chunk: BeforeChunkCallback | None = None,
    after_chunk: AfterChunkCallback | None = None,
    checkpoint_dir: Path | None = None,
    normalized_asset: NormalizedAudioAsset | None = None,
    cancel_event: threading.Event | None = None,
    owner_workspace: AudioWorkspace | None = None,
) -> dict[str, object]:
    checked = validated or validate_transcription_request(manager, request)
    cancel_event = cancel_event or current_context().cancel_event
    if cancel_event.is_set():
        raise asyncio.CancelledError
    if request.quality_profile == "quality_v1":
        from asr_server.quality.pipeline import run_quality_transcription

        return await run_quality_transcription(
            upload_path, workspace=workspace, manager=manager, settings=settings,
            request=request, checked=checked, stage_callback=stage_callback,
            before_chunk=before_chunk, after_chunk=after_chunk, checkpoint_dir=checkpoint_dir,
            normalized_asset=normalized_asset, cancel_event=cancel_event, owner_workspace=owner_workspace,
        )
    if stage_callback is not None:
        await stage_callback("preprocessing", {"percent": 1.0})
    if normalized_asset is not None:
        await asyncio.to_thread(normalized_asset.validate, spec=manager.runtime_for(checked.selected_model).definition.audio_spec)
        normalized = NormalizedAudioPath(normalized_asset.path, 0.0, normalized_asset)
        emit("media.reused", output_bytes=normalized_asset.size_bytes)
    else:
        from asr_server.audio.workspace import remaining_workspace_output_bytes
        budget = await asyncio.to_thread(remaining_workspace_output_bytes, workspace, settings,
            exclude=(checkpoint_dir,) if checkpoint_dir else (), output_path=workspace / "normalized.wav")
        lease = None
        if owner_workspace is not None:
            spec = manager.runtime_for(checked.selected_model).definition.audio_spec
            budget = min(budget, int(21600 * spec.bytes_per_second) + 44)
            lease = await owner_workspace._manager.reserve_resource(budget, workspace)
            owner_workspace.resource_leases.append(lease)
        normalized = await normalize_audio_path_to_wav_async(
            upload_path, workspace / "normalized.wav", timeout_seconds=settings.ffmpeg_timeout_seconds,
            max_output_bytes=budget, cancel_event=cancel_event, resource_lease=lease,
            spec=manager.runtime_for(checked.selected_model).definition.audio_spec,
        )
        if lease is not None:
            await lease.release_unused()
    await asyncio.to_thread(validate_workspace_limits, workspace, settings)
    metadata = await asyncio.to_thread(inspect_audio_path, normalized.path)
    plan = checked.execution_policy.plan(
        requested_split_strategy=request.split_strategy,
        audio_duration_seconds=metadata.duration_seconds,
        max_chunk_seconds=request.max_chunk_seconds,
        overlap_seconds=request.overlap_seconds,
    )
    plan = _anchor_compatible_plan(plan, request.speaker_resolution)
    if stage_callback is not None:
        await stage_callback("splitting", {"percent": 3.0})
    split_cancel = cancel_event
    split_operation = asyncio.create_task(
        asyncio.to_thread(
            split_audio_path,
            normalized.path,
            split_strategy=plan.split_strategy,
            max_chunk_seconds=plan.max_chunk_seconds,
            overlap_seconds=request.overlap_seconds,
            cancel_event=split_cancel,
            hard_chunk_seconds=plan.hard_chunk_seconds,
        )
    )
    try:
        split = await asyncio.shield(split_operation)
    except asyncio.CancelledError:
        split_cancel.set()
        try:
            await split_operation
        except SplitCancelled:
            pass
        raise
    emit("split.completed", strategy=split.strategy, chunk_count=len(split.chunks))
    split = replace(
        split,
        requested_strategy=request.split_strategy,
        warnings=list(
            dict.fromkeys(
                [
                    *split.warnings,
                    *plan.warnings,
                    *(
                        ["moss_anchor_replay_body_chunk_limit:1200"]
                        if request.speaker_resolution != "off"
                        and plan.max_chunk_seconds == MOSS_ANCHOR_REPLAY_BODY_CHUNK_SECONDS
                        else []
                    ),
                ]
            )
        ),
    )
    resolved_max_new_tokens = _resolved_max_new_tokens(
        checked,
        split,
        invocation_overhead_seconds=(
            60.0 if request.speaker_resolution != "off" and len(split.chunks) > 1 else 0.0
        ),
        minimum_auto_max_new_tokens=(
            MOSS_ANCHOR_REPLAY_MIN_MAX_NEW_TOKENS
            if request.speaker_resolution != "off" and len(split.chunks) > 1
            else None
        ),
    )
    speaker_scope = _speaker_scope(checked, plan, len(split.chunks), request.speaker_resolution)
    if stage_callback is not None:
        await stage_callback(
            "loading_model",
            {
                "percent": 5.0,
                "split": _split_summary(split, plan, speaker_scope),
                "total_chunks": len(split.chunks),
                "completed_chunks": 0,
                "chunk_windows": [(chunk.start, chunk.end) for chunk in split.chunks],
            },
        )
    async def guarded_before_chunk(chunk_index: int, total_chunks: int) -> None:
        if cancel_event.is_set():
            raise asyncio.CancelledError
        emit("chunk.started", chunk_index=chunk_index, total_chunks=total_chunks)
        if before_chunk is not None:
            await before_chunk(chunk_index, total_chunks)

    async def guarded_after_chunk(chunk_index: int, total_chunks: int, result: TranscriptionResult) -> None:
        emit("chunk.completed", chunk_index=chunk_index, total_chunks=total_chunks,
             duration_ms=result.timings.inference_ms, generated_tokens=result.generation.generated_tokens)
        if after_chunk is not None:
            await after_chunk(chunk_index, total_chunks, result)

    effective_before_chunk: BeforeChunkCallback | None = guarded_before_chunk
    effective_after_chunk: AfterChunkCallback | None = guarded_after_chunk
    if plan.execution_mode == "native_long_form":
        effective_after_chunk = None

        async def native_before_chunk(_chunk_index: int, _total_chunks: int) -> None:
            if cancel_event.is_set():
                raise asyncio.CancelledError
            if stage_callback is not None:
                await stage_callback("transcribing", {"percent": 5.0})

        effective_before_chunk = native_before_chunk
    audio_inputs: list[AudioInput] = (
        [AudioPath(path=normalized.path)]
        if plan.execution_mode == "native_long_form"
        else [chunk.as_audio_input() for chunk in split.chunks]
    )
    anchor_sequence: AnchorReplaySequence | None = None
    if request.speaker_resolution != "off" and len(split.chunks) > 1:
        anchor_sequence = AnchorReplaySequence(split.chunks)
        resolved_backend, chunk_results, timings = await manager.transcribe_sequence(
            anchor_sequence,
            model_id=checked.selected_model,
            backend=request.backend,
            language=request.language,
            timestamps=request.timestamps,
            context=checked.adapter_context,
            max_new_tokens=resolved_max_new_tokens,
            before_chunk=effective_before_chunk,
            after_chunk=effective_after_chunk,
        )
    else:
        resolved_backend, chunk_results, timings = await manager.transcribe_chunks(
            audio_inputs,
            model_id=checked.selected_model,
            backend=request.backend,
            language=request.language,
            timestamps=request.timestamps,
            context=checked.adapter_context,
            max_new_tokens=resolved_max_new_tokens,
            batch_size=1,
            before_chunk=effective_before_chunk,
            after_chunk=effective_after_chunk,
        )
    await asyncio.to_thread(validate_workspace_limits, workspace, settings)
    timings = replace(
        timings,
        total_ms=timings.total_ms + normalized.decode_ms,
        decode_ms=timings.decode_ms + normalized.decode_ms,
    )
    if stage_callback is not None:
        await stage_callback("merging", {"percent": 99.0})
    diarization = _diarization_summary(
        request.speaker_resolution,
        anchor_sequence,
        checked.supports_diarization,
        chunk_results,
    )
    if request.speaker_resolution == "required" and diarization is not None and diarization["status"] != "complete":
        raise AsrError(
            422,
            "speaker_resolution_incomplete",
            "not every speaker segment could be resolved across chunks",
            diarization,
        )
    speaker_scope = str(diarization["speaker_scope"]) if diarization is not None else speaker_scope
    result = await asyncio.to_thread(
        merge_transcription_results,
        split.chunks,
        chunk_results,
        source_duration=split.metadata.duration_seconds,
        preserve_segments=request.preserve_segments,
        timings=timings,
        speaker_scope=(
            "mixed" if speaker_scope == "mixed" else "global" if speaker_scope == "global" else "chunk"
        ),
    )
    warnings = _warnings(
        result.warnings,
        [*split.warnings, *checked.context.warnings],
        resolved_max_new_tokens,
        request.max_new_tokens,
    )
    timing = timing_analysis(
        result.timestamps,
        transcript=result.text,
        duration=result.duration,
        chunks=split.chunks,
        speech_intervals=getattr(split, "speech_intervals", ()),
        vad_based_chunks=split.strategy in {"energy", "silero"},
        pause_threshold_seconds=request.pause_threshold_seconds,
    )
    return transcription_payload(
        id_=f"tr_{uuid4().hex}",
        selected_model=checked.selected_model,
        resolved_backend=resolved_backend,
        split=split,
        text=result.text,
        language=result.language,
        duration=result.duration,
        chunks=result.chunks,
        segments=result.segments if request.response_format == "verbose_json" else [],
        timestamps=result.timestamps,
        timings=result.timings.to_api(),
        warnings=warnings,
        execution=plan.to_api(speaker_scope=speaker_scope),
        generation=_generation_payload(split, chunk_results, resolved_max_new_tokens),
        diarization=diarization,
        context=_context_payload(checked.context, chunk_results),
        timing=timing,
        purpose=request.purpose,
    )


def transcription_payload(
    *,
    id_: str,
    selected_model: str,
    resolved_backend: str,
    split: SplitResult | PathSplitResult,
    text: str,
    language: str,
    duration: float,
    chunks: list[dict[str, object]],
    segments: list[dict[str, object]],
    timestamps: list[dict[str, object]],
    timings: dict[str, float],
    warnings: list[str],
    execution: dict[str, object],
    generation: dict[str, object],
    diarization: dict[str, object] | None,
    context: dict[str, object],
    timing: dict[str, object],
    purpose: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": id_,
        "model": selected_model,
        "backend": resolved_backend,
        "language": language,
        "text": text,
        "duration": duration,
        "timestamps": timestamps,
        "segments": segments,
        "split": split.summary(),
        "chunks": chunks,
        "usage": {"audio_seconds": duration},
        "timings": timings,
        "warnings": warnings,
        "execution": execution,
        "generation": generation,
        "context": context,
        "timing": timing,
        "purpose": purpose,
    }
    if diarization is not None:
        payload["diarization"] = diarization
    return payload


def _resolved_max_new_tokens(
    checked: ValidatedTranscription,
    split: SplitResult | PathSplitResult,
    invocation_overhead_seconds: float = 0.0,
    minimum_auto_max_new_tokens: int | None = None,
) -> int:
    invocation_duration = max((chunk.duration for chunk in split.chunks), default=0.0) + invocation_overhead_seconds
    resolved = checked.execution_policy.resolve_max_new_tokens(
        checked.max_new_tokens,
        invocation_duration_seconds=invocation_duration,
    )
    if checked.max_new_tokens is not None or minimum_auto_max_new_tokens is None:
        return resolved
    return min(max(resolved, minimum_auto_max_new_tokens), checked.execution_policy.max_new_tokens)


def _speaker_scope(
    checked: ValidatedTranscription,
    plan: ExecutionPlan,
    chunk_count: int,
    speaker_resolution: str = "off",
) -> str:
    if not checked.supports_diarization:
        return "none"
    if chunk_count == 1:
        return "global"
    if speaker_resolution != "off":
        return "global"
    return "chunk"


def _anchor_compatible_plan(plan: ExecutionPlan, speaker_resolution: str) -> ExecutionPlan:
    if speaker_resolution == "off" or plan.execution_mode != "chunked":
        return plan
    body_limit = MOSS_ANCHOR_REPLAY_BODY_CHUNK_SECONDS
    if plan.max_chunk_seconds is None or plan.max_chunk_seconds <= body_limit:
        return plan
    return replace(
        plan,
        max_chunk_seconds=body_limit,
    )


def _diarization_summary(
    speaker_resolution: str,
    sequence: AnchorReplaySequence | None,
    supports_diarization: bool,
    results: list[TranscriptionResult],
) -> dict[str, object] | None:
    if speaker_resolution == "off" or not supports_diarization:
        return None
    if sequence is not None:
        return sequence.summary()
    segments = [segment for result in results for segment in result.segments]
    speakers = {segment.speaker for segment in segments if segment.speaker is not None}
    unresolved = sum(segment.speaker is None for segment in segments)
    missing_segment_chunks = sum(bool(result.text.strip()) and not result.segments for result in results)
    complete = bool(segments) and unresolved == 0 and missing_segment_chunks == 0
    return {
        "method": "model_native",
        "status": "complete" if complete else "partial",
        "speaker_scope": "global" if complete else "mixed",
        "speaker_count": len(speakers),
        "unresolved_segments": unresolved,
        "conflicts": 0,
        "anchor_budget_limited": False,
        "candidate_speakers": 0,
        "missing_segment_chunks": missing_segment_chunks,
    }


def _split_summary(
    split: SplitResult | PathSplitResult,
    plan: ExecutionPlan,
    speaker_scope: str,
) -> dict[str, object]:
    return {
        **split.summary(),
        "execution_mode": plan.execution_mode,
        "speaker_scope": speaker_scope,
    }


def _generation_payload(
    split: SplitResult | PathSplitResult,
    results: list[TranscriptionResult],
    max_new_tokens: int,
) -> dict[str, object]:
    chunks: list[ChunkLike] = list(split.chunks)
    prompt_values = [item.generation.prompt_tokens for item in results if item.generation.prompt_tokens is not None]
    generated_values = [
        item.generation.generated_tokens for item in results if item.generation.generated_tokens is not None
    ]
    peak_values = [
        item.generation.peak_vram_allocated_mb
        for item in results
        if item.generation.peak_vram_allocated_mb is not None
    ]
    coverage_ends = [
        chunk.start + result.generation.segment_coverage_end_seconds
        for chunk, result in zip(chunks, results, strict=True)
        if result.generation.segment_coverage_end_seconds is not None
    ]
    coverage_end = max(coverage_ends, default=None)
    coverage_ratio = (
        min(coverage_end / split.metadata.duration_seconds, 1.0)
        if coverage_end is not None and split.metadata.duration_seconds > 0
        else None
    )
    audible_ends = [
        chunk.start + result.generation.audible_speech_end_seconds
        for chunk, result in zip(chunks, results, strict=True)
        if result.generation.audible_speech_end_seconds is not None
    ]
    uncovered_speech = [
        result.generation.uncovered_speech_intervals
        for result in results
        if result.generation.uncovered_speech_intervals is not None
    ]
    return {
        "prompt_tokens": sum(prompt_values) if prompt_values else None,
        "generated_tokens": sum(generated_values) if generated_values else None,
        "max_new_tokens": max_new_tokens,
        "peak_vram_allocated_mb": max(peak_values) if peak_values else None,
        "segment_coverage_end_seconds": coverage_end,
        "segment_coverage_ratio": coverage_ratio,
        "audible_speech_end_seconds": max(audible_ends, default=None),
        "uncovered_speech_intervals": sum(uncovered_speech) if uncovered_speech else None,
        "coverage_is_not_accuracy": True,
        "invocation_count": len(results),
        "truncated": any(item.generation.truncated for item in results),
        "finish_reasons": list(
            dict.fromkeys(
                item.generation.finish_reason
                for item in results
                if item.generation.finish_reason is not None
            )
        ),
    }


def _context_payload(prepared: PreparedContext, results: list[TranscriptionResult]) -> dict[str, object]:
    applications = [result.context for result in results]
    applied = prepared.requested and bool(applications) and all(item.applied for item in applications)
    methods = list(dict.fromkeys(item.method for item in applications if item.method is not None))
    ignored_reasons = list(
        dict.fromkeys(item.ignored_reason for item in applications if item.ignored_reason is not None)
    )
    return {
        **prepared.request_summary(),
        "applied": applied,
        "method": methods[0] if len(methods) == 1 else methods or None,
        "ignored_reason": ignored_reasons[0] if len(ignored_reasons) == 1 else ignored_reasons or None,
        "invocation_count": len(applications),
    }


def _warnings(
    result_warnings: list[str],
    split_warnings: list[str],
    max_new_tokens: int | None,
    requested_max_new_tokens: int | None,
) -> list[str]:
    generation_warnings = [f"max_new_tokens_override:{max_new_tokens}"] if requested_max_new_tokens is not None else []
    return list(dict.fromkeys([*result_warnings, *split_warnings, *generation_warnings]))
