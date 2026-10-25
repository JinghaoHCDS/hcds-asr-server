from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from asr_server.adapters.base import AudioPath, TimestampItem, TranscriptionResult, TranscriptionTimings
from asr_server.adapters.qwen import ALIGNER_REVISION
from asr_server.audio.metadata import inspect_audio_path
from asr_server.audio.preprocess import NormalizedAudioPath, normalize_audio_path_to_wav_async
from asr_server.audio.assets import NORMALIZATION_VERSION, NormalizedAudioAsset
from asr_server.audio.workspace import AudioWorkspace
from asr_server.observability import current_context, emit
from asr_server.audio.timing import timing_analysis
from asr_server.errors import AsrError
from asr_server.quality import PROFILE_VERSION
from asr_server.quality.checkpoints import Checkpoints, file_digest, fingerprint
from asr_server.quality.glossary import glossary_context, normalize_terms
from asr_server.quality.splitter import QualityChunk, audio_has_energy, empty_audio_evidence, plan_quality_chunks
from asr_server.quality.segments import reading_segments
from asr_server.quality.speech_evidence import detector_identity, non_speech_evidence

if TYPE_CHECKING:
    from asr_server.config import Settings
    from asr_server.lifecycle import ModelLifecycleManager
    from asr_server.transcription import AfterChunkCallback, BeforeChunkCallback, StageCallback, TranscriptionRequest, ValidatedTranscription


_REPEATS = re.compile(r"(.{1,30}?)\1{5,}")
_RECOVERABLE = {"generation_truncated", "empty_transcription"}


def _issues(result: TranscriptionResult, snapshot: dict[str, object] | None) -> list[str]:
    issues: list[str] = []
    if result.generation.truncated:
        issues.append("generation_truncated")
    if not result.text.strip() and "confirmed_empty_audio" not in result.warnings:
        issues.append("empty_transcription")
    if _REPEATS.search(re.sub(r"\s+", "", result.text)):
        issues.append("suspected_repetition")
    _text, _changes, unresolved = normalize_terms(result.text, snapshot)
    if unresolved:
        issues.append("suspected_term")
    return issues


def _absolute(items: list[TimestampItem], offset: float) -> list[TimestampItem]:
    return [replace(item, start=item.start + offset, end=item.end + offset) for item in items]


def _joined_text(items: list[TimestampItem]) -> str:
    text = ""
    for item in items:
        if text and text[-1].isascii() and text[-1].isalnum() and item.text[:1].isascii() and item.text[:1].isalnum():
            text += " "
        text += item.text
    return text


def _owned_text(result: TranscriptionResult, chunk: QualityChunk) -> tuple[str, list[TimestampItem]]:
    absolute = _absolute(result.timestamps, chunk.start)
    if chunk.start == chunk.core_start and chunk.end == chunk.core_end:
        return result.text, absolute
    selected = [item for item in absolute if chunk.core_start <= (item.start + item.end) / 2 < chunk.core_end]
    # Text ownership is always determined by real timestamps. Never reuse the
    # old fuzzy prefix deletion or invent equally spaced word timestamps.
    if not absolute:
        if not result.text.strip():
            return "", []
        raise AsrError(422, "alignment_required", "overlapping quality chunks require real timestamps")
    if not selected:
        return "", []
    # Preserve punctuation by mapping aligned units back onto the model text.
    cursor = 0
    spans: list[tuple[int, int]] = []
    for item in absolute:
        position = result.text.casefold().find(item.text.casefold(), cursor)
        if position < 0:
            return _joined_text(selected), selected
        spans.append((position, position + len(item.text)))
        cursor = position + len(item.text)
    first = absolute.index(selected[0])
    last = absolute.index(selected[-1])
    left = 0 if first == 0 else spans[first][0]
    right = spans[last + 1][0] if last + 1 < len(spans) else len(result.text)
    return result.text[left:right].strip(), selected


def _replace_interval(
    result: TranscriptionResult, chunk: QualityChunk, replacement: TranscriptionResult,
    start: float, end: float,
) -> TranscriptionResult | None:
    """Patch a reviewed acoustic interval only when original units map exactly."""
    original = _absolute(result.timestamps, chunk.start)
    selected = [index for index, item in enumerate(original) if start <= (item.start + item.end) / 2 < end]
    if not selected or not replacement.timestamps:
        return None
    cursor = 0
    spans = []
    for item in original:
        position = result.text.casefold().find(item.text.casefold(), cursor)
        if position < 0:
            return None
        spans.append((position, position + len(item.text)))
        cursor = position + len(item.text)
    first, last = selected[0], selected[-1]
    # Keep surrounding punctuation, while never guessing characters outside
    # the precisely mapped reviewed region.
    text = result.text[:spans[first][0]] + replacement.text + result.text[spans[last][1]:]
    replacement_items = _absolute(replacement.timestamps, start)
    kept = original[:first] + replacement_items + original[last + 1:]
    kept.sort(key=lambda item: (item.start, item.end))
    return replace(result, text=text, timestamps=[replace(item, start=item.start - chunk.start, end=item.end - chunk.start) for item in kept])


async def run_quality_transcription(
    upload_path: Path,
    *,
    workspace: Path,
    manager: ModelLifecycleManager,
    settings: Settings,
    request: TranscriptionRequest,
    checked: ValidatedTranscription,
    stage_callback: StageCallback | None,
    before_chunk: BeforeChunkCallback | None,
    after_chunk: AfterChunkCallback | None,
    checkpoint_dir: Path | None,
    normalized_asset: NormalizedAudioAsset | None = None,
    cancel_event: threading.Event | None = None,
    owner_workspace: AudioWorkspace | None = None,
) -> dict[str, object]:
    started = perf_counter()
    if stage_callback:
        await stage_callback("preprocessing", {})
    cancel_event = cancel_event or current_context().cancel_event
    if normalized_asset is not None:
        await asyncio.to_thread(normalized_asset.validate, spec=manager.runtime_for(checked.selected_model).definition.audio_spec)
        normalized = NormalizedAudioPath(normalized_asset.path, 0.0, normalized_asset)
        emit("media.reused", output_bytes=normalized_asset.size_bytes)
    else:
        from asr_server.audio.workspace import remaining_workspace_output_bytes
        budget_bytes = await asyncio.to_thread(remaining_workspace_output_bytes, workspace, settings,
            exclude=(checkpoint_dir,) if checkpoint_dir else (), output_path=workspace / "normalized.wav")
        lease = None
        if owner_workspace is not None:
            spec = manager.runtime_for(checked.selected_model).definition.audio_spec
            budget_bytes = min(budget_bytes, int(21600 * spec.bytes_per_second) + 44)
            lease = await owner_workspace._manager.reserve_resource(budget_bytes, workspace)
            owner_workspace.resource_leases.append(lease)
        normalized = await normalize_audio_path_to_wav_async(upload_path, workspace / "normalized.wav",
            timeout_seconds=settings.ffmpeg_timeout_seconds, max_output_bytes=budget_bytes, cancel_event=cancel_event, resource_lease=lease)
        if lease is not None:
            await lease.release_unused()
    if stage_callback:
        await stage_callback("splitting", {})
    target = request.quality_chunk_seconds if request.quality_chunk_seconds is not None else 60.0
    padding = request.overlap_seconds if request.overlap_seconds is not None else 0.0
    chunks, duration = await asyncio.to_thread(plan_quality_chunks, normalized.path, target, padding, cancel_event=cancel_event)
    model_revision = manager.runtime_for(checked.selected_model).definition.revision
    prompt, glossary_report = glossary_context(request.context, request.hotwords, request.glossary_snapshot)
    context_report = glossary_report.pop("context")
    request_identity = asdict(request)
    request_identity.pop("idempotency_key", None)
    speech_detector = await asyncio.to_thread(detector_identity)
    identity = await asyncio.to_thread(fingerprint, {
        "audio": await asyncio.to_thread(file_digest, normalized.path), "profile": PROFILE_VERSION,
        "execution_identity": manager.runtime_for(checked.selected_model).definition.identity(),
        "normalization_version": NORMALIZATION_VERSION,
        "model": checked.selected_model, "revision": model_revision, "aligner_revision": ALIGNER_REVISION, "request": request_identity,
        "chunks": [asdict(chunk) for chunk in chunks],
        "speech_detector": speech_detector,
    })
    checkpoints = await asyncio.to_thread(Checkpoints, checkpoint_dir, identity)
    budget = sum(chunk.duration for chunk in chunks) * (0.30 if request.quality_review_enabled else 0.0)
    extra_seconds = checkpoints.extra_seconds
    reviews: list[dict[str, object]] = []
    all_issues: list[dict[str, object]] = []
    results: list[TranscriptionResult] = []
    empty_audio_records: list[dict[str, object]] = []
    max_tokens = request.max_new_tokens if request.max_new_tokens is not None else 4096
    if stage_callback:
        await stage_callback("loading_model", {"total_chunks": len(chunks), "completed_chunks": 0, "chunk_windows": [(chunk.start, chunk.end) for chunk in chunks], "split": {"strategy": "quality_low_energy", "chunk_count": len(chunks), "soft_chunk_seconds": target, "hard_chunk_seconds": 180, "overlap_seconds": padding}})

    async def recognize(start: float, end: float) -> TranscriptionResult:
        if cancel_event.is_set():
            raise asyncio.CancelledError
        emit("chunk.started", start_s=start, end_s=end)
        try:
            _backend, items, _timings = await manager.transcribe_chunks(
                [AudioPath(normalized.path, start, end)], model_id=checked.selected_model,
                backend=request.backend, language=request.language, timestamps="auto", context=prompt,
                max_new_tokens=max_tokens, batch_size=1,
            )
            result = items[0]
        except AsrError as error:
            if error.code != "empty_transcription":
                raise
            evidence = await asyncio.to_thread(empty_audio_evidence, normalized.path, start, end)
            if evidence is None:
                raise
            result = TranscriptionResult("", end - start, request.language, [])
        else:
            if result.text.strip() or result.generation.truncated:
                return result
            evidence = await asyncio.to_thread(empty_audio_evidence, normalized.path, start, end)
            if evidence is None:
                return result
        empty_audio_records.append({"method": "empty_audio", "start": start, "end": end,
                                    "accepted": True, "evidence": evidence})
        return replace(result, warnings=[*result.warnings, "confirmed_empty_audio"])

    for index, chunk in enumerate(chunks):
        if before_chunk:
            await before_chunk(index, len(chunks))
        if stage_callback:
            await stage_callback("transcribing", {"total_chunks": len(chunks), "completed_chunks": index, "current_chunk": index + 1, "current_chunk_start": chunk.start, "current_chunk_end": chunk.end})
        cached = await asyncio.to_thread(checkpoints.load, index)
        emit("quality.checkpoint_hit" if cached is not None else "quality.checkpoint_miss", chunk_index=index)
        if cached is not None:
            result, cached_reviews, cached_extra = cached
            chunk_reviews = cached_reviews
            reviews.extend(cached_reviews)
            if cached_extra > budget + 0.001 or extra_seconds > budget + 0.001:
                raise AsrError(409, "checkpoint_version_mismatch", "checkpoint retry budget is invalid")
        else:
            chunk_reviews = []
            empty_records_start = len(empty_audio_records)
            initial_error: AsrError | None = None
            result = TranscriptionResult("", chunk.duration, request.language, [])
            try:
                result = await recognize(chunk.start, chunk.end)
            except AsrError as error:
                if error.code not in _RECOVERABLE:
                    raise
                initial_error = error
            causes = [initial_error.code] if initial_error else _issues(result, request.glossary_snapshot)
            chunk_extra = 0.0
            review_key = f"chunk:{index}"
            for attempt in range(checkpoints.review_attempts.get(review_key, 0) + 1, 3):
                if not causes or extra_seconds + chunk.duration > budget + 0.001:
                    break
                if before_chunk:
                    await before_chunk(index, len(chunks))
                await asyncio.to_thread(checkpoints.reserve_review, chunk.duration, budget, review_key)
                extra_seconds = checkpoints.extra_seconds
                emit("quality.review", chunk_index=chunk.index, review_budget_seconds=budget, reason="shorter_windows", attempt=attempt)
                chunk_extra += chunk.duration
                # Shorter re-segmentation changes acoustic boundaries rather
                # than repeating an identical deterministic generation call.
                parts = 2 ** attempt
                candidates: list[TranscriptionResult] = []
                retry_error: AsrError | None = None
                retry_started = perf_counter()
                for part in range(parts):
                    start = chunk.start + chunk.duration * part / parts
                    end = chunk.start + chunk.duration * (part + 1) / parts
                    try:
                        candidate = await recognize(start, end)
                        candidates.append(replace(candidate, timestamps=_absolute(candidate.timestamps, start - chunk.start)))
                    except AsrError as error:
                        if error.code not in _RECOVERABLE:
                            raise
                        retry_error = error
                        break
                record: dict[str, object] = {"chunk_index": index, "attempt": attempt, "causes": causes, "start": chunk.start, "end": chunk.end, "extra_audio_seconds": chunk.duration, "elapsed_seconds": perf_counter() - retry_started, "method": "shorter_windows", "accepted": retry_error is None}
                chunk_reviews.append(record)
                if retry_error is None:
                    result = TranscriptionResult(
                        text="\n".join(candidate.text for candidate in candidates), duration=chunk.duration,
                        language=next((candidate.language for candidate in candidates if candidate.language), request.language),
                        warnings=list(dict.fromkeys(warning for candidate in candidates for warning in candidate.warnings)),
                        timestamps=[item for candidate in candidates for item in candidate.timestamps],
                        timings=TranscriptionTimings(inference_ms=sum(candidate.timings.inference_ms for candidate in candidates), postprocess_ms=sum(candidate.timings.postprocess_ms for candidate in candidates)),
                        generation=candidates[-1].generation, context=candidates[-1].context,
                    )
                    initial_error = None
                    causes = _issues(result, request.glossary_snapshot)
                else:
                    initial_error = retry_error
                    causes = [retry_error.code]
            if request.quality_review_enabled and causes == ["empty_transcription"]:
                fallback_evidence = await asyncio.to_thread(non_speech_evidence, normalized.path, chunk.start, chunk.end, speech_detector)
                if fallback_evidence is not None:
                    # Recognition has already covered this entire interval.
                    # Independent, full-frame speech checks corroborate an
                    # empty result, but never prove it correct: keep a review
                    # marker and retain the original audio without adding text.
                    result = TranscriptionResult("", chunk.duration, request.language, ["confirmed_empty_audio"])
                    empty_audio_records.append({"method": "empty_audio", "start": chunk.start, "end": chunk.end,
                                                "accepted": True, "evidence": fallback_evidence})
                    initial_error = None
                    causes = []
            chunk_reviews.extend({**record, "chunk_index": index} for record in empty_audio_records[empty_records_start:])
            reviews.extend(chunk_reviews)
            if initial_error or "generation_truncated" in causes or "empty_transcription" in causes:
                stop_reason = "review_attempt_limit" if checkpoints.review_attempts.get(review_key, 0) >= 2 else "review_audio_budget"
                raise AsrError(422, "quality_incomplete", "audio interval remains unresolved after bounded automatic review", {"start": chunk.start, "end": chunk.end, "chunk_index": index, "cause": initial_error.code if initial_error else causes[0], "stop_reason": stop_reason, "partial_text": "\n\n".join(item.text for item in results), "review_audio_seconds": extra_seconds, "review_budget_seconds": budget})
            await asyncio.to_thread(checkpoints.save, index, result, chunk_reviews, chunk_extra)
            emit("chunk.completed", chunk_index=index, start_s=chunk.start, end_s=chunk.end)
        for record in chunk_reviews:
            evidence = record.get("evidence")
            if record.get("method") == "empty_audio" and isinstance(evidence, dict):
                kind = {"near_digital_silence_tail": "near_silent_tail", "silero_no_speech_detected": "empty_without_detected_speech"}.get(str(evidence.get("method")))
                if kind:
                    all_issues.append({"kind": kind, "start": record["start"], "end": record["end"],
                                       "chunk_index": index, "status": "needs_review", "evidence": evidence})
        for cause in _issues(result, request.glossary_snapshot):
            all_issues.append({"kind": cause, "start": chunk.core_start, "end": chunk.core_end, "chunk_index": index, "status": "needs_review"})
        results.append(result)
        if after_chunk:
            await after_chunk(index, len(chunks), result)

    if stage_callback:
        await stage_callback("merging", {})
    # Long acoustic regions without aligned text are an uncertainty signal,
    # not proof of omitted speech (music can also have energy). Review a
    # bounded original-audio excerpt and expose its candidate for listening;
    # never silently inject an unverified music/silence hallucination.
    for chunk, result in zip(chunks, results, strict=True):
        if "mock_adapter" in result.warnings or not result.timestamps:
            continue
        local_items = sorted(result.timestamps, key=lambda item: item.start)
        boundaries = [(0.0, local_items[0].start), *[(left.end, right.start) for left, right in zip(local_items, local_items[1:])], (local_items[-1].end, chunk.duration)]
        for gap_start, gap_end in boundaries:
            if gap_end - gap_start <= 6:
                continue
            start = max(chunk.start, chunk.start + gap_start)
            end = min(chunk.end, chunk.start + gap_end)
            if not await asyncio.to_thread(audio_has_energy, normalized.path, start, end):
                continue
            issue: dict[str, object] = {"kind": "suspected_uncovered_audio", "start": start, "end": end, "chunk_index": chunk.index, "status": "needs_review", "evidence": "energy_without_aligned_text"}
            review_end = min(end, start + 10)
            gap_key = f"gap:{chunk.index}:{start:.3f}:{end:.3f}"
            if extra_seconds + review_end - start <= budget and checkpoints.review_attempts.get(gap_key, 0) < 2:
                if before_chunk:
                    await before_chunk(chunk.index, len(chunks))
                emit("quality.review", chunk_index=chunk.index, reason="acoustic_gap", review_budget_seconds=budget)
                await asyncio.to_thread(checkpoints.reserve_review, review_end - start, budget, gap_key)
                extra_seconds = checkpoints.extra_seconds
                try:
                    candidate = await recognize(start, review_end)
                    issue["candidate_text"] = candidate.text
                    reviews.append({"chunk_index": chunk.index, "start": start, "end": review_end, "method": "acoustic_gap", "accepted": False, "candidate_text": candidate.text, "extra_audio_seconds": review_end - start})
                except AsrError as error:
                    if error.code not in _RECOVERABLE:
                        raise
                    reviews.append({"chunk_index": chunk.index, "start": start, "end": review_end, "method": "acoustic_gap", "accepted": False, "cause": error.code, "extra_audio_seconds": review_end - start})
            all_issues.append(issue)
    # Reconcile discrepant overlapping context against one original-audio
    # window, then reapply timestamp ownership on either side. A retry that
    # cannot be mapped safely stays a visible issue instead of deleting text.
    for index in range(len(chunks) - 1):
        left_chunk, right_chunk = chunks[index:index + 2]
        overlap_start, overlap_end = right_chunk.start, left_chunk.end
        if overlap_end <= overlap_start:
            continue
        left_overlap = [item for item in _absolute(results[index].timestamps, left_chunk.start) if overlap_start <= (item.start + item.end) / 2 < overlap_end]
        right_overlap = [item for item in _absolute(results[index + 1].timestamps, right_chunk.start) if overlap_start <= (item.start + item.end) / 2 < overlap_end]
        if _joined_text(left_overlap).casefold() == _joined_text(right_overlap).casefold():
            continue
        repaired = False
        boundary_key = f"boundary:{index}"
        for attempt in range(checkpoints.review_attempts.get(boundary_key, 0) + 1, 3):
            start = max(left_chunk.start, left_chunk.core_end - (4 - attempt))
            end = min(right_chunk.end, left_chunk.core_end + (4 - attempt))
            if extra_seconds + end - start > budget:
                break
            if before_chunk:
                await before_chunk(index, len(chunks))
            emit("quality.review", chunk_index=index, reason="boundary", review_budget_seconds=budget, attempt=attempt)
            await asyncio.to_thread(checkpoints.reserve_review, end - start, budget, boundary_key)
            extra_seconds = checkpoints.extra_seconds
            try:
                reviewed = await recognize(start, end)
            except AsrError as error:
                if error.code not in _RECOVERABLE:
                    raise
                reviews.append({"start": start, "end": end, "attempt": attempt, "method": "boundary_window", "accepted": False, "cause": error.code})
                continue
            patched_left = _replace_interval(results[index], left_chunk, reviewed, start, end)
            patched_right = _replace_interval(results[index + 1], right_chunk, reviewed, start, end)
            repaired = patched_left is not None and patched_right is not None
            reviews.append({"start": start, "end": end, "attempt": attempt, "method": "boundary_window", "accepted": repaired, "extra_audio_seconds": end - start})
            if repaired:
                assert patched_left is not None and patched_right is not None
                results[index], results[index + 1] = patched_left, patched_right
                break
        if not repaired:
            all_issues.append({"kind": "boundary_conflict", "start": overlap_start, "end": overlap_end, "status": "needs_review"})
    segments: list[dict[str, object]] = []
    timestamps: list[dict[str, object]] = []
    changes: list[dict[str, object]] = []
    for chunk, result in zip(chunks, results, strict=True):
        raw_text, owned_items = _owned_text(result, chunk)
        text, term_changes, ambiguous = normalize_terms(raw_text, request.glossary_snapshot)
        for change in term_changes:
            changes.append({**change, "chunk_index": chunk.index, "start": chunk.core_start, "end": chunk.core_end})
        if ambiguous and not any(item.get("chunk_index") == chunk.index and item["kind"] == "suspected_term" for item in all_issues):
            all_issues.append({"kind": "suspected_term", "start": chunk.core_start, "end": chunk.core_end, "chunk_index": chunk.index, "terms": ambiguous, "status": "needs_review"})
        final_items = owned_items
        if term_changes and text.strip():
            # Canonical spellings have different token lengths; old alignment
            # is never reused for the modified text.
            aligned = await manager.align_text(AudioPath(normalized.path, chunk.core_start, chunk.core_end), text=text, model_id=checked.selected_model, backend=request.backend, language=result.language, timestamps="auto")
            final_items = _absolute(aligned, chunk.core_start)
        if text.strip() and not final_items:
            if "mock_adapter" not in result.warnings:
                raise AsrError(422, "alignment_failed", "quality transcript requires real timestamp alignment", {"chunk_index": chunk.index})
            all_issues.append({"kind": "alignment_unavailable_mock", "start": chunk.core_start, "end": chunk.core_end, "status": "needs_review"})
        absolute_payload = [item.to_api() for item in final_items]
        segments.append({"id": f"segment-{chunk.index:05d}", "index": chunk.index, "start": chunk.core_start, "end": chunk.core_end, "text": text, "raw_text": raw_text, "timestamps": absolute_payload, "timestamp_source": "qwen3_forced_aligner" if final_items else "unavailable"})
        timestamps.extend(absolute_payload)
    text = "\n\n".join(str(segment["text"]) for segment in segments if segment["text"])
    raw_text = "\n\n".join(str(segment["raw_text"]) for segment in segments if segment["raw_text"])
    segments = reading_segments(segments)
    for segment in segments:
        if segment.get("reading_segmentation") == "unmapped_alignment":
            all_issues.append({"kind": "alignment_text_mapping", "start": segment["start"], "end": segment["end"], "status": "needs_review"})
    warnings = list(dict.fromkeys(["quality_profile_not_quality_validated", *[warning for result in results for warning in result.warnings]]))
    for item in timestamps:
        if float(str(item["start"])) < 0 or float(str(item["end"])) > duration + 0.2:
            raise AsrError(422, "alignment_failed", "timestamp lies outside source audio")
    quality: dict[str, object] = {"profile_version": PROFILE_VERSION, "validated": False, "fingerprint": identity, "status": "needs_review" if all_issues else "complete", "issues": all_issues, "changes": changes, "reviews": reviews, "review_audio_seconds": extra_seconds, "review_budget_seconds": budget, "model_revision": model_revision, "aligner_revision": ALIGNER_REVISION, "speech_detector": speech_detector}
    metadata = await asyncio.to_thread(inspect_audio_path, normalized.path)
    return {
        "id": f"tr_{uuid4().hex}", "model": checked.selected_model, "backend": checked.resolved_backend,
        "language": next((result.language for result in results if result.language), request.language),
        "text": text, "raw_text": raw_text, "duration": duration, "segments": segments, "timestamps": timestamps,
        "chunks": [{"index": chunk.index, "start": chunk.start, "end": chunk.end, "core_start": chunk.core_start, "core_end": chunk.core_end, "text": result.text} for chunk, result in zip(chunks, results, strict=True)],
        "split": {"strategy": "quality_low_energy", "requested_strategy": request.split_strategy, "chunk_count": len(chunks), "soft_chunk_seconds": target, "hard_chunk_seconds": 180, "overlap_seconds": padding, "coverage": "complete_source_tiling"},
        "usage": {"audio_seconds": duration}, "timings": {"total_ms": (perf_counter() - started) * 1000, "decode_ms": normalized.decode_ms, "inference_ms": sum(result.timings.inference_ms for result in results), "postprocess_ms": sum(result.timings.postprocess_ms for result in results)},
        "warnings": warnings, "quality_profile": "quality_v1", "quality": quality, "glossary": glossary_report,
        "execution": {"mode": "chunked", "speaker_scope": "none", "profile_version": PROFILE_VERSION},
        "generation": {"max_new_tokens": max_tokens, "truncated": False},
        "context": context_report,
        "timing": timing_analysis(timestamps, transcript=text, duration=metadata.duration_seconds, chunks=[], speech_intervals=(), vad_based_chunks=False, pause_threshold_seconds=request.pause_threshold_seconds),
        "purpose": request.purpose,
    }
