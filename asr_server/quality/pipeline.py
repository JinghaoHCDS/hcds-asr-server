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
from asr_server.quality import PROFILE_VERSION, RECOVERY_POLICY_VERSION
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
_UNRESOLVED = _RECOVERABLE | {"suspected_repetition", "alignment_failed", "suspected_uncovered_audio", "review_budget_exhausted"}


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


def language_recovery_evidence(
    chunks: list[QualityChunk], results: list[TranscriptionResult], resolved: set[int],
    *, legacy_indices: set[int] | None = None,
) -> dict[str, object] | None:
    """Conservative model-label evidence, never independent language truth."""
    usable = [(index, item) for index, item in enumerate(results)
              if index in resolved and item.text.strip()]
    if len(usable) < 3:
        return None
    recent = usable[-3:]
    # Old v1.4 adapters silently filled zh when a language label was absent.
    # Full native model labels remain attributable; bare legacy codes do not.
    if any(index in (legacy_indices or set()) and item.language in {"zh", "en"} for index, item in recent):
        return None
    aliases = {"Chinese": "zh", "chinese": "zh", "zh": "zh", "English": "en", "english": "en", "en": "en"}
    labels = [aliases.get(item.language, item.language) for _, item in recent]
    if not labels[0] or labels[0] in {"auto", "unknown", "None"} or len(set(labels)) != 1 or any(separator in labels[0] for separator in (",", "/", "+", "|", ";")):
        return None
    if any(any(signal in warning.lower() for signal in ("mixed", "language_conflict", "language_label_missing")) for _, item in recent for warning in item.warnings):
        return None
    seconds = sum(chunks[index].core_end - chunks[index].core_start for index, _ in recent)
    if seconds < 30:
        return None
    return {"effective_language": labels[0], "source_chunk_indices": [index for index, _ in recent],
            "model_language_labels": [item.language for _, item in recent], "evidence_audio_seconds": seconds,
            "language_evidence_method": "three_consistent_model_labels", "independent_language_truth": False}


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
    await asyncio.to_thread(checkpoints.initialize_recovery, {"base_fingerprint": identity, "chunks": [asdict(chunk) for chunk in chunks], "requested_language": request.language, "normalization_version": NORMALIZATION_VERSION, "recognition_then_alignment": True})
    budget = sum(chunk.duration for chunk in chunks) * (0.30 if request.quality_review_enabled else 0.0)
    extra_seconds = checkpoints.extra_seconds
    reviews: list[dict[str, object]] = []
    all_issues: list[dict[str, object]] = []
    results: list[TranscriptionResult] = []
    gaps: list[dict[str, object]] = []
    empty_audio_records: list[dict[str, object]] = []
    max_tokens = request.max_new_tokens if request.max_new_tokens is not None else 4096

    def load_validated_caches() -> dict[int, tuple[TranscriptionResult, list[dict[str, object]], float]]:
        cached_results: dict[int, tuple[TranscriptionResult, list[dict[str, object]], float]] = {}
        if checkpoints.extra_seconds > budget + 0.001:
            raise AsrError(409, "checkpoint_version_mismatch", "checkpoint retry budget is invalid")
        for position in range(len(chunks)):
            if cancel_event.is_set():
                raise asyncio.CancelledError
            cached = checkpoints.load(position)
            if cached is not None:
                if cached[2] > budget + 0.001:
                    raise AsrError(409, "checkpoint_version_mismatch", "checkpoint retry budget is invalid")
                cached_results[position] = cached
        return cached_results

    # Validate the complete sparse cache before publishing any counts. A
    # resume starts with all durable successful intervals, not a zero prefix.
    cached_results = await asyncio.to_thread(load_validated_caches)
    resolved: set[int] = set(cached_results)
    attempted: set[int] = set(cached_results)
    if stage_callback:
        await stage_callback("loading_model", {"total_chunks": len(chunks), "completed_chunks": len(resolved), "resolved_chunks": len(resolved), "attempted_chunks": len(attempted), "chunk_windows": [(chunk.start, chunk.end) for chunk in chunks], "split": {"strategy": "quality_low_energy", "chunk_count": len(chunks), "soft_chunk_seconds": target, "hard_chunk_seconds": 180, "overlap_seconds": padding}})

    async def recognize(start: float, end: float, *, language: str | None = None,
                        call_key: str | None = None, review: bool = False) -> TranscriptionResult:
        if cancel_event.is_set():
            raise asyncio.CancelledError
        effective_language = language or request.language
        call_key = call_key or f"aux:{start:.6f}:{end:.6f}:{effective_language}"
        stored = await asyncio.to_thread(checkpoints.call, call_key)
        if stored is not None and stored.get("status") == "error":
            raise AsrError(422, str(stored["code"]), "saved recognition attempt remains unresolved")
        if stored is not None and isinstance(stored.get("result"), dict):
            result = checkpoints.result_from_payload(stored["result"])
            if isinstance(stored.get("empty_evidence"), dict):
                empty_audio_records.append({"method": "empty_audio", "start": start, "end": end, "accepted": True, "evidence": stored["empty_evidence"]})
        else:
            emit("chunk.started", start_s=start, end_s=end)
            await asyncio.to_thread(checkpoints.begin_call, call_key,
                {"start": start, "end": end, "start_sample": round(start * 16000), "end_sample": round(end * 16000),
                 "requested_language": request.language, "effective_language": effective_language,
                 "max_new_tokens": max_tokens, "timestamps": "none", "model": checked.selected_model, "backend": checked.resolved_backend, "prompt_fingerprint": fingerprint(prompt), "base_fingerprint": identity, "recovery_policy_version": RECOVERY_POLICY_VERSION},
                budget=budget, review_seconds=end - start if review else 0)
            call_empty_evidence: dict[str, object] | None = None
            try:
                _backend, items, _timings = await manager.transcribe_chunks(
                    [AudioPath(normalized.path, start, end)], model_id=checked.selected_model,
                    backend=request.backend, language=effective_language, timestamps="none", context=prompt,
                    max_new_tokens=max_tokens, batch_size=1,
                )
                result = items[0]
            except AsrError as error:
                if error.code not in _RECOVERABLE:
                    raise
                evidence = await asyncio.to_thread(empty_audio_evidence, normalized.path, start, end) if error.code == "empty_transcription" else None
                if evidence is None:
                    await asyncio.to_thread(checkpoints.finish_call, call_key, error=error.code)
                    raise
                call_empty_evidence = evidence
                result = TranscriptionResult("", end - start, effective_language, ["confirmed_empty_audio"])
                empty_audio_records.append({"method": "empty_audio", "start": start, "end": end, "accepted": True, "evidence": evidence})
            else:
                if not result.text.strip() and not result.generation.truncated:
                    evidence = await asyncio.to_thread(empty_audio_evidence, normalized.path, start, end)
                    if evidence is not None:
                        call_empty_evidence = evidence
                        result = replace(result, warnings=[*result.warnings, "confirmed_empty_audio"])
                        empty_audio_records.append({"method": "empty_audio", "start": start, "end": end, "accepted": True, "evidence": evidence})
            # Persist raw recognition before a separate aligner call. Worker
            # failure during alignment cannot erase usable text or rerun ASR.
            await asyncio.to_thread(checkpoints.finish_call, call_key, result, evidence=call_empty_evidence)
        if result.text.strip() and not result.timestamps and not result.generation.truncated and "mock_adapter" not in result.warnings:
            try:
                aligned = await manager.align_text(AudioPath(normalized.path, start, end), text=result.text,
                    model_id=checked.selected_model, backend=request.backend, language=result.language, timestamps="auto")
                if not aligned:
                    raise AsrError(422, "alignment_failed", "aligner returned no real timestamps")
                result = replace(result, timestamps=aligned, warnings=[warning for warning in result.warnings if warning != "alignment_pending"])
                await asyncio.to_thread(checkpoints.save_aligned_call, call_key, result)
            except AsrError as error:
                if error.code not in {"alignment_failed", "alignment_required", "capability_not_supported"}:
                    raise
                result = replace(result, warnings=[*result.warnings, "alignment_pending"])
        return result

    for index, chunk in enumerate(chunks):
        if cancel_event.is_set():
            raise asyncio.CancelledError
        cached = cached_results.get(index)
        if cached is None:
            if before_chunk:
                await before_chunk(index, len(chunks))
            if stage_callback:
                await stage_callback("transcribing", {"total_chunks": len(chunks), "completed_chunks": len(resolved), "resolved_chunks": len(resolved), "attempted_chunks": len(attempted), "current_chunk": index + 1, "current_chunk_start": chunk.start, "current_chunk_end": chunk.end, "gaps": gaps})
        emit("quality.checkpoint_hit" if cached is not None else "quality.checkpoint_miss", chunk_index=index)
        chunk_extra = 0.0
        causes: list[str] = []
        if cached is not None:
            result, chunk_reviews, _cached_extra = cached
        else:
            chunk_reviews = []
            empty_records_start = len(empty_audio_records)
            initial_key = f"chunk:{index}:initial"
            if checkpoints.recovery.get("legacy_review_attempts", {}).get(f"chunk:{index}", 0) and checkpoints.call(initial_key) is None:
                raise AsrError(409, "legacy_attempt_evidence_required", "legacy failed attempts need an auditable cause import before recovery", {"chunk_index": index, "next_action": "import_legacy_failure"})
            result = TranscriptionResult("", chunk.duration, request.language, [])
            try:
                result = await recognize(chunk.start, chunk.end, call_key=initial_key)
                causes = _issues(result, request.glossary_snapshot)
                if "alignment_pending" in result.warnings:
                    causes.append("alignment_failed")
            except AsrError as error:
                if error.code not in _RECOVERABLE:
                    raise
                causes = [error.code]
            initial_causes = list(causes)
            evidence = language_recovery_evidence(chunks, results, resolved, legacy_indices={position for position in range(len(chunks)) if f"chunk-{position:05d}.json" in checkpoints.recovery.get("legacy_checkpoint_digests", {})}) if request.language == "auto" and causes == ["empty_transcription"] else None
            actions: list[tuple[str, int, str, dict[str, object]]] = []
            if evidence is not None:
                actions.append(("language_fallback", 1, str(evidence["effective_language"]), evidence))
            actions.extend([("split_2", 2, request.language, {}), ("split_4", 4, request.language, {})])
            for action, parts, effective_language, language_evidence in actions:
                if not request.quality_review_enabled or not any(cause in (_UNRESOLVED - {"alignment_failed"}) | {"suspected_term"} for cause in causes):
                    break
                action_key = f"chunk:{index}:{action}"
                previously_started = action_key in checkpoints.recovery["actions"]
                if checkpoints.action_used(index, action) and not previously_started:
                    continue
                if not previously_started:
                    if not checkpoints.can_reserve_action(index, action, chunk.duration, budget):
                        continue
                    await asyncio.to_thread(checkpoints.reserve_action, index, action, chunk.duration, budget,
                        {"requested_language": request.language, "effective_language": effective_language, "trigger": list(causes), **language_evidence})
                    chunk_extra += chunk.duration
                if before_chunk:
                    await before_chunk(index, len(chunks))
                retry_started = perf_counter()
                candidates: list[TranscriptionResult] = []
                child_records: list[dict[str, object]] = []
                child_causes: list[str] = []
                coverage_checks: list[dict[str, object]] = []
                for part in range(parts):
                    # Integer sample boundaries tile the original interval.
                    first_sample, last_sample = round(chunk.start * 16000), round(chunk.end * 16000)
                    start = (first_sample + (last_sample - first_sample) * part // parts) / 16000
                    end = (first_sample + (last_sample - first_sample) * (part + 1) // parts) / 16000
                    try:
                        candidate = await recognize(start, end, language=effective_language, call_key=f"{action_key}:part:{part}", review=True)
                        candidate_causes = _issues(candidate, request.glossary_snapshot)
                        if "alignment_pending" in candidate.warnings:
                            candidate_causes.append("alignment_failed")
                        if action == "language_fallback" and candidate.text.strip() and candidate.timestamps:
                            aligned_items = sorted(candidate.timestamps, key=lambda item: item.start)
                            uncovered = [(0.0, aligned_items[0].start), *[(left.end, right.start) for left, right in zip(aligned_items, aligned_items[1:])], (aligned_items[-1].end, end - start)]
                            for gap_start, gap_end in uncovered:
                                if gap_end - gap_start <= 6:
                                    continue
                                absolute_start, absolute_end = start + gap_start, start + gap_end
                                negative = await asyncio.to_thread(empty_audio_evidence, normalized.path, absolute_start, absolute_end)
                                if negative is None:
                                    negative = await asyncio.to_thread(non_speech_evidence, normalized.path, absolute_start, absolute_end, speech_detector)
                                coverage_checks.append({"start": absolute_start, "end": absolute_end, "negative_speech_evidence": negative,
                                    "evidence_status": "complete_negative_evidence" if negative is not None else "negative_evidence_not_established",
                                    "acoustic_content": "unclassified", "omitted_speech_confirmed": False, "status": "needs_review"})
                                if negative is None:
                                    candidate_causes.append("suspected_uncovered_audio")
                        child_causes.extend(cause for cause in candidate_causes if cause in _UNRESOLVED)
                        candidates.append(replace(candidate, timestamps=_absolute(candidate.timestamps, start - chunk.start)))
                        child_records.append({"start": start, "end": end, "part_index": part, "status": "unresolved" if any(cause in _UNRESOLVED for cause in candidate_causes) else "resolved", "causes": candidate_causes, "requested_language": request.language, "effective_language": effective_language})
                    except AsrError as error:
                        if error.code not in _RECOVERABLE | {"review_budget_exhausted"}:
                            raise
                        child_causes.append(error.code)
                        child_records.append({"start": start, "end": end, "part_index": part, "status": "unresolved", "causes": [error.code], "requested_language": request.language, "effective_language": effective_language})
                        # Quality failure of one child never skips its siblings.
                accepted = not child_causes and len(candidates) == parts
                record: dict[str, object] = {"chunk_index": index, "attempt": 1 if parts == 2 else 2 if parts == 4 else 1,
                    "causes": list(causes), "start": chunk.start, "end": chunk.end, "extra_audio_seconds": chunk.duration,
                    "elapsed_seconds": perf_counter() - retry_started, "method": "language_fallback" if parts == 1 else "shorter_windows",
                    "action": action, "accepted": accepted, "children": child_records, "coverage_checks": coverage_checks, "requested_language": request.language,
                    "effective_language": effective_language, "recovery_policy_version": RECOVERY_POLICY_VERSION, **language_evidence}
                chunk_reviews.append(record)
                emit("quality.review", chunk_index=index, reason=action, accepted=accepted,
                    recovery_policy_version=RECOVERY_POLICY_VERSION, attempt_id=checkpoints.recovery["actions"][action_key]["attempt_id"],
                    requested_language=request.language, effective_language=effective_language,
                    source_chunk_indices=language_evidence.get("source_chunk_indices"), model_language_labels=language_evidence.get("model_language_labels"),
                    review_audio_seconds=checkpoints.extra_seconds, review_budget_seconds=budget)
                if accepted:
                    result = TranscriptionResult(text="\n".join(candidate.text for candidate in candidates), duration=chunk.duration,
                        language=candidates[0].language, warnings=list(dict.fromkeys(warning for candidate in candidates for warning in candidate.warnings)),
                        timestamps=[item for candidate in candidates for item in candidate.timestamps],
                        timings=TranscriptionTimings(inference_ms=sum(candidate.timings.inference_ms for candidate in candidates), postprocess_ms=sum(candidate.timings.postprocess_ms for candidate in candidates)),
                        generation=replace(candidates[-1].generation, truncated=any(candidate.generation.truncated for candidate in candidates)), context=candidates[-1].context)
                    causes = _issues(result, request.glossary_snapshot)
                    if action == "language_fallback":
                        result = replace(result, warnings=[*result.warnings, "language_fallback"])
                        all_issues.append({"kind": "language_fallback", "chunk_index": index, "start": chunk.start, "end": chunk.end, "status": "needs_review", **language_evidence})
                else:
                    causes = list(dict.fromkeys(child_causes or causes))
            if request.quality_review_enabled and causes == ["empty_transcription"]:
                fallback_evidence = await asyncio.to_thread(non_speech_evidence, normalized.path, chunk.start, chunk.end, speech_detector)
                if fallback_evidence is not None:
                    result = TranscriptionResult("", chunk.duration, request.language, ["confirmed_empty_audio"])
                    empty_audio_records.append({"method": "empty_audio", "start": chunk.start, "end": chunk.end, "accepted": True, "evidence": fallback_evidence})
                    causes = []
            chunk_reviews.extend({**record, "chunk_index": index} for record in empty_audio_records[empty_records_start:])
            if any(cause in _UNRESOLVED for cause in causes):
                stop_reason = "review_audio_budget" if checkpoints.extra_seconds + chunk.duration > budget + 0.001 else "review_attempt_limit"
                if "alignment_failed" in causes:
                    stop_reason = "alignment_pending"
                gap: dict[str, object] = {"chunk_index": index, "start": chunk.core_start, "end": chunk.core_end,
                    "start_sample": round(chunk.core_start * 16000), "end_sample": round(chunk.core_end * 16000),
                    "reason": causes[0], "initial_cause": initial_causes[0] if initial_causes else causes[0], "stop_reason": stop_reason,
                    "alignment_status": "pending" if "alignment_failed" in causes else "unavailable"}
                gaps.append(gap)
                await asyncio.to_thread(checkpoints.interval, index, {**gap, "resolution": "unresolved", "reviews": chunk_reviews})
            else:
                await asyncio.to_thread(checkpoints.save, index, result, chunk_reviews, chunk_extra)
                await asyncio.to_thread(checkpoints.interval, index, {"chunk_index": index, "start_sample": round(chunk.core_start * 16000), "end_sample": round(chunk.core_end * 16000), "resolution": "empty_with_evidence" if not result.text.strip() else "resolved", "alignment_status": "aligned" if result.timestamps else "unavailable"})
                emit("chunk.completed", chunk_index=index, start_s=chunk.start, end_s=chunk.end)
        reviews.extend(chunk_reviews)
        for record in chunk_reviews:
            empty_evidence = record.get("evidence")
            if record.get("method") == "empty_audio" and isinstance(empty_evidence, dict):
                kind = {"near_digital_silence_tail": "near_silent_tail", "silero_no_speech_detected": "empty_without_detected_speech"}.get(str(empty_evidence.get("method")))
                if kind:
                    all_issues.append({"kind": kind, "start": record["start"], "end": record["end"], "chunk_index": index, "status": "needs_review", "evidence": empty_evidence})
        if "language_fallback" in result.warnings and not any(issue.get("chunk_index") == index and issue["kind"] == "language_fallback" for issue in all_issues):
            all_issues.append({"kind": "language_fallback", "chunk_index": index, "start": chunk.start, "end": chunk.end, "status": "needs_review"})
        for cause in _issues(result, request.glossary_snapshot):
            all_issues.append({"kind": cause, "start": chunk.core_start, "end": chunk.core_end, "chunk_index": index, "status": "needs_review"})
        results.append(result)
        if not any(gap["chunk_index"] == index for gap in gaps):
            resolved.add(index)
            if after_chunk:
                await after_chunk(index, len(chunks), result)
        attempted.add(index)
        if stage_callback:
            await stage_callback("transcribing", {"total_chunks": len(chunks), "completed_chunks": len(resolved), "resolved_chunks": len(resolved), "attempted_chunks": len(attempted), "gaps": list(gaps)})
    extra_seconds = checkpoints.extra_seconds

    if stage_callback:
        await stage_callback("merging", {})
    # Long acoustic regions without aligned text are an uncertainty signal,
    # not proof of omitted speech (music can also have energy). Review a
    # bounded original-audio excerpt and expose its candidate for listening;
    # never silently inject an unverified music/silence hallucination.
    for chunk, result in zip(chunks, results, strict=True):
        if chunk.index not in resolved:
            continue
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
            action_key = f"chunk:{chunk.index}:{gap_key}"
            legacy_completed = f"chunk-{chunk.index:05d}.json" in checkpoints.recovery.get("legacy_checkpoint_digests", {})
            if not legacy_completed and (action_key in checkpoints.recovery["actions"] or checkpoints.can_reserve_action(chunk.index, gap_key, review_end - start, budget)):
                if before_chunk:
                    await before_chunk(chunk.index, len(chunks))
                emit("quality.review", chunk_index=chunk.index, reason="acoustic_gap", review_budget_seconds=budget)
                if action_key not in checkpoints.recovery["actions"]:
                    await asyncio.to_thread(checkpoints.reserve_action, chunk.index, gap_key, review_end - start, budget, {"method": "acoustic_gap", "start": start, "end": review_end})
                extra_seconds = checkpoints.extra_seconds
                try:
                    candidate = await recognize(start, review_end, call_key=action_key, review=True)
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
        if index not in resolved or index + 1 not in resolved:
            continue
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
        legacy_completed = any(f"chunk-{position:05d}.json" in checkpoints.recovery.get("legacy_checkpoint_digests", {}) for position in (index, index + 1))
        for attempt in ([] if legacy_completed else [1, 2]):
            start = max(left_chunk.start, left_chunk.core_end - (4 - attempt))
            end = min(right_chunk.end, left_chunk.core_end + (4 - attempt))
            action = f"{boundary_key}:attempt:{attempt}"
            action_key = f"chunk:{index}:{action}"
            if action_key not in checkpoints.recovery["actions"] and not checkpoints.can_reserve_action(index, action, end - start, budget):
                break
            if before_chunk:
                await before_chunk(index, len(chunks))
            emit("quality.review", chunk_index=index, reason="boundary", review_budget_seconds=budget, attempt=attempt)
            if action_key not in checkpoints.recovery["actions"]:
                await asyncio.to_thread(checkpoints.reserve_action, index, action, end - start, budget, {"method": "boundary_window", "start": start, "end": end})
            extra_seconds = checkpoints.extra_seconds
            try:
                reviewed = await recognize(start, end, call_key=action_key, review=True)
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
        owned_items: list[TimestampItem]
        if chunk.index not in resolved and not result.timestamps:
            raw_text, owned_items = result.text, []
        else:
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
            normalization_key = f"normalized:{chunk.index}:{fingerprint(text)}"
            saved_alignment = await asyncio.to_thread(checkpoints.call, normalization_key)
            if saved_alignment is not None and saved_alignment.get("status") == "aligned":
                normalized_result = checkpoints.result_from_payload(saved_alignment["result"])
                final_items = _absolute(normalized_result.timestamps, chunk.core_start)
            else:
                normalized_result = replace(result, text=text, timestamps=[])
                if saved_alignment is None:
                    await asyncio.to_thread(checkpoints.begin_call, normalization_key,
                        {"operation": "align_text", "start": chunk.core_start, "end": chunk.core_end,
                         "requested_language": request.language, "effective_language": result.language,
                         "text_fingerprint": fingerprint(text)}, budget=budget)
                    await asyncio.to_thread(checkpoints.finish_call, normalization_key, normalized_result)
                try:
                    aligned = await manager.align_text(AudioPath(normalized.path, chunk.core_start, chunk.core_end), text=text, model_id=checked.selected_model, backend=request.backend, language=result.language, timestamps="auto")
                    if not aligned:
                        raise AsrError(422, "alignment_failed", "normalized text alignment returned no timestamps")
                    normalized_result = replace(normalized_result, timestamps=aligned)
                    await asyncio.to_thread(checkpoints.save_aligned_call, normalization_key, normalized_result)
                    final_items = _absolute(aligned, chunk.core_start)
                except AsrError as error:
                    if error.code not in {"alignment_failed", "alignment_required", "capability_not_supported"}:
                        raise
                    final_items = []
                    if not any(gap["chunk_index"] == chunk.index for gap in gaps):
                        gaps.append({"chunk_index": chunk.index, "start": chunk.core_start, "end": chunk.core_end,
                            "start_sample": round(chunk.core_start * 16000), "end_sample": round(chunk.core_end * 16000),
                            "reason": "alignment_failed", "stop_reason": "alignment_pending", "alignment_status": "pending"})
        if text.strip() and not final_items:
            if "mock_adapter" not in result.warnings and chunk.index in resolved and not any(gap["chunk_index"] == chunk.index for gap in gaps):
                raise AsrError(422, "alignment_failed", "quality transcript requires real timestamp alignment", {"chunk_index": chunk.index})
            all_issues.append({"kind": "alignment_unavailable_mock" if "mock_adapter" in result.warnings else "alignment_pending", "start": chunk.core_start, "end": chunk.core_end, "status": "needs_review"})
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
    quality: dict[str, object] = {"profile_version": PROFILE_VERSION, "recovery_policy_version": RECOVERY_POLICY_VERSION, "validated": False, "fingerprint": identity, "status": "needs_review" if all_issues else "complete", "issues": all_issues, "changes": changes, "reviews": reviews, "review_audio_seconds": extra_seconds, "review_budget_seconds": budget, "model_revision": model_revision, "aligner_revision": ALIGNER_REVISION, "speech_detector": speech_detector}
    metadata = await asyncio.to_thread(inspect_audio_path, normalized.path)
    payload: dict[str, object] = {
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

    completeness: dict[str, object] = {"complete": not gaps, "resolved_chunks": len(resolved), "total_chunks": len(chunks), "gap_count": len(gaps), "attempted_chunks": len(chunks)}
    retryability = "retry" if any(gap["alignment_status"] == "pending" for gap in gaps) else "exhausted" if gaps else "complete"
    next_action = "resume_unresolved" if retryability == "retry" else "listen_or_change_parameters" if gaps else "none"
    payload.update({"completeness": completeness, "gaps": gaps, "retryability": retryability, "next_action": next_action})
    if gaps:
        first = gaps[0]
        raise AsrError(422, "quality_incomplete", "audio intervals remain unresolved after bounded automatic recovery", {
            "start": first["start"], "end": first["end"], "chunk_index": first["chunk_index"], "cause": first["reason"], "stop_reason": first["stop_reason"],
            "review_audio_seconds": extra_seconds, "review_budget_seconds": budget, "partial_result": payload,
            "completeness": completeness, "gaps": gaps, "retryability": retryability, "next_action": next_action})
    return payload
