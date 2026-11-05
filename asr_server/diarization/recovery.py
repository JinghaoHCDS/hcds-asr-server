"""Durable MOSS results and bounded, speaker-aware review of coverage candidates.

Raw model output is immutable. Cached raw results rebuild Anchor Replay in the
same order, so a resumed block never borrows a label from another namespace.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any, TYPE_CHECKING

from asr_server.adapters.base import AudioComposition, AudioInput, AudioPath, AudioSilence, TranscriptionResult, TranscriptionSegment, TranscriptionTimings
from asr_server.audio.assets import NormalizedAudioAsset
from asr_server.diarization.anchor_replay import AnchorChunk, AnchorReplaySequence
from asr_server.errors import AsrError
from asr_server.observability import emit
from asr_server.quality.checkpoints import Checkpoints, atomic_json, fingerprint

if TYPE_CHECKING:
    from asr_server.lifecycle import ModelLifecycleManager
    from asr_server.transcription import TranscriptionRequest, ValidatedTranscription

POLICY_VERSION = "moss-recovery-v1"
REVIEW_ACTIONS_PER_CHUNK = 3
REVIEW_FRACTION = .30
ResultCallback = Callable[[int, int, TranscriptionResult], Awaitable[None]]
ProgressCallback = Callable[[int, int], Awaitable[None]]


class MossCheckpoints:
    def __init__(self, root: Path, identity: dict[str, Any]) -> None:
        self.root = root / POLICY_VERSION
        self.identity = fingerprint(identity)
        manifest = self.root / 'identity.json'
        if manifest.exists():
            if Checkpoints._read(manifest).get('fingerprint') != self.identity:
                raise AsrError(409, 'checkpoint_version_mismatch', 'saved MOSS input, model, parameters or recovery version differs')
        else:
            atomic_json(manifest, {'fingerprint': self.identity, 'policy_version': POLICY_VERSION})
        self.ledger_path = self.root / 'reviews.json'
        self.ledger: dict[str, Any] = Checkpoints._read(self.ledger_path) if self.ledger_path.exists() else {'seconds': 0., 'actions': {}}

    def load(self, name: str, input_identity: str) -> dict[str, Any] | None:
        path = self.root / (name + '.json')
        if not path.exists():
            return None
        data = Checkpoints._read(path)
        if data.get('identity') != self.identity or data.get('input') != input_identity:
            raise AsrError(409, 'checkpoint_version_mismatch', 'MOSS checkpoint belongs to another input window')
        payload = data.get('payload')
        if not isinstance(payload, dict) or fingerprint(payload) != data.get('digest'):
            raise AsrError(409, 'checkpoint_corrupt', 'MOSS checkpoint failed integrity validation')
        return payload

    def save(self, name: str, input_identity: str, payload: dict[str, Any]) -> None:
        atomic_json(self.root / (name + '.json'), {'identity': self.identity, 'input': input_identity,
                    'digest': fingerprint(payload), 'payload': payload})

    def reserve(self, key: str, index: int, seconds: float, budget: float) -> bool:
        actions = self.ledger['actions']
        if key in actions:
            # Interrupted calls are charged; do not silently repeat them.
            return False
        if sum(value['chunk_index'] == index for value in actions.values()) >= REVIEW_ACTIONS_PER_CHUNK:
            return False
        if self.ledger['seconds'] + seconds > budget + .001:
            return False
        self.ledger['seconds'] += seconds
        actions[key] = {'chunk_index': index, 'audio_seconds': seconds}
        atomic_json(self.ledger_path, self.ledger)
        return True


def _input_identity(audio: AudioInput) -> str:
    if isinstance(audio, AudioPath):
        return fingerprint({'start': audio.start, 'end': audio.end})
    assert isinstance(audio, AudioComposition)
    return fingerprint({'prefix': audio.prefix_duration, 'parts': [
        {'start': part.start, 'end': part.end} if isinstance(part, AudioPath) else {'silence': part.duration}
        for part in audio.parts]})


def _gap_intervals(error: dict[str, Any], result: TranscriptionResult, prefix: float) -> list[dict[str, Any]]:
    details = error.get('details', {})
    intervals = list(details.get('uncovered_speech_intervals', []))
    if float(details.get('uncovered_tail_seconds', 0)) > float(details.get('allowed_tail_seconds', 60)):
        intervals.append({'start': float(details.get('last_segment_end_seconds', prefix)),
                          'end': float(details.get('audible_speech_end_seconds', result.duration))})
    if not intervals:
        intervals = [{'start': prefix, 'end': result.duration}]
    # Overlapping candidates are one review region, not repeated GPU work.
    merged: list[dict[str, Any]] = []
    for interval in sorted(intervals, key=lambda value: value['start']):
        start, end = max(0., float(interval['start'])), min(result.duration, float(interval['end']))
        if end <= start:
            continue
        if merged and start <= merged[-1]['end']:
            merged[-1]['end'] = max(merged[-1]['end'], end)
        else:
            merged.append({'start': start, 'end': end, 'reason': error['code'], 'evidence': 'candidate_audio_activity'})
    return merged


def _review_input(chunk: AnchorChunk, result: TranscriptionResult, prefix: float,
                  gap: dict[str, Any]) -> tuple[AudioComposition, list[tuple[str, float, float]], float]:
    start = max(prefix, float(gap['start']) - 4)
    end = min(result.duration, float(gap['end']) + 4, start + 120)
    parts: list[AudioPath | AudioSilence] = []
    references: list[tuple[str, float, float]] = []
    cursor = 0.
    candidates: dict[str, TranscriptionSegment] = {}
    for segment in result.segments:
        if segment.speaker is None or segment.start < prefix or segment.end - segment.start < 2:
            continue
        if any(other is not segment and other.start < segment.end and other.end > segment.start
               for other in result.segments):
            continue
        previous = candidates.get(segment.speaker)
        if previous is None or segment.end - segment.start > previous.end - previous.start:
            candidates[segment.speaker] = segment
    for speaker, segment in sorted(candidates.items())[:4]:
        duration = min(8., segment.end - segment.start)
        source_start = chunk.start + segment.start - prefix
        parts.extend([AudioPath(chunk.audio_path, source_start, source_start + duration), AudioSilence(.5)])
        references.append((speaker, cursor, cursor + duration))
        cursor += duration + .5
    parts.append(AudioPath(chunk.audio_path, chunk.start + start - prefix, chunk.start + end - prefix))
    return AudioComposition(tuple(parts), cursor), references, start


def _patch_gap(original: TranscriptionResult, reviewed: TranscriptionResult, gap: dict[str, Any],
               references: list[tuple[str, float, float]], prefix: float, window_start: float) -> TranscriptionResult | None:
    mapping: dict[str, str] = {}
    conflicting: set[str] = set()
    for original_speaker, start, end in references:
        overlaps: dict[str, float] = {}
        for segment in reviewed.segments:
            if segment.speaker is not None:
                overlap = max(0., min(end, segment.end) - max(start, segment.start))
                overlaps[segment.speaker] = overlaps.get(segment.speaker, 0.) + overlap
        if overlaps:
            speaker, amount = max(overlaps.items(), key=lambda item: item[1])
            if amount > (end - start) * .5:
                if speaker in mapping and mapping[speaker] != original_speaker:
                    conflicting.add(speaker)
                mapping[speaker] = original_speaker
    additions = []
    for segment in reviewed.segments:
        if segment.start < prefix or not segment.text.strip():
            continue
        candidate = replace(segment, start=segment.start - prefix + window_start,
                            end=segment.end - prefix + window_start,
                            speaker=mapping.get(segment.speaker or '') if segment.speaker not in conflicting else None)
        if candidate.start >= gap['end'] or candidate.end <= gap['start']:
            continue
        # Never overwrite context or manufacture clipped timestamps. A competing
        # reading of an already transcribed span stays a visible review issue.
        if any(existing.start < candidate.end and existing.end > candidate.start for existing in original.segments):
            return None
        additions.append(candidate)
    if not additions:
        return None
    # The original detector's condition was ANY segment overlap; retain it for
    # short candidates. Long missing tails require a separate bounded recovery.
    if gap['end'] - gap['start'] > 2:
        return None
    segments = sorted([*original.segments, *additions], key=lambda item: (item.start, item.end))
    return replace(original, text='\n'.join(item.text for item in segments), segments=segments,
                   warnings=list(dict.fromkeys([*original.warnings, 'moss_local_coverage_review'])))


async def run_moss_chunks(*, manager: ModelLifecycleManager, request: TranscriptionRequest,
                          checked: ValidatedTranscription, chunks: Sequence[AnchorChunk],
                          asset: NormalizedAudioAsset, directory: Path, max_new_tokens: int,
                          sequence: AnchorReplaySequence | None, cancel_event: threading.Event,
                          before_chunk: ProgressCallback | None, after_chunk: ResultCallback | None,
                          ) -> tuple[str, list[TranscriptionResult], TranscriptionTimings, list[dict[str, Any]], dict[str, Any]]:
    started = perf_counter()
    identity = {'policy': POLICY_VERSION, 'audio': {key: value for key, value in asset.to_metadata().items() if key != 'owner'},
                'model': manager.runtime_for(checked.selected_model).definition.identity(), 'request': asdict(request),
                'max_new_tokens': max_new_tokens, 'chunks': [(item.start, item.end) for item in chunks]}
    checkpoints = await asyncio.to_thread(MossCheckpoints, directory, identity)
    results: list[TranscriptionResult] = []
    gaps: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    backend = checked.resolved_backend
    total_inference = total_load = 0.
    budget = asset.duration * REVIEW_FRACTION

    def check_cancel() -> None:
        if cancel_event.is_set():
            raise asyncio.CancelledError

    async def recognize(audio: AudioInput) -> TranscriptionResult:
        nonlocal backend, total_inference, total_load
        check_cancel()
        backend, items, timings = await manager.transcribe_chunks([audio], model_id=checked.selected_model,
            backend=request.backend, language=request.language, timestamps=request.timestamps,
            context=checked.adapter_context, max_new_tokens=max_new_tokens, batch_size=1)
        total_inference += timings.inference_ms
        total_load += timings.load_ms
        check_cancel()
        return items[0]

    for index, chunk in enumerate(chunks):
        check_cancel()
        if before_chunk:
            await before_chunk(index, len(chunks))
        audio = sequence.audio_for(index) if sequence else AudioPath(chunk.audio_path, chunk.start, chunk.end)
        prefix = audio.prefix_duration if isinstance(audio, AudioComposition) else 0.
        input_id = _input_identity(audio)
        name = f'chunk-{index:05d}'
        saved = await asyncio.to_thread(checkpoints.load, name, input_id)
        if saved is None:
            try:
                raw = await recognize(audio)
                error: dict[str, Any] = {}
            except AsrError as exc:
                if exc.code not in {'incomplete_transcript', 'generation_truncated', 'empty_transcription'}:
                    raise
                raw = exc.partial_transcription or TranscriptionResult('', chunk.end - chunk.start + prefix,
                    request.language, [exc.code])
                total_inference += raw.timings.inference_ms
                error = {'code': exc.code, 'details': {key: value for key, value in exc.details.items()
                    if key in {'uncovered_speech_intervals', 'uncovered_speech_interval_count', 'uncovered_tail_seconds',
                               'allowed_tail_seconds', 'last_segment_end_seconds', 'audible_speech_end_seconds'}}}
            check_cancel()
            saved = {'result': asdict(raw), 'error': error}
            await asyncio.to_thread(checkpoints.save, name, input_id, saved)
        else:
            emit('chunk.reused', chunk_index=index, total_chunks=len(chunks))
        raw = Checkpoints.result_from_payload(saved['result'])
        error = saved.get('error', {})
        unresolved = _gap_intervals(error, raw, prefix) if error else []
        for gap_index, gap in enumerate(unresolved):
            if gap['start'] < prefix or gap['end'] - gap['start'] > 2 or not raw.segments:
                continue
            review_audio, references, window_start = _review_input(chunk, raw, prefix, gap)
            key = f'review-{index:05d}-{gap_index:05d}'
            review_id = fingerprint({'input': _input_identity(review_audio), 'raw': fingerprint(saved), 'gap': gap})
            stored_review = await asyncio.to_thread(checkpoints.load, key, review_id)
            if stored_review is None and await asyncio.to_thread(checkpoints.reserve, key, index, review_audio.duration, budget):
                try:
                    candidate = await recognize(review_audio)
                    stored_review = {'result': asdict(candidate), 'error': None}
                except AsrError as exc:
                    if exc.code not in {'incomplete_transcript', 'generation_truncated', 'empty_transcription'}:
                        raise
                    stored_review = {'result': None, 'error': exc.code}
                check_cancel()
                await asyncio.to_thread(checkpoints.save, key, review_id, stored_review)
            patched = None
            if stored_review and stored_review.get('result'):
                candidate = Checkpoints.result_from_payload(stored_review['result'])
                patched = _patch_gap(raw, candidate, gap, references, review_audio.prefix_duration, window_start)
            accepted = patched is not None
            reviews.append({'chunk_index': index, 'start': chunk.start + gap['start'] - prefix,
                            'end': chunk.start + gap['end'] - prefix, 'accepted': accepted, 'method': 'anchor_window'})
            if patched is not None:
                raw = patched
                gap['resolved'] = True
        for gap in unresolved:
            if not gap.get('resolved'):
                gaps.append({'chunk_index': index, 'start': max(chunk.start, chunk.start + gap['start'] - prefix),
                    'end': min(chunk.end, max(chunk.start + .001, chunk.start + gap['end'] - prefix)),
                    'reason': gap['reason'], 'evidence': gap['evidence'], 'stop_reason': 'local_review_unresolved',
                    'alignment_status': 'segment_timed', 'status': 'needs_review'})
        if unresolved:
            raw = replace(raw, generation=replace(raw.generation,
                uncovered_speech_intervals=sum(not gap.get("resolved") for gap in unresolved)))
        if any(not gap.get("resolved") for gap in unresolved):
            raw = replace(raw, warnings=list(dict.fromkeys([*raw.warnings, "moss_coverage_pending"])))
        result = sequence.accept(index, raw) if sequence else raw
        results.append(result)
        if after_chunk:
            await after_chunk(index, len(chunks), result)
    timings = TranscriptionTimings(total_ms=(perf_counter() - started) * 1000, load_ms=total_load, inference_ms=total_inference)
    return backend, results, timings, gaps, {'policy_version': POLICY_VERSION, 'reviews': reviews,
        'review_audio_seconds': checkpoints.ledger['seconds'], 'review_budget_seconds': budget,
        'status': 'needs_review' if gaps else 'complete'}
