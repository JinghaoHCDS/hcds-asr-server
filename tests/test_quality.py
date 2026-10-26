from __future__ import annotations

import io
import wave
from array import array
from dataclasses import replace
from pathlib import Path

import pytest

from asr_server.adapters.base import AudioInput, AudioPath, GenerationMetrics, TimestampItem, TranscriptionResult
from asr_server.adapters.mock import MockAsrAdapter
from asr_server.adapters.qwen import _QwenWorkerBackend
from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.lifecycle import ModelLifecycleManager
from asr_server.quality.checkpoints import Checkpoints
from asr_server.quality.glossary import glossary_context, normalize_terms
from asr_server.quality.pipeline import _owned_text
from asr_server.quality.splitter import QualityChunk, empty_audio_evidence, plan_quality_chunks
from asr_server.quality.segments import reading_segments
from asr_server.registry import default_models
from asr_server.transcription import TranscriptionRequest, run_transcription_path, validate_transcription_request


def wav_bytes(seconds: float, amplitude: int = 0) -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(amplitude.to_bytes(2, "little", signed=True) * int(seconds * 16000))
    return target.getvalue()


def request(**changes: object) -> TranscriptionRequest:
    original = TranscriptionRequest(
        model="qwen3-asr-1.7b", language="auto", response_format="verbose_json", timestamps="auto",
        backend="auto", max_new_tokens=None, context="", hotwords=None, split_strategy="auto",
        max_chunk_seconds=None, overlap_seconds=None, preserve_segments=True, quality_profile="quality_v1",
    )
    return replace(original, **changes)  # type: ignore[arg-type]


def test_low_energy_planner_preserves_all_samples_and_bounded_context(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(370))
    chunks, duration = plan_quality_chunks(source, target=180, padding=1)
    assert duration == 370
    assert chunks[0].core_start == 0
    assert chunks[-1].core_end == duration
    assert all(left.core_end == right.core_start for left, right in zip(chunks, chunks[1:]))
    assert all(chunk.duration <= 180 for chunk in chunks)
    assert sum(chunk.core_end - chunk.core_start for chunk in chunks) == duration


def test_planner_prefers_quiet_boundary_without_discarding_quiet_audio(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x10\x10" * (27 * 16000))
        target.writeframes(b"\x00\x00" * 16000)
        target.writeframes(b"\x10\x10" * (32 * 16000))
    chunks, duration = plan_quality_chunks(source, target=30)
    assert chunks[0].core_end == pytest.approx(27.9)
    assert duration == 60
    assert chunks[1].core_start == chunks[0].core_end


def test_alias_normalization_is_non_cascading_and_rejects_collisions() -> None:
    snapshot: dict[str, object] = {"entries": [
        {"canonical": "Qwen3-ASR", "aliases": ["千问 ASR"], "auto_normalize": True},
        {"canonical": "Other", "aliases": ["千问 ASR"], "auto_normalize": True},
        {"canonical": "ASR", "aliases": ["语音识别"], "auto_normalize": True},
        {"canonical": "Automatic Speech Recognition", "aliases": ["ASR"], "auto_normalize": True},
    ]}
    normalized, changes, unresolved = normalize_terms("千问 ASR，语音识别", snapshot)
    assert normalized == "千问 ASR，ASR"
    assert len(changes) == 1
    assert unresolved == ["千问 ASR"]


def test_latin_alias_respects_word_boundaries() -> None:
    snapshot: dict[str, object] = {"entries": [{"canonical": "Qwen", "aliases": ["qwen"], "auto_normalize": True}]}
    normalized, _changes, _unresolved = normalize_terms("qwen qwenish", snapshot)
    assert normalized == "Qwen qwenish"


def test_user_hotwords_precede_snapshot_and_omissions_are_reported() -> None:
    snapshot: dict[str, object] = {"version": 2, "entries": [{"canonical": f"word-{i}"} for i in range(110)]}
    prompt, report = glossary_context("", "personal", snapshot)
    assert "personal" in prompt
    used = report["used"]
    assert isinstance(used, list) and used[0] == "personal" and len(used) == 100
    assert report["omitted"]


def test_glossary_reporting_matches_casefolded_model_prompt() -> None:
    snapshot: dict[str, object] = {"entries": [{"canonical": "qwen"}, {"canonical": "Other"}]}
    _prompt, report = glossary_context("", "Qwen", snapshot)
    assert report["used"] == ["Qwen", "Other"]
    assert report["omitted"] == []
    summary = report["context"]
    assert isinstance(summary, dict) and summary["hotword_count"] == 2


def test_timestamp_ownership_keeps_real_repeated_words_and_punctuation() -> None:
    result = TranscriptionResult("你好，你好。再见！", 4, "zh", [], timestamps=[
        TimestampItem("你好", 0, 1, "word"), TimestampItem("你好", 1, 2, "word"), TimestampItem("再见", 2, 3, "word"),
    ])
    text, items = _owned_text(result, QualityChunk(1, 0, 4, 1, 4))
    assert text == "你好。再见！"
    assert [item.text for item in items] == ["你好", "再见"]


def test_checkpoint_mismatch_and_retry_budget_survive_restart(tmp_path: Path) -> None:
    checkpoint = Checkpoints(tmp_path, "identity")
    checkpoint.reserve_review(3, 10)
    restored = Checkpoints(tmp_path, "identity")
    assert restored.extra_seconds == 3
    with pytest.raises(AsrError) as error:
        restored.reserve_review(8, 10)
    assert error.value.code == "review_budget_exhausted"
    with pytest.raises(AsrError) as error:
        Checkpoints(tmp_path, "another-model-revision")
    assert error.value.code == "checkpoint_version_mismatch"


class RecordingAdapter(MockAsrAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int | None] = []
        self.aligned: list[str] = []

    async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
        del model_id, backend, language, context, timestamps
        self.calls.append(max_new_tokens)
        return TranscriptionResult("千问 ASR。", 3, "zh", [], timestamps=[TimestampItem("千问", 0, 1, "word"), TimestampItem("ASR", 1, 2, "word")], generation=GenerationMetrics(max_new_tokens=max_new_tokens))

    async def align_text(self, audio: AudioInput, *, text: str, language: str = "auto", timestamps: str = "auto") -> list[TimestampItem]:
        del audio, language, timestamps
        self.aligned.append(text)
        return [TimestampItem("Qwen3-ASR", 0, 2, "word")]


@pytest.mark.asyncio
async def test_quality_normalization_realigns_and_resumes_completed_chunks(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(3))
    adapter = RecordingAdapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    snapshot: dict[str, object] = {"version": 1, "entries": [{"canonical": "Qwen3-ASR", "aliases": ["千问 ASR"], "auto_normalize": True}]}
    parameters = request(glossary_snapshot=snapshot)
    try:
        first = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=parameters, checkpoint_dir=tmp_path / "checkpoints")
        second = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=parameters, checkpoint_dir=tmp_path / "checkpoints")
        assert first["text"] == second["text"] == "Qwen3-ASR。"
        assert first["raw_text"] == "千问 ASR。"
        assert adapter.calls == [4096]
        assert adapter.aligned == ["Qwen3-ASR。"]  # Persisted normalization alignment is reused.
        assert first["timestamps"] == [TimestampItem("Qwen3-ASR", 0, 2, "word").to_api()]
        assert isinstance(first["quality"], dict) and first["quality"]["validated"] is False
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_quality_respects_explicit_generation_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(3))
    adapter = RecordingAdapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    try:
        await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(max_new_tokens=123))
        assert adapter.calls == [123]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_unrecoverable_missing_audio_never_reports_completed(tmp_path: Path) -> None:
    class EmptyAdapter(RecordingAdapter):
        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            raise AsrError(422, "generation_truncated", "truncated")

    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(3, amplitude=100))
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: EmptyAdapter(), idle_unload_seconds=0)
    try:
        with pytest.raises(AsrError) as error:
            await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request())
        assert error.value.code == "quality_incomplete"
        assert error.value.details["end"] == 3
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_truncation_is_resegmented_within_real_audio_budget(tmp_path: Path) -> None:
    class OnceTruncated(RecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.failed_once = False

        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            if not self.failed_once:
                self.failed_once = True
                raise AsrError(422, "generation_truncated", "truncated first input")
            return await super().transcribe(audio, model_id=model_id, backend=backend, language=language, context=context, timestamps=timestamps, max_new_tokens=max_new_tokens)

    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(240))
    adapter = OnceTruncated()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    try:
        result = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints")
        quality = result["quality"]
        assert isinstance(quality, dict)
        assert quality["review_audio_seconds"] == 60
        assert quality["review_budget_seconds"] == 72
        assert len(adapter.calls) == 5  # Two repaired half-windows plus three untouched chunks.
        assert str(result["text"]).count("千问 ASR") == 5
        restored = Checkpoints(tmp_path / "checkpoints", str(quality["fingerprint"]))
        assert restored.review_attempts["chunk:0"] == 1
    finally:
        await manager.shutdown()


def test_quality_cannot_be_used_for_moss_or_unaligned_deployments() -> None:
    manager = ModelLifecycleManager(default_models(enable_moss=True), lambda _: MockAsrAdapter())
    with pytest.raises(AsrError) as error:
        validate_transcription_request(manager, request(model="moss-transcribe-diarize-0.9b"))
    assert error.value.code == "capability_not_supported"
    with pytest.raises(AsrError) as error:
        validate_transcription_request(manager, request())
    assert error.value.code == "capability_not_supported"


def test_reading_segments_preserve_punctuation_and_original_aliases() -> None:
    chunks: list[dict[str, object]] = [{
        "index": 0, "start": 0, "end": 12, "text": "你好。Qwen3-ASR 很好！再见。", "raw_text": "你好。千问 ASR 很好！再见。",
        "timestamps": [
            TimestampItem("你好", 0, 2, "word").to_api(),
            TimestampItem("Qwen3-ASR", 3, 4, "word").to_api(),
            TimestampItem("很好", 4, 5, "word").to_api(),
            TimestampItem("再见", 9, 10, "word").to_api(),
        ],
    }]
    segments = reading_segments(chunks)
    assert [item["text"] for item in segments] == ["你好。", "Qwen3-ASR 很好！", "再见。"]
    assert segments[1]["raw_text"] == "千问 ASR 很好！"
    assert all(float(str(item["end"])) - float(str(item["start"])) <= 6 for item in segments)


@pytest.mark.asyncio
async def test_qwen_supplied_text_alignment_does_not_transcribe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backend = _QwenWorkerBackend("qwen3-asr-1.7b", enable_alignment=True)
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(1))
    seen: list[str] = []

    def align(path: str, transcript: str, language: str, *, requested_granularity: str) -> tuple[list[TimestampItem], list[str]]:
        del path, language, requested_granularity
        seen.append(transcript)
        return [TimestampItem(transcript, 0, 1, "word")], []

    monkeypatch.setattr(backend, "_align", align)
    aligned = await backend.align_text(AudioPath(source), text="用户修改后的原稿")
    assert seen == ["用户修改后的原稿"]
    assert aligned[0].text == "用户修改后的原稿"


@pytest.mark.parametrize('amplitude', [0, 1, 20])
async def test_quiet_nonzero_audio_is_not_silently_accepted_as_digital_silence(tmp_path: Path, amplitude: int) -> None:
    class EmptyAdapter(RecordingAdapter):
        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            raise AsrError(422, "empty_transcription", "no text")
    source = tmp_path / 'quiet.wav'
    source.write_bytes(wav_bytes(1, amplitude=amplitude))
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: EmptyAdapter(), idle_unload_seconds=0)
    try:
        if amplitude:
            with pytest.raises(AsrError) as error:
                await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request())
            assert error.value.code == 'quality_incomplete'
        else:
            result = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request())
            assert result['text'] == ''
    finally:
        await manager.shutdown()


def write_pcm_samples(path: Path, samples: array[int]) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(samples.tobytes())


@pytest.mark.parametrize(
    ("peak", "nonzero_count", "last_nonzero", "file_seconds", "expected"),
    [
        (0, 0, 0, 10, "digital_silence"),
        (4, 8000, 7999, 10, "near_digital_silence_tail"),  # Exactly 5% nonzero.
        (4, 8001, 8000, 10, None),
        (5, 1, 0, 10, None),
        (4, 1, 0, 11, None),  # A quiet interior interval is never a tail.
        (4, 1, 79999, 10, "near_digital_silence_tail"),  # Exactly half is all zero.
        (4, 1, 80000, 10, None),
        (1, 160000, 159999, 10, None),  # Constant quiet audio has no zero suffix.
    ],
)
def test_empty_audio_evidence_preserves_conservative_tail_boundaries(
    tmp_path: Path, peak: int, nonzero_count: int, last_nonzero: int,
    file_seconds: int, expected: str | None,
) -> None:
    samples = array("h", [0]) * (file_seconds * 16000)
    if nonzero_count:
        samples[:nonzero_count] = array("h", [peak]) * nonzero_count
        if last_nonzero >= nonzero_count:
            samples[nonzero_count - 1] = 0
            samples[last_nonzero] = peak
    source = tmp_path / "tail.wav"
    write_pcm_samples(source, samples)
    evidence = empty_audio_evidence(source, 0, 10)
    if expected is None:
        assert evidence is None
    else:
        assert evidence is not None and evidence["method"] == expected


def test_near_silent_tail_requires_at_least_one_second_of_digital_silence(tmp_path: Path) -> None:
    samples = array("h", [0]) * 16000
    samples[0] = 1
    source = tmp_path / "short-tail.wav"
    write_pcm_samples(source, samples)
    assert empty_audio_evidence(source, 0, 1) is None


@pytest.mark.asyncio
async def test_near_silent_tail_is_recognized_then_delivered_with_review_evidence(tmp_path: Path) -> None:
    class EmptyAdapter(RecordingAdapter):
        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            self.calls.append(max_new_tokens)
            raise AsrError(422, "empty_transcription", "no text")

    samples = array("h", [0]) * 160000
    samples[:1600] = array("h", [4]) * 1600
    source = tmp_path / "tail.wav"
    write_pcm_samples(source, samples)
    adapter = EmptyAdapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    try:
        for _ in range(2):
            result = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints")
            assert result["text"] == "" and result["timestamps"] == []
            quality = result["quality"]
            assert isinstance(quality, dict) and quality["status"] == "needs_review"
            issues = quality["issues"]
            assert isinstance(issues, list)
            tail_issues = [issue for issue in issues if issue["kind"] == "near_silent_tail"]
            assert len(tail_issues) == 1
            issue = tail_issues[0]
            assert issue["status"] == "needs_review" and issue["start"] == 0 and issue["end"] == 10
            evidence = issue["evidence"]
            assert isinstance(evidence, dict) and evidence["method"] == "near_digital_silence_tail"
        assert adapter.calls == [4096]  # Real recognition first; the second run reuses its checkpoint.
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_resegmented_speech_with_silent_subwindow_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class MixedAdapter(RecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.windows: list[tuple[float, float | None]] = []

        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            assert isinstance(audio, AudioPath) and audio.end is not None
            window = (audio.start, audio.end)
            self.windows.append(window)
            if window == (0, 60):
                raise AsrError(422, "generation_truncated", "retry the shorter windows")
            if window == (30, 60):
                raise AsrError(422, "empty_transcription", "silent subwindow")
            text = "第一段。" if window == (0, 30) else "后续段。"
            duration = audio.end - audio.start
            return TranscriptionResult(text, duration, "zh", ["mock_adapter"], timestamps=[TimestampItem(text, 0, duration, "word")])

    # Fixed windows isolate retry behavior from the independently tested planner.
    windows = [QualityChunk(index, index * 60, (index + 1) * 60, index * 60, (index + 1) * 60) for index in range(4)]
    monkeypatch.setattr("asr_server.quality.pipeline.plan_quality_chunks", lambda *_args, **_kwargs: (windows, 240))
    samples = array("h", [100]) * (240 * 16000)
    samples[30 * 16000:60 * 16000] = array("h", [0]) * (30 * 16000)
    source = tmp_path / "speech-and-silence.wav"
    write_pcm_samples(source, samples)
    adapter = MixedAdapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    try:
        result = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request())
        assert str(result["text"]).startswith("第一段。")
        assert str(result["text"]).count("后续段。") == 3
        assert adapter.windows == [(0, 60), (0, 30), (30, 60), (60, 120), (120, 180), (180, 240)]
        quality = result["quality"]
        assert isinstance(quality, dict) and quality["review_audio_seconds"] == 60
        reviews = quality["reviews"]
        assert isinstance(reviews, list) and any(review["method"] == "shorter_windows" and review["accepted"] for review in reviews)
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_overlapping_digital_silence_does_not_require_fabricated_alignment(tmp_path: Path) -> None:
    class EmptyAdapter(RecordingAdapter):
        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            self.calls.append(max_new_tokens)
            raise AsrError(422, "empty_transcription", "digital silence")

    source = tmp_path / "silence.wav"
    source.write_bytes(wav_bytes(61))
    adapter = EmptyAdapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    try:
        result = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(overlap_seconds=1, quality_chunk_seconds=30))
        assert result["text"] == "" and result["timestamps"] == []
        assert len(adapter.calls) == 3  # Every overlapping input was actually recognized.
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_reviewed_empty_audio_uses_independent_evidence_and_keeps_audit_on_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyAdapter(RecordingAdapter):
        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            self.calls.append(max_new_tokens)
            raise AsrError(422, "empty_transcription", "no text")

    detector = {"available": True, "detector": "test-detector"}
    inspected: list[tuple[float, float]] = []
    def evidence(_path: Path, start: float, end: float, expected: dict[str, object]) -> dict[str, object]:
        assert expected == detector
        inspected.append((start, end))
        return {"method": "silero_no_speech_detected", "detector_identity": detector, "views": 2}
    monkeypatch.setattr("asr_server.quality.pipeline.detector_identity", lambda: detector)
    monkeypatch.setattr("asr_server.quality.pipeline.non_speech_evidence", evidence)
    source = tmp_path / "background.wav"
    source.write_bytes(wav_bytes(240, amplitude=1000))
    adapter = EmptyAdapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    try:
        for _ in range(2):
            result = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints")
            assert result["text"] == "" and result["timestamps"] == []
            chunks = result["chunks"]
            assert isinstance(chunks, list) and len(chunks) == 4
            quality = result["quality"]
            assert isinstance(quality, dict) and quality["status"] == "needs_review"
            assert quality["review_audio_seconds"] == 60 and quality["review_budget_seconds"] == 72
            issues = quality["issues"]
            assert isinstance(issues, list) and len(issues) == 4
            assert all(issue["kind"] == "empty_without_detected_speech" and issue["status"] == "needs_review" for issue in issues)
        assert inspected == [(0, 60), (60, 120), (120, 180), (180, 240)]
        assert len(adapter.calls) == 6  # Four full inputs and both bounded review siblings.
        monkeypatch.setattr("asr_server.quality.pipeline.detector_identity", lambda: {**detector, "available": False})
        with pytest.raises(AsrError) as error:
            await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints")
        assert error.value.code == "checkpoint_version_mismatch"
    finally:
        await manager.shutdown()
