from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from asr_server.adapters.base import AudioInput, AudioPath, GenerationMetrics, TimestampItem, TranscriptionResult
from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.lifecycle import ModelLifecycleManager
from asr_server.quality.checkpoints import Checkpoints, atomic_json, file_digest
from asr_server.quality.pipeline import language_recovery_evidence
from asr_server.quality.splitter import QualityChunk
from asr_server.registry import default_models
from asr_server.transcription import run_transcription_path
from tests.test_quality import RecordingAdapter, request, wav_bytes


def fixed_chunks(monkeypatch: pytest.MonkeyPatch, count: int) -> list[QualityChunk]:
    windows = [QualityChunk(i, i * 60, (i + 1) * 60, i * 60, (i + 1) * 60) for i in range(count)]
    monkeypatch.setattr("asr_server.quality.pipeline.plan_quality_chunks", lambda *_a, **_kw: (windows, count * 60))
    monkeypatch.setattr("asr_server.quality.pipeline.non_speech_evidence", lambda *_a, **_kw: None)
    return windows


def result_for(audio: AudioInput, text: str = "已识别。", *, language: str = "zh", truncated: bool = False) -> TranscriptionResult:
    assert isinstance(audio, AudioPath) and audio.end is not None
    duration = audio.end - audio.start
    return TranscriptionResult(text, duration, language, [], timestamps=[TimestampItem(text, 0, duration, "word")], generation=GenerationMetrics(truncated=truncated))


def test_language_recovery_needs_three_consistent_real_model_labels() -> None:
    chunks = [QualityChunk(i, i * 10, (i + 1) * 10, i * 10, (i + 1) * 10) for i in range(3)]
    results = [TranscriptionResult("文字", 10, "Chinese", []) for _ in chunks]
    assert language_recovery_evidence(chunks, results[:2], {0, 1}) is None
    evidence = language_recovery_evidence(chunks, results, {0, 1, 2})
    assert evidence is not None and evidence["effective_language"] == "zh"
    assert evidence["source_chunk_indices"] == [0, 1, 2]
    for changed in (TranscriptionResult("English", 10, "English", []), TranscriptionResult("文字", 10, "zh", ["language_label_missing"]), TranscriptionResult("双语", 10, "zh", ["mixed_language"])):
        assert language_recovery_evidence(chunks, [*results[:2], changed], {0, 1, 2}) is None


@pytest.mark.asyncio
async def test_evidenced_language_recovery_is_charged_once_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_chunks(monkeypatch, 5)
    class Adapter(RecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.windows: list[tuple[float, float | None, str]] = []
        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            assert isinstance(audio, AudioPath)
            self.windows.append((audio.start, audio.end, language))
            if audio.start == 180 and language == "auto":
                raise AsrError(422, "empty_transcription", "auto failed")
            return result_for(audio)
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(300, 100))
    adapter = Adapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    try:
        first = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints")
        second = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints")
        assert first["text"] == second["text"]
        assert len(adapter.windows) == 6
        assert adapter.windows.count((180, 240, "zh")) == 1
        assert first["quality"]["review_audio_seconds"] == 60  # type: ignore[index]
        fallback = [item for item in first["quality"]["reviews"] if item["method"] == "language_fallback"]  # type: ignore[index]
        assert len(fallback) == 1 and fallback[0]["source_chunk_indices"] == [0, 1, 2]
        assert first["completeness"]["complete"] is True  # type: ignore[index]
        assert "language_fallback" in second["warnings"]  # type: ignore[operator]
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_first_failed_child_does_not_skip_sibling_or_hide_child_truncation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_chunks(monkeypatch, 10)
    class Adapter(RecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.windows: list[tuple[float, float | None]] = []
        async def transcribe(self, audio: AudioInput, **kwargs: object) -> TranscriptionResult:
            assert isinstance(audio, AudioPath)
            self.windows.append((audio.start, audio.end))
            if (audio.start, audio.end) == (0, 60):
                raise AsrError(422, "generation_truncated", "base failed")
            # The first half has a returned truncation flag, while the last
            # half is good. Parent resolution must consider both children.
            return result_for(audio, truncated=(audio.start, audio.end) == (0, 30))
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(600, 100))
    adapter = Adapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    try:
        result = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints")
        assert adapter.windows[:7] == [(0, 60), (0, 30), (30, 60), (0, 15), (15, 30), (30, 45), (45, 60)]
        assert result["quality"]["review_audio_seconds"] == 120  # type: ignore[index]
        reviews = result["quality"]["reviews"]  # type: ignore[index]
        assert reviews[0]["accepted"] is False and reviews[1]["accepted"] is True
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_sparse_successes_survive_exhausted_gap_and_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_chunks(monkeypatch, 4)
    class Adapter(RecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.windows: list[tuple[float, float | None]] = []
        async def transcribe(self, audio: AudioInput, **kwargs: object) -> TranscriptionResult:
            assert isinstance(audio, AudioPath) and audio.end is not None
            self.windows.append((audio.start, audio.end))
            if audio.end <= 60:
                raise AsrError(422, "empty_transcription", "speech unrecognized")
            return result_for(audio)
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(240, 100))
    adapter = Adapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    try:
        for run in range(2):
            counts: list[int] = []
            async def stage(phase: str, progress: dict[str, object]) -> None:
                if "completed_chunks" in progress:
                    counts.append(int(str(progress["completed_chunks"])))
            with pytest.raises(AsrError) as caught:
                await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints", stage_callback=stage)
            assert counts[0] == (0 if run == 0 else 3)
            assert counts == sorted(counts), "sparse later successes count before visiting their indices"
            error = caught.value
            assert error.code == "quality_incomplete"
            partial = error.details["partial_result"]
            assert partial["completeness"] == {"complete": False, "resolved_chunks": 3, "total_chunks": 4, "gap_count": 1, "attempted_chunks": 4}
            assert len(partial["gaps"]) == 1 and partial["gaps"][0]["chunk_index"] == 0
            assert partial["text"].count("已识别") == 3
            assert partial["retryability"] == "exhausted"
        assert adapter.windows == [(0, 60), (0, 30), (30, 60), (60, 120), (120, 180), (180, 240)]
        root = tmp_path / "checkpoints" / "recovery-v1"
        assert (root / "chunk-00003.json").exists() and not (root / "chunk-00000.json").exists()
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_raw_transcription_survives_alignment_failure_without_repeating_asr(tmp_path: Path) -> None:
    class Adapter(RecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.asr_calls = 0
            self.align_calls = 0
        async def transcribe(self, audio: AudioInput, **kwargs: object) -> TranscriptionResult:
            self.asr_calls += 1
            return TranscriptionResult("保留原稿。", 3, "zh", [])
        async def align_text(self, audio: AudioInput, *, text: str, language: str = "auto", timestamps: str = "auto") -> list[TimestampItem]:
            self.align_calls += 1
            if self.align_calls == 1:
                raise AsrError(422, "alignment_failed", "injected aligner failure")
            return [TimestampItem(text, 0, 3, "word")]
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(3, 100))
    adapter = Adapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    try:
        with pytest.raises(AsrError) as caught:
            await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints")
        partial = caught.value.details["partial_result"]
        assert partial["raw_text"] == "保留原稿。" and partial["retryability"] == "retry"
        assert partial["timestamps"] == []
        result = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints")
        assert result["text"] == "保留原稿。" and adapter.asr_calls == 1 and adapter.align_calls == 2
    finally:
        await manager.shutdown()


def test_legacy_import_preserves_success_bytes_and_inherits_spent_actions(tmp_path: Path) -> None:
    chunks = [QualityChunk(i, i * 60, (i + 1) * 60, i * 60, (i + 1) * 60) for i in range(4)]
    original = TranscriptionResult("原结果", 60, "Chinese", [], timestamps=[TimestampItem("原结果", 0, 60, "word")])
    atomic_json(tmp_path / "identity.json", {"fingerprint": "base"})
    atomic_json(tmp_path / "chunk-00000.json", {"result": asdict(original), "reviews": [], "extra_seconds": 0})
    atomic_json(tmp_path / "review-budget.json", {"extra_seconds": 120, "attempts": {"chunk:3": 2}})
    digest = file_digest(tmp_path / "chunk-00000.json")
    atomic_json(tmp_path / "recovery-import.json", {"base_fingerprint": "base", "source_error_digest": "saved-job-error-sha256", "failed_chunks": [{"index": 3, "start": 180, "end": 240, "cause": "empty_transcription"}]})
    checkpoint = Checkpoints(tmp_path, "base")
    checkpoint.initialize_recovery({"chunks": [asdict(chunk) for chunk in chunks], "requested_language": "auto"})
    assert checkpoint.load(0)[0].text == "原结果"  # type: ignore[index]
    assert checkpoint.action_used(3, "split_2") and checkpoint.action_used(3, "split_4")
    assert checkpoint.call("chunk:3:initial")["code"] == "empty_transcription"  # type: ignore[index]
    checkpoint.reserve_action(3, "language_fallback", 60, 180, {"effective_language": "zh"})
    restored = Checkpoints(tmp_path, "base")
    assert restored.extra_seconds == 180 and restored.action_count(3) == 3
    with pytest.raises(AsrError):
        restored.reserve_action(3, "language_fallback", 60, 240, {})
    assert file_digest(tmp_path / "chunk-00000.json") == digest
    state = json.loads((tmp_path / "recovery-v1" / "state.json").read_text())
    assert state["legacy_checkpoint_digests"] == {"chunk-00000.json": digest}


def test_reserved_quality_action_and_budget_survive_interrupted_call(tmp_path: Path) -> None:
    checkpoint = Checkpoints(tmp_path, "base")
    checkpoint.initialize_recovery({"chunks": []})
    checkpoint.reserve_action(0, "split_2", 60, 100, {})
    checkpoint.begin_call("child", {"start": 0, "end": 30}, budget=100)
    restored = Checkpoints(tmp_path, "base")
    assert restored.extra_seconds == 60 and restored.action_used(0, "split_2")
    restored.begin_call("child", {"start": 0, "end": 30}, budget=100, review_seconds=30)
    assert restored.extra_seconds == 90
    with pytest.raises(AsrError) as caught:
        restored.begin_call("child", {"start": 0, "end": 30}, budget=100, review_seconds=30)
    assert caught.value.code == "recovery_attempt_exhausted"


@pytest.mark.asyncio
async def test_language_recovery_nonempty_text_cannot_hide_uncovered_speech(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_chunks(monkeypatch, 5)
    class Adapter(RecordingAdapter):
        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            assert isinstance(audio, AudioPath)
            if audio.start == 180:
                if language == "auto":
                    raise AsrError(422, "empty_transcription", "auto failed")
                return TranscriptionResult("少量候选。", 60, "zh", [], timestamps=[TimestampItem("少量候选", 0, 2, "word")])
            return result_for(audio)
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(300, 100))
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: Adapter(), idle_unload_seconds=0)
    try:
        with pytest.raises(AsrError) as caught:
            await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints")
        partial = caught.value.details["partial_result"]
        assert partial["completeness"]["resolved_chunks"] == 4
        assert partial["gaps"][0]["reason"] == "suspected_uncovered_audio"
        fallback = [record for record in partial["quality"]["reviews"] if record["method"] == "language_fallback"]
        assert len(fallback) == 1 and not fallback[0]["accepted"]
        assert fallback[0]["coverage_checks"][0]["negative_speech_evidence"] is None
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_canonical_text_alignment_failure_preserves_raw_and_only_realigns(tmp_path: Path) -> None:
    class Adapter(RecordingAdapter):
        async def align_text(self, audio: AudioInput, *, text: str, language: str = "auto", timestamps: str = "auto") -> list[TimestampItem]:
            self.aligned.append(text)
            if len(self.aligned) == 1:
                raise AsrError(422, "alignment_failed", "canonical aligner injected failure")
            return [TimestampItem("Qwen3-ASR", 0, 2, "word")]
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(3, 100))
    adapter = Adapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    snapshot: dict[str, object] = {"version": 1, "entries": [{"canonical": "Qwen3-ASR", "aliases": ["千问 ASR"], "auto_normalize": True}]}
    try:
        with pytest.raises(AsrError) as caught:
            await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(glossary_snapshot=snapshot), checkpoint_dir=tmp_path / "checkpoints")
        partial = caught.value.details["partial_result"]
        assert partial["raw_text"] == "千问 ASR。" and partial["text"] == "Qwen3-ASR。"
        assert partial["timestamps"] == [] and partial["retryability"] == "retry"
        for _ in range(2):
            result = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(glossary_snapshot=snapshot), checkpoint_dir=tmp_path / "checkpoints")
            assert result["text"] == "Qwen3-ASR。"
        assert adapter.calls == [4096] and len(adapter.aligned) == 2
    finally:
        await manager.shutdown()


@pytest.mark.parametrize(("completed", "total"), [(36, 73), (19, 119)])
@pytest.mark.asyncio
async def test_head_v14_legacy_prefix_is_immutable_and_only_new_language_action_is_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, completed: int, total: int) -> None:
    from asr_server.adapters.qwen import ALIGNER_REVISION
    from asr_server.audio.assets import NORMALIZATION_VERSION, NormalizedAudioAsset
    from asr_server.quality import PROFILE_VERSION
    from asr_server.quality.checkpoints import fingerprint
    from asr_server.quality.speech_evidence import detector_identity
    windows = [QualityChunk(i, i * 10, (i + 1) * 10, i * 10, (i + 1) * 10) for i in range(total)]
    monkeypatch.setattr("asr_server.quality.pipeline.plan_quality_chunks", lambda *_a, **_kw: (windows, total * 10))
    class Adapter(RecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.windows: list[tuple[float, float | None, str]] = []
        async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
            assert isinstance(audio, AudioPath)
            self.windows.append((audio.start, audio.end, language))
            assert audio.start >= completed * 10, "confirmed old prefix must never be inferred again"
            if audio.start == completed * 10:
                assert language == "zh", "legacy empty auto call and both split actions have already happened"
            return result_for(audio, text=f"新块{int(audio.start // 10)}。", language="Chinese")
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(total * 10, 100))
    asset = NormalizedAudioAsset.from_path(source)
    adapter = Adapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    parameters = request()
    definition = manager.runtime_for(parameters.model).definition
    frozen_request = asdict(parameters)
    frozen_request.pop("idempotency_key", None)
    # Exact HEAD quality_v1.4 identity and {result,reviews,extra_seconds}
    # checkpoint shape, without new policy metadata.
    identity = fingerprint({"audio": file_digest(source), "profile": PROFILE_VERSION,
        "execution_identity": definition.identity(), "normalization_version": NORMALIZATION_VERSION,
        "model": parameters.model, "revision": definition.revision, "aligner_revision": ALIGNER_REVISION,
        "request": frozen_request, "chunks": [asdict(window) for window in windows], "speech_detector": detector_identity()})
    directory = tmp_path / "checkpoints"
    atomic_json(directory / "identity.json", {"fingerprint": identity})
    hashes = {}
    for i in range(completed):
        old = result_for(AudioPath(source, i * 10, (i + 1) * 10), text=f"旧块{i}。", language="Chinese")
        atomic_json(directory / f"chunk-{i:05d}.json", {"result": asdict(old), "reviews": [], "extra_seconds": 0})
        hashes[i] = file_digest(directory / f"chunk-{i:05d}.json")
    atomic_json(directory / "review-budget.json", {"extra_seconds": 20, "attempts": {f"chunk:{completed}": 2}})
    atomic_json(directory / "recovery-import.json", {"base_fingerprint": identity, "source_error_digest": "frozen-old-error",
        "failed_chunks": [{"index": completed, "start": completed * 10, "end": (completed + 1) * 10, "cause": "empty_transcription"}]})
    try:
        for run in range(2):
            counts: list[int] = []
            attempted_counts: list[int] = []
            current_chunks: list[int] = []
            before_indices: list[int] = []
            reconciled_indices: list[int] = []
            async def stage(phase: str, progress: dict[str, object]) -> None:
                if "completed_chunks" in progress:
                    counts.append(int(str(progress["completed_chunks"])))
                if "attempted_chunks" in progress:
                    attempted_counts.append(int(str(progress["attempted_chunks"])))
                if "current_chunk" in progress:
                    current_chunks.append(int(str(progress["current_chunk"])))
            async def before(index: int, count: int) -> None:
                before_indices.append(index)
            async def after(index: int, count: int, chunk_result: TranscriptionResult) -> None:
                reconciled_indices.append(index)
            result = await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=parameters, checkpoint_dir=directory, normalized_asset=asset, stage_callback=stage, before_chunk=before, after_chunk=after)
            assert counts[0] == (completed if run == 0 else total)
            assert counts == sorted(counts), "durable resume progress must never count back from zero"
            assert attempted_counts == sorted(attempted_counts)
            assert all(index >= completed for index in before_indices)
            assert all(index > completed for index in current_chunks)
            assert reconciled_indices == list(range(total)), "cached files still repair missing DB publication"
            if run == 1:
                assert before_indices == [] and current_chunks == []
            assert result["completeness"]["complete"] is True  # type: ignore[index]
            assert result["completeness"]["resolved_chunks"] == total  # type: ignore[index]
            assert result["quality"]["review_audio_seconds"] == 30  # type: ignore[index]
            assert result["quality"]["fingerprint"] == identity  # type: ignore[index]
        assert len(adapter.windows) == total - completed
        assert adapter.windows[0] == (completed * 10, (completed + 1) * 10, "zh")
        assert all(file_digest(directory / f"chunk-{i:05d}.json") == digest for i, digest in hashes.items())
        recovered = Checkpoints(directory, identity)
        assert recovered.action_count(completed) == 3
        assert recovered.review_attempts[f"chunk:{completed}"] == 2
        assert len(list(directory.glob("chunk-*.json"))) == completed
    finally:
        await manager.shutdown()


def test_legacy_inferred_codes_and_mixed_labels_cannot_authorize_language_recovery() -> None:
    chunks = [QualityChunk(i, i * 10, (i + 1) * 10, i * 10, (i + 1) * 10) for i in range(3)]
    for label in ("zh", "en", "Chinese/English", "Chinese, English"):
        results = [TranscriptionResult("text", 10, label, []) for _ in chunks]
        assert language_recovery_evidence(chunks, results, {0, 1, 2}, legacy_indices={0, 1, 2}) is None
    results = [TranscriptionResult("旧块", 10, "Chinese", []) for _ in chunks]
    assert language_recovery_evidence(chunks, results, {0, 1, 2}, legacy_indices={0, 1, 2}) is not None


@pytest.mark.parametrize("mutation", ["missing", "bad_json", "payload_changed"])
def test_imported_legacy_checkpoint_damage_is_a_task_local_explicit_error(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "chunk-00000.json"
    atomic_json(path, {"result": asdict(TranscriptionResult("文字", 60, "Chinese", [])), "reviews": [], "extra_seconds": 0})
    checkpoint = Checkpoints(tmp_path, "base")
    checkpoint.initialize_recovery({"chunks": []})
    if mutation == "missing":
        path.unlink()
    elif mutation == "bad_json":
        path.write_text("{")
    else:
        path.write_text(path.read_text().replace("文字", "修改"))
    restored = Checkpoints(tmp_path, "base")
    with pytest.raises(AsrError) as caught:
        restored.initialize_recovery({"chunks": []})
    assert caught.value.code == "checkpoint_corrupt"
    assert caught.value.details["artifact"] == path.name


@pytest.mark.asyncio
async def test_interrupted_review_without_replay_budget_keeps_sibling_and_later_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_chunks(monkeypatch, 4)
    class Adapter(RecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.windows: list[tuple[float, float | None]] = []
        async def transcribe(self, audio: AudioInput, **kwargs: object) -> TranscriptionResult:
            assert isinstance(audio, AudioPath)
            self.windows.append((audio.start, audio.end))
            if (audio.start, audio.end) == (0, 60):
                raise AsrError(422, "generation_truncated", "initial call truncated")
            if (audio.start, audio.end) == (0, 30):
                raise AsrError(503, "worker_exited", "interrupted after durable quality debit")
            return result_for(audio)
    source = tmp_path / "source.wav"
    source.write_bytes(wav_bytes(240, 100))
    adapter = Adapter()
    manager = ModelLifecycleManager(default_models(enable_qwen_alignment=True), lambda _: adapter, idle_unload_seconds=0)
    try:
        with pytest.raises(AsrError) as caught:
            await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints")
        assert caught.value.code == "worker_exited"
        with pytest.raises(AsrError) as caught:
            await run_transcription_path(source, workspace=tmp_path, manager=manager, settings=Settings(), request=request(), checkpoint_dir=tmp_path / "checkpoints")
        partial = caught.value.details["partial_result"]
        assert partial["completeness"]["resolved_chunks"] == 3
        assert partial["gaps"][0]["reason"] == "review_budget_exhausted"
        assert partial["quality"]["review_audio_seconds"] == 60
        assert adapter.windows == [(0, 60), (0, 30), (30, 60), (60, 120), (120, 180), (180, 240)]
    finally:
        await manager.shutdown()


def test_durable_quality_events_include_recovery_parameters_without_private_content(tmp_path: Path) -> None:
    from asr_server.observability import ExecutionContext, bind_context, configure_logging
    runtime = configure_logging(log_dir=tmp_path / "logs", level="INFO")
    checkpoint = Checkpoints(tmp_path / "checkpoints", "base")
    try:
        with bind_context(ExecutionContext.new(job_id="job_recovery_log")):
            checkpoint.initialize_recovery({"chunks": []})
            checkpoint.reserve_action(3, "language_fallback", 10, 30,
                {"requested_language": "auto", "effective_language": "zh", "source_chunk_indices": [0, 1, 2],
                 "model_language_labels": ["Chinese", "Chinese", "Chinese"], "context": "private-prompt-secret"})
            checkpoint.begin_call("chunk:3:language_fallback:part:0",
                {"requested_language": "auto", "effective_language": "zh", "start": 30, "end": 40}, budget=30)
            checkpoint.finish_call("chunk:3:language_fallback:part:0", TranscriptionResult("private-transcript-secret", 10, "Chinese", []))
    finally:
        runtime.close()
    serialized = (tmp_path / "logs" / "events.jsonl").read_text()
    assert "private-prompt-secret" not in serialized and "private-transcript-secret" not in serialized
    events = [json.loads(line) for line in serialized.splitlines()]
    reservation = next(item for item in events if item["event"] == "quality.action_reserved")
    assert reservation["source_chunk_indices"] == [0, 1, 2]
    assert reservation["model_language_labels"] == ["Chinese", "Chinese", "Chinese"]
    assert reservation["budget_before_seconds"] == 30 and reservation["budget_after_seconds"] == 20
    assert reservation["requested_language"] == "auto" and reservation["effective_language"] == "zh"
    started = next(item for item in events if item["event"] == "quality.call_started")
    saved = next(item for item in events if item["event"] == "quality.call_persisted")
    assert started["attempt_id"] == saved["attempt_id"]
    assert saved["call_key"] == "chunk:3:language_fallback:part:0" and saved["status"] == "transcribed"
