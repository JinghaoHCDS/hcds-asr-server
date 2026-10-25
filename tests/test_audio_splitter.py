from __future__ import annotations

import io
from functools import partial
from tests.helpers import make_wav as shared_wav
import wave
import struct
import threading
import subprocess
from array import array
from pathlib import Path

import pytest

from asr_server.audio import splitter as splitter_module
from asr_server.audio.metadata import AudioMetadata
from asr_server.audio.preprocess import normalize_audio_to_wav
from asr_server.audio.splitter import SplitCancelled, split_audio, split_audio_path
from asr_server.errors import AsrError


make_wav = partial(shared_wav, amplitude=0, sample_rate=8000, channels=1)


def make_segmented_wav(segments: list[tuple[float, int]], sample_rate: int = 16_000) -> bytes:
    samples = array("h")
    for duration_seconds, amplitude in segments:
        samples.extend([amplitude] * int(duration_seconds * sample_rate))
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return output.getvalue()


def wav_duration(audio: bytes) -> float:
    with wave.open(io.BytesIO(audio), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def test_short_audio_does_not_split_in_auto_mode() -> None:
    result = split_audio(
        b"a" * 160,
        split_strategy="auto",
        max_chunk_seconds=1.0,
        overlap_seconds=0.1,
    )

    assert result.strategy == "none"
    assert len(result.chunks) == 1
    assert result.chunks[0].start == 0.0
    assert result.chunks[0].end == pytest.approx(0.01)


def test_fixed_strategy_splits_raw_audio_with_overlap() -> None:
    result = split_audio(
        b"a" * 1760,
        split_strategy="fixed",
        max_chunk_seconds=0.04,
        overlap_seconds=0.01,
    )

    assert result.strategy == "fixed"
    assert len(result.chunks) == 4
    assert result.chunks[0].start == 0.0
    assert result.chunks[1].start == pytest.approx(0.03)
    assert result.chunks[-1].end == pytest.approx(0.11)
    assert all(current.start < current.end for current in result.chunks)
    assert all(
        previous.start <= current.start <= previous.end
        for previous, current in zip(result.chunks, result.chunks[1:])
    )


def test_fixed_strategy_keeps_wav_chunks_decodable() -> None:
    result = split_audio(
        make_wav(0.11),
        split_strategy="fixed",
        max_chunk_seconds=0.04,
        overlap_seconds=0.01,
    )

    assert len(result.chunks) == 4
    assert wav_duration(result.chunks[0].audio) == pytest.approx(0.04)
    assert wav_duration(result.chunks[-1].audio) == pytest.approx(0.02)


def test_energy_strategy_splits_wav_on_silence() -> None:
    audio = make_segmented_wav(
        [
            (0.2, 0),
            (0.4, 10_000),
            (0.5, 0),
            (0.4, 10_000),
            (0.2, 0),
        ]
    )

    result = split_audio(
        audio,
        split_strategy="energy",
        max_chunk_seconds=0.8,
        overlap_seconds=0.03,
    )

    assert result.strategy == "energy"
    assert result.vad_backend == "energy"
    assert len(result.chunks) == 2
    assert result.chunks[0].start < 0.22
    assert result.chunks[0].end > 0.55
    assert result.chunks[1].start > 1.0
    assert result.chunks[1].end < 1.6


def test_vad_alias_falls_back_to_energy_when_silero_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = make_segmented_wav(
        [
            (0.2, 0),
            (0.4, 10_000),
            (0.5, 0),
            (0.4, 10_000),
        ]
    )

    def unavailable(*args: object, **kwargs: object) -> list[tuple[float, float]]:
        raise splitter_module._SileroUnavailable("test unavailable")

    monkeypatch.setattr(splitter_module, "_silero_intervals", unavailable)

    result = split_audio(
        audio,
        split_strategy="vad",
        max_chunk_seconds=0.8,
        overlap_seconds=0.03,
    )

    assert result.strategy == "energy"
    assert result.requested_strategy == "vad"
    assert result.vad_backend == "energy"
    assert any(warning.startswith("silero_vad_unavailable") for warning in result.warnings)
    assert len(result.chunks) == 2


def test_auto_prefers_silero_for_long_audio_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = make_segmented_wav(
        [
            (0.2, 0),
            (0.4, 10_000),
            (0.5, 0),
            (0.4, 10_000),
        ]
    )

    def silero_intervals(*args: object, **kwargs: object) -> list[tuple[float, float]]:
        return [(0.17, 0.63), (1.07, 1.5)]

    monkeypatch.setattr(splitter_module, "_silero_intervals", silero_intervals)

    result = split_audio(
        audio,
        split_strategy="auto",
        max_chunk_seconds=0.8,
        overlap_seconds=0.03,
    )

    assert result.strategy == "silero"
    assert result.requested_strategy == "auto"
    assert result.vad_backend == "silero"
    assert len(result.chunks) == 2


def test_auto_falls_back_to_energy_when_silero_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = make_segmented_wav(
        [
            (0.2, 0),
            (0.4, 10_000),
            (0.5, 0),
            (0.4, 10_000),
        ]
    )

    def unavailable(*args: object, **kwargs: object) -> list[tuple[float, float]]:
        raise splitter_module._SileroUnavailable("test unavailable")

    monkeypatch.setattr(splitter_module, "_silero_intervals", unavailable)

    result = split_audio(
        audio,
        split_strategy="auto",
        max_chunk_seconds=0.8,
        overlap_seconds=0.03,
    )

    assert result.strategy == "energy"
    assert result.requested_strategy == "auto"
    assert result.vad_backend == "energy"
    assert len(result.chunks) == 2
    assert any(warning.startswith("silero_vad_unavailable") for warning in result.warnings)


def test_overlap_must_be_smaller_than_chunk_length() -> None:
    with pytest.raises(AsrError) as exc_info:
        split_audio(
            b"a" * 1600,
            split_strategy="fixed",
            max_chunk_seconds=1.0,
            overlap_seconds=1.0,
        )

    assert exc_info.value.code == "bad_request"


def test_duration_limit_is_enforced_before_sync_transcription(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        splitter_module,
        "inspect_audio",
        lambda audio: AudioMetadata(
            duration_seconds=21_601.0,
            format="raw",
            byte_length=len(audio),
        ),
    )

    with pytest.raises(AsrError) as exc_info:
        split_audio(
            b"a",
            split_strategy="none",
            max_chunk_seconds=None,
            overlap_seconds=None,
        )

    assert exc_info.value.code == "duration_limit_exceeded"
    assert exc_info.value.details["max_audio_seconds_per_file"] == 21_600.0


def test_generated_mp3_is_preprocessed_and_energy_split(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> list[tuple[float, float]]:
        raise splitter_module._SileroUnavailable("test unavailable")

    monkeypatch.setattr(splitter_module, "_silero_intervals", unavailable)

    generated_mp3 = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=16000:duration=210",
            "-af", "volume='if(lt(mod(t,15),10),1,0)':eval=frame",
            "-c:a", "libmp3lame", "-b:a", "32k", "-f", "mp3", "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    normalized = normalize_audio_to_wav(generated_mp3)
    result = split_audio(
        normalized.audio,
        split_strategy="auto",
        max_chunk_seconds=180.0,
        overlap_seconds=2.0,
    )

    assert result.metadata.format == "wav"
    assert result.metadata.sample_rate == 16_000
    assert result.metadata.channels == 1
    assert result.metadata.duration_seconds == pytest.approx(210.0, abs=0.2)
    assert result.strategy == "energy"
    assert result.vad_backend == "energy"
    assert len(result.chunks) > 1
    assert all(chunk.duration <= 180.0 for chunk in result.chunks)
    assert all(chunk.start < chunk.end for chunk in result.chunks)


def test_none_strategy_rejects_audio_above_hard_model_window() -> None:
    audio = b"x" * (16_000 * 301)

    with pytest.raises(AsrError) as exc_info:
        split_audio(audio, split_strategy="none", max_chunk_seconds=None, overlap_seconds=None)

    assert exc_info.value.code == "duration_limit_exceeded"


def test_split_rejects_adversarial_chunk_count_before_materialization() -> None:
    audio = b"x" * (16_000 * 60)

    with pytest.raises(AsrError) as exc_info:
        split_audio(audio, split_strategy="fixed", max_chunk_seconds=0.01, overlap_seconds=0.0)

    assert exc_info.value.details["max_chunks_per_file"] == 4096


def test_path_splitter_returns_descriptors_without_audio_bytes(tmp_path: Path) -> None:
    audio_path = tmp_path / "normalized.wav"
    audio_path.write_bytes(
        make_segmented_wav([(0.2, 0), (0.4, 10_000), (0.5, 0), (0.4, 10_000)])
    )

    result = split_audio_path(
        audio_path,
        split_strategy="auto",
        max_chunk_seconds=0.8,
        overlap_seconds=0.03,
    )

    assert result.strategy == "energy"
    assert result.chunks
    assert all(chunk.audio_path == audio_path for chunk in result.chunks)
    assert all(not hasattr(chunk, "audio") for chunk in result.chunks)


def test_six_hour_sparse_wav_creates_only_bounded_descriptors(tmp_path: Path) -> None:
    sample_rate = 16_000
    data_size = 21_600 * sample_rate * 2
    audio_path = tmp_path / "six-hours.wav"
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    with audio_path.open("wb") as output:
        output.write(header)
        output.truncate(44 + data_size)

    result = split_audio_path(
        audio_path,
        split_strategy="fixed",
        max_chunk_seconds=120,
        overlap_seconds=2,
    )

    assert result.metadata.duration_seconds == 21_600
    assert len(result.chunks) == 184
    assert result.chunks[-1].end == 21_600


def test_path_vad_honors_cancellation_event(tmp_path: Path) -> None:
    audio_path = tmp_path / "normalized.wav"
    audio_path.write_bytes(make_segmented_wav([(1.0, 10_000)]))
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(SplitCancelled):
        split_audio_path(
            audio_path,
            split_strategy="energy",
            max_chunk_seconds=0.5,
            overlap_seconds=0.01,
            cancel_event=cancelled,
        )


@pytest.mark.parametrize("chunk,overlap", [
    (0.0, 0.0), (1.0, float("nan")), (float("nan"), 0.0),
    (float("inf"), 0.0), (1.0, float("inf")), (1.0, 1.0 - 1e-12),
    (1e-9, 0.0),
])
def test_window_parameters_reject_nonfinite_zero_and_nonadvancing_sample_options(
    tmp_path: Path, chunk: float, overlap: float,
) -> None:
    path = tmp_path / "input.wav"
    path.write_bytes(make_wav(0.1, sample_rate=16_000))
    for source in (path, path.read_bytes()):
        with pytest.raises(AsrError) as error:
            if isinstance(source, Path):
                split_audio_path(source, split_strategy="energy", max_chunk_seconds=chunk, overlap_seconds=overlap)
            else:
                split_audio(source, split_strategy="energy", max_chunk_seconds=chunk, overlap_seconds=overlap)
        assert error.value.code == "bad_request"


def test_vad_windows_share_one_task_limit_and_observe_mid_generation_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(splitter_module, "MAX_CHUNKS_PER_FILE", 3)
    with pytest.raises(AsrError) as error:
        splitter_module._pack_vad_intervals([(0, 2), (3, 5)], 1, 0)
    assert error.value.code == "duration_limit_exceeded"
    assert error.value.details["chunk_count"] == 4

    cancelled = threading.Event()
    append = splitter_module._append_window
    def cancel_after_one(*args: object, **kwargs: object) -> None:
        append(*args, **kwargs)  # type: ignore[arg-type]
        cancelled.set()
    monkeypatch.setattr(splitter_module, "_append_window", cancel_after_one)
    with pytest.raises(SplitCancelled):
        splitter_module._interval_windows(0, 2, 1, 0, cancel_event=cancelled)
