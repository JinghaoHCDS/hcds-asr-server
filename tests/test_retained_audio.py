from __future__ import annotations

import asyncio
import hashlib
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from asr_server.audio import retained
from asr_server.errors import AsrError
from asr_server.observability import ExecutionContext
from tests.helpers import make_wav


def encode(source: Path, target: Path, *options: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(source),
                    *options, str(target)], check=True, capture_output=True, timeout=10)


async def prepare(source: Path, target: Path, **options: Any) -> dict[str, Any]:
    return await retained.prepare_retained_audio(source, target, max_duration_seconds=options.pop("max_duration_seconds", 21600),
                                                max_output_bytes=options.pop("max_output_bytes", 8 * 1024 * 1024),
                                                min_free_disk_bytes=options.pop("min_free_disk_bytes", 0), **options)


@pytest.mark.parametrize("rate,channels,expected_rate,expected_channels", [
    (44100, 1, 44100, 1), (48000, 2, 48000, 2), (16000, 1, 48000, 1), (48000, 6, 48000, 2),
])
async def test_retained_mp3_preserves_required_format_and_complete_effective_timeline(
    tmp_path: Path, rate: int, channels: int, expected_rate: int, expected_channels: int,
) -> None:
    source, target = tmp_path / "opaque-upload", tmp_path / "audio.mp3"
    source.write_bytes(make_wav(.25, sample_rate=rate, channels=channels))
    metadata = await prepare(source, target)
    assert metadata["sample_rate"] == expected_rate
    assert metadata["channels"] == expected_channels
    assert metadata["codec"] == "mp3" and metadata["bitrate_bps"] == 192000
    assert metadata["duration_seconds"] == pytest.approx(.25)
    assert metadata["frame_count"] == int(.25 * expected_rate)
    assert metadata["size_bytes"] == target.stat().st_size
    assert metadata["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert metadata["reused"] is False
    assert (metadata["channel_processing"] == "preserved") == (channels <= 2)
    retained.validate_retained_audio(target, metadata)
    assert source.exists() and not list(tmp_path.glob("*.partial*"))


@pytest.mark.parametrize("container", ["aac", "mp4"])
async def test_aac_and_video_choose_first_audio_track_and_record_it(tmp_path: Path, container: str) -> None:
    wav = tmp_path / "source.wav"
    wav.write_bytes(make_wav(.35, sample_rate=44100, channels=2))
    source = tmp_path / f"source.{container}"
    if container == "mp4":
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=32x32:d=1",
                        "-itsoffset", "0.2", "-i", str(wav), "-map", "0:v", "-map", "1:a", "-map", "1:a",
                        "-c:v", "mpeg4", "-c:a", "aac", str(source)], check=True, capture_output=True, timeout=10)
    else:
        encode(wav, source, "-c:a", "aac")
    metadata = await prepare(source, tmp_path / "audio.mp3")
    assert metadata["source_stream_index"] == (1 if container == "mp4" else 0)
    assert metadata["source_audio_streams"] == (2 if container == "mp4" else 1)
    if container == "mp4":
        assert metadata["timeline"]["source_start_seconds"] > .1
    assert metadata["duration_seconds"] > 0


async def test_compatible_mp3_reuses_bytes_and_vbr_or_lower_bitrate_transcodes(tmp_path: Path) -> None:
    wav = tmp_path / "source.wav"
    wav.write_bytes(make_wav(.3, sample_rate=48000, channels=2))
    first = await prepare(wav, tmp_path / "first.mp3")
    copied = await prepare(tmp_path / "first.mp3", tmp_path / "copied.mp3")
    assert copied["reused"] is True
    assert copied["sha256"] == first["sha256"]
    assert (tmp_path / "first.mp3").read_bytes() == (tmp_path / "copied.mp3").read_bytes()
    for name, options in [("vbr.mp3", ["-q:a", "4"]), ("low.mp3", ["-b:a", "64k"]),
                          ("low-rate.mp3", ["-ar", "16000", "-b:a", "64k"])]:
        encode(wav, tmp_path / name, "-c:a", "libmp3lame", *options)
        metadata = await prepare(tmp_path / name, tmp_path / (name + ".out.mp3"))
        assert metadata["reused"] is False and metadata["sample_rate"] == 48000


async def test_partial_or_whole_frame_tail_truncation_does_not_become_ready(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(make_wav(.5, sample_rate=48000, channels=1))
    await prepare(source, tmp_path / "complete.mp3")
    content = (tmp_path / "complete.mp3").read_bytes()
    # 576 bytes is one complete 192k/48k MPEG frame. Xing's count detects
    # loss of a whole final frame even when a decoder would silently accept it.
    for missing in (1, 576):
        broken = tmp_path / f"broken-{missing}.mp3"
        broken.write_bytes(content[:-missing])
        with pytest.raises(AsrError) as error:
            await prepare(broken, tmp_path / f"result-{missing}.mp3")
        assert error.value.code == "audio_integrity_failed"
        assert broken.exists() and not (tmp_path / f"result-{missing}.mp3").exists()
    assert not list(tmp_path.glob("*.partial*"))


async def test_actual_frames_enforce_duration_despite_incorrect_probe_estimate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, output = tmp_path / "source", tmp_path / "audio.mp3"
    source.write_bytes(make_wav(.4))
    original = retained.probe_retained_source
    async def underestimated(path: Path, **options: Any) -> dict[str, Any]:
        result = await original(path, **options)
        result["duration_seconds"] = .01
        return result
    monkeypatch.setattr(retained, "probe_retained_source", underestimated)
    with pytest.raises(AsrError) as error:
        await prepare(source, output, max_duration_seconds=.2)
    assert error.value.code == "duration_limit_exceeded"
    assert source.exists() and not output.exists()


@pytest.mark.parametrize("case", ["invalid", "truncated-wav", "playlist", "budget", "disk"])
async def test_failed_conversion_keeps_only_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    source, output = tmp_path / "source", tmp_path / "audio.mp3"
    content = make_wav(.3)
    if case == "invalid":
        content = b"not media"
    elif case == "truncated-wav":
        content = content[:-200]
    elif case == "playlist":
        content = b"#EXTM3U\n#EXTINF:1,\nhttp://127.0.0.1:9/audio\n#EXT-X-ENDLIST\n"
    source.write_bytes(content)
    options: dict[str, Any] = {}
    if case == "budget":
        options["max_output_bytes"] = 10
    if case == "disk":
        from collections import namedtuple
        usage = namedtuple("usage", "total used free")
        monkeypatch.setattr("asr_server.audio.retained.shutil.disk_usage", lambda _path: usage(100, 100, 0))
    with pytest.raises(AsrError):
        await prepare(source, output, **options)
    assert source.read_bytes() == content
    assert not output.exists() and not list(tmp_path.glob("*.partial*"))


async def test_missing_encoder_is_explicit_and_preserves_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    source.write_bytes(make_wav(.1))
    original = retained._command
    async def missing(*args: str, **options: Any) -> tuple[bytes, bytes]:
        if "-encoders" in args:
            return b"no mp3 encoder", b""
        return await original(*args, **options)
    monkeypatch.setattr(retained, "_command", missing)
    with pytest.raises(AsrError) as error:
        await prepare(source, tmp_path / "audio.mp3")
    assert error.value.code == "mp3_encoder_unavailable" and source.exists()


async def test_cancellation_and_timeout_reap_media_and_remove_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    source.write_bytes(make_wav(.5))
    context = ExecutionContext(trace_id="retained-cancel", cancel_event=threading.Event())
    async def cancel_on_reservation(_size: int) -> None:
        context.cancel_event.set()
    with pytest.raises(asyncio.CancelledError):
        await prepare(source, tmp_path / "cancel.mp3", context=context, ensure_output_capacity=cancel_on_reservation)
    assert source.exists() and not list(tmp_path.glob("*.partial*"))
    monkeypatch.setattr(retained, "OPERATION_TIMEOUT_SECONDS", .0001)
    with pytest.raises(AsrError) as error:
        await prepare(source, tmp_path / "timeout.mp3")
    assert error.value.code == "audio_preprocess_timeout"
    assert not (tmp_path / "timeout.mp3").exists() and not list(tmp_path.glob("*.partial*"))


async def test_capacity_callbacks_admit_before_encoding_and_settle_actual_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(make_wav(.3))
    admitted = 0
    observed: list[int] = []
    async def capacity(size: int) -> None:
        nonlocal admitted
        admitted = max(size, admitted)
    async def materialized(size: int) -> None:
        assert size <= admitted
        observed.append(size)
    metadata = await prepare(source, tmp_path / "audio.mp3", ensure_output_capacity=capacity, output_written=materialized)
    assert observed[-1] == metadata["size_bytes"] <= admitted
    copied = await prepare(tmp_path / "audio.mp3", tmp_path / "copy.mp3", ensure_output_capacity=capacity, output_written=materialized)
    assert copied["reused"] is True and observed[-1] == copied["size_bytes"]


async def test_recovery_detects_mutation_and_unknown_audio_version(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "audio.mp3"
    source.write_bytes(make_wav(.1))
    metadata = await prepare(source, output)
    retained.validate_retained_audio(output, metadata)
    with pytest.raises(AsrError) as version:
        retained.validate_retained_audio(output, {**metadata, "version": "future"})
    assert version.value.code == "retained_audio_version_mismatch"
    with output.open("r+b") as stream:
        stream.seek(-8, 2)
        stream.write(b"modified")
    with pytest.raises(AsrError) as identity:
        retained.validate_retained_audio(output, metadata)
    assert identity.value.code == "retained_audio_invalid"


async def test_begin_middle_end_markers_remain_within_100ms_after_mp3_decode(tmp_path: Path) -> None:
    """Engineering timestamp check, separate from speech/model accuracy."""
    from array import array
    import math
    import wave
    from asr_server.audio.preprocess import normalize_audio_path_to_wav_async

    rate, duration = 48000, 1.8
    markers = [.15, .85, 1.55]
    samples = array("h", [0]) * int(duration * rate)
    for marker in markers:
        for offset in range(int(.05 * rate)):
            time = offset / rate
            envelope = math.sin(math.pi * time / .05) ** 2
            samples[int(marker * rate) + offset] = int(20000 * envelope * math.sin(2 * math.pi * 1000 * time))
    source = tmp_path / "markers.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())
    metadata = await prepare(source, tmp_path / "audio.mp3")
    await normalize_audio_path_to_wav_async(tmp_path / "audio.mp3", tmp_path / "decoded.wav")
    with wave.open(str(tmp_path / "decoded.wav"), "rb") as decoded:
        restored = array("h", decoded.readframes(decoded.getnframes()))
    for marker in markers:
        left, right = int((marker - .03) * 16000), int((marker + .08) * 16000)
        peak = max(range(left, right), key=lambda index: abs(restored[index])) / 16000
        assert abs(peak - (marker + .025)) < .1
    assert metadata["duration_seconds"] == pytest.approx(duration)
