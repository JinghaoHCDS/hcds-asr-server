from __future__ import annotations

import io
from functools import partial
from tests.helpers import make_wav as shared_wav
import wave
import asyncio
from pathlib import Path

import pytest

from asr_server.audio.preprocess import normalize_audio_path_to_wav_async, normalize_audio_to_wav
from asr_server.errors import AsrError


make_wav = partial(shared_wav, amplitude=0, sample_rate=8000, channels=2)


def test_normalize_audio_to_wav_outputs_asr_friendly_wav() -> None:
    normalized = normalize_audio_to_wav(make_wav(0.1))

    with wave.open(io.BytesIO(normalized.audio), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16_000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 1600
    assert normalized.decode_ms >= 0


async def test_async_ffmpeg_timeout_terminates_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class HangingStream:
        async def read(self, _size: int) -> bytes:
            await asyncio.sleep(60)
            return b""

    class HangingProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminated = False
            self.stdout = HangingStream()
            self.stderr = HangingStream()

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    process = HangingProcess()

    async def create_process(*args: object, **kwargs: object) -> HangingProcess:
        del args, kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    with pytest.raises(AsrError) as exc_info:
        await normalize_audio_path_to_wav_async(
            tmp_path / "upload",
            tmp_path / "normalized.wav",
            timeout_seconds=0.01,
        )

    assert exc_info.value.code == "audio_preprocess_timeout"
    assert process.terminated is True


async def test_decode_enforces_actual_pcm_budget_and_does_not_deliver_truncated_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from asr_server.audio.preprocess import normalize_audio_path_to_wav

    source, output = tmp_path / "upload", tmp_path / "bounded.wav"
    # Probe estimates are intentionally absent: only actual decoded bytes decide.
    source.write_bytes(make_wav(33.0, sample_rate=16_000, channels=1))
    observed_sizes: list[int] = []
    unlink = Path.unlink
    def capture_partial(path: Path, missing_ok: bool = False) -> None:
        if path == output and path.exists():
            observed_sizes.append(path.stat().st_size)
        unlink(path, missing_ok=missing_ok)
    monkeypatch.setattr(Path, "unlink", capture_partial)
    with pytest.raises(AsrError) as error:
        await normalize_audio_path_to_wav_async(source, output, max_output_bytes=1024 * 1024)
    assert error.value.code == "storage_unavailable"
    assert observed_sizes and max(observed_sizes) <= 1024 * 1024
    assert not output.exists()
    # The retained synchronous entry uses exactly the same hard limit.
    with pytest.raises(AsrError) as error:
        await asyncio.to_thread(normalize_audio_path_to_wav, source, output, max_output_bytes=1024 * 1024)
    assert error.value.code == "storage_unavailable"

    source.write_bytes(make_wav(0.1, sample_rate=16_000, channels=1))
    result = await normalize_audio_path_to_wav_async(source, output, max_output_bytes=44 + 3200)
    assert result.path.stat().st_size == 44 + 3200
    assert result.asset is not None and result.asset.frame_count == 1600


async def test_decoder_cancellation_reaps_process_and_removes_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import threading

    entered = asyncio.Event()
    cancelled = threading.Event()
    class HangingStream:
        async def read(self, _size: int) -> bytes:
            entered.set()
            await asyncio.Event().wait()
            return b""
    class Process:
        returncode: int | None = None
        stdout = HangingStream()
        stderr = HangingStream()
        def terminate(self) -> None:
            self.returncode = -15
        def kill(self) -> None:
            self.returncode = -9
        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode
    process = Process()
    async def create_process(*args: object, **kwargs: object) -> Process:
        return process
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    output = tmp_path / "partial.wav"
    task = asyncio.create_task(normalize_audio_path_to_wav_async(tmp_path / "upload", output,
                                                                cancel_event=cancelled))
    await asyncio.wait_for(entered.wait(), timeout=1)
    cancelled.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    assert process.returncode == -15
    assert not output.exists()


async def test_normalized_asset_reuse_verifies_pcm_identity_without_owning_source_deletion(tmp_path: Path) -> None:
    from asr_server.audio.assets import NormalizedAudioAsset
    from asr_server.audio.preprocess import normalize_audio_path_to_wav_async

    source, output = tmp_path / "source", tmp_path / "audio.wav"
    source.write_bytes(make_wav(0.2))
    normalized = await normalize_audio_path_to_wav_async(source, output, owner="job-example")
    assert normalized.asset is not None
    asset = NormalizedAudioAsset.from_metadata(output, normalized.asset.to_metadata())
    assert asset.audio_view().path == output
    assert asset.duration == pytest.approx(0.2)
    asset.validate()
    # Changing a sample invalidates reuse even when size, rate and frames match.
    with output.open("r+b") as audio:
        audio.seek(44)
        audio.write(b"\x01\x00")
    with pytest.raises(AsrError) as error:
        asset.validate()
    assert error.value.code == "normalized_asset_invalid"
    assert source.exists() and output.exists()


@pytest.mark.parametrize("suffix,codec", [
    ("wav", "pcm_s16le"), ("mp3", "libmp3lame"), ("flac", "flac"),
    ("aac", "aac"), ("ogg", "libopus"), ("m4a", "aac"), ("mp4", "aac"),
    ("mov", "aac"), ("webm", "libopus"), ("mkv", "flac"),
])
async def test_media_policy_accepts_representative_regular_containers(tmp_path: Path, suffix: str, codec: str) -> None:
    import subprocess
    from asr_server.audio.preprocess import probe_audio_duration_path_async

    source, output = tmp_path / f"source.{suffix}", tmp_path / "normalized.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=0.1",
                    "-c:a", codec, str(source)], check=True, timeout=10, capture_output=True)
    assert await probe_audio_duration_path_async(source) is not None
    normalized = await normalize_audio_path_to_wav_async(source, output)
    assert normalized.asset is not None and normalized.asset.frame_count > 0


@pytest.mark.parametrize("kind", ["concat", "hls-local", "hls-network"])
async def test_media_policy_rejects_secondary_resource_playlists_in_isolated_scratch(tmp_path: Path, kind: str) -> None:
    from asr_server.audio.preprocess import probe_audio_duration_path_async

    dummy = tmp_path / "dummy.wav"
    dummy.write_bytes(make_wav(0.1))
    source, output = tmp_path / "upload", tmp_path / "normalized.wav"
    if kind == "concat":
        payload = f"ffconcat version 1.0\nfile '{dummy}'\n"
    elif kind == "hls-local":
        payload = f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:0.1,\n{dummy}\n#EXT-X-ENDLIST\n"
    else:
        # No listener or internet access is needed: the demuxer is rejected.
        payload = "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:0.1,\nhttp://127.0.0.1:9/dummy\n#EXT-X-ENDLIST\n"
    source.write_text(payload)
    assert await probe_audio_duration_path_async(source) is None
    with pytest.raises(AsrError) as error:
        await normalize_audio_path_to_wav_async(source, output)
    assert error.value.code == "audio_decode_failed"
    assert not output.exists() and dummy.exists()


async def test_all_async_normalization_instances_share_decoder_slot_until_cancel_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import Any
    from asr_server.audio.preprocess import MediaPreprocessor, NormalizedAudioPath

    entered, draining, allow_cleanup, second_entered = (asyncio.Event() for _ in range(4))
    async def normalize(self: MediaPreprocessor, source: Path, output: Path, **options: Any) -> NormalizedAudioPath:
        if source.name == "first":
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                draining.set()
                await allow_cleanup.wait()
        second_entered.set()
        output.write_bytes(b"test fixture")
        return NormalizedAudioPath(output, 0)
    monkeypatch.setattr(MediaPreprocessor, "_normalize", normalize)
    first = asyncio.create_task(normalize_audio_path_to_wav_async(tmp_path / "first", tmp_path / "first.wav"))
    await entered.wait()
    second = asyncio.create_task(MediaPreprocessor().normalize(tmp_path / "second", tmp_path / "second.wav"))
    first.cancel()
    await draining.wait()
    await asyncio.sleep(0)
    assert not second_entered.is_set()
    allow_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await asyncio.wait_for(second, 1)
    assert second_entered.is_set()
