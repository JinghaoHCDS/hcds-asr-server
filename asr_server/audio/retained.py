"""Verified, retained workbench audio, separate from engine PCM workspaces.

Only the caller commits metadata and removes the original upload.  A successful
return means the complete MP3 has been decoded and atomically installed.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from asr_server.audio.preprocess import (
    INPUT_OPTIONS, PCM_BLOCK_BYTES, STDERR_TAIL_BYTES, _decode_slot,
    _guard_operation, _read_tail,
)
from asr_server.errors import AsrError
from asr_server.observability import ExecutionContext


RETAINED_AUDIO_VERSION = "mp3-cbr192-v1"
BITRATE_BPS = 192_000
MAX_DURATION_SECONDS = 21_600.0
OPERATION_TIMEOUT_SECONDS = 1800.0
PROBE_LIMIT_BYTES = 256 * 1024


def _event(context: ExecutionContext | None) -> threading.Event | None:
    return context.cancel_event if context is not None else None


def _check_cancel(context: ExecutionContext | None) -> None:
    if context is not None and context.cancel_event.is_set():
        raise asyncio.CancelledError


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


async def _command(*arguments: str, context: ExecutionContext | None = None,
                   timeout_seconds: float = 30.0) -> tuple[bytes, bytes]:
    _check_cancel(context)
    try:
        process = await asyncio.create_subprocess_exec(*arguments, stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError as exc:
        raise AsrError(503, "audio_preprocess_unavailable", "ffmpeg and ffprobe are required") from exc

    async def collect() -> tuple[bytes, bytes]:
        stdout = process.stdout
        assert stdout is not None
        async def read_output() -> bytes:
            data = bytearray()
            while block := await stdout.read(PCM_BLOCK_BYTES):
                data.extend(block)
                if len(data) > PROBE_LIMIT_BYTES:
                    raise AsrError(400, "audio_probe_failed", "media metadata exceeds the allowed size")
            return bytes(data)
        output_task = asyncio.create_task(read_output())
        error_task = asyncio.create_task(_read_tail(process.stderr, STDERR_TAIL_BYTES))
        try:
            output, errors = await asyncio.gather(output_task, error_task)
            await process.wait()
            if process.returncode != 0:
                raise AsrError(400, "audio_decode_failed", "media could not be read completely")
            return output, errors
        finally:
            for task in (output_task, error_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(output_task, error_task, return_exceptions=True)

    return await _guard_operation(process, collect(), timeout_seconds=timeout_seconds,
                                  timeout_code="audio_preprocess_timeout", cancel_event=_event(context))


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


async def probe_retained_source(source: Path, *, context: ExecutionContext | None = None) -> dict[str, Any]:
    """Bounded first-audio-track discovery; duration is an estimate, not acceptance."""
    output, _ = await _command(
        "ffprobe", "-v", "error", *INPUT_OPTIONS, "-show_entries",
        "stream=index,codec_type,codec_name,sample_rate,channels,channel_layout,start_time,duration:format=duration",
        "-of", "json", str(source.resolve()), context=context,
    )
    try:
        info = json.loads(output)
        audio = [stream for stream in info["streams"] if stream.get("codec_type") == "audio"]
        first = audio[0]
        rate, channels, index = int(first["sample_rate"]), int(first["channels"]), int(first["index"])
        if rate <= 0 or channels <= 0 or channels > 64 or index < 0:
            raise ValueError
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise AsrError(400, "audio_decode_failed", "media has no supported audio track") from exc
    return {
        "index": index, "audio_streams": len(audio), "sample_rate": rate, "channels": channels,
        "codec": str(first.get("codec_name", "")), "channel_layout": str(first.get("channel_layout", "")),
        "start_time_seconds": _finite(first.get("start_time")) or 0.0,
        "duration_seconds": _finite(first.get("duration")) or _finite(info.get("format", {}).get("duration")),
        "target_sample_rate": rate if rate in {44_100, 48_000} else 48_000,
        "target_channels": min(channels, 2),
    }


def _mix_filter(channels: int) -> str | None:
    if channels <= 2:
        return None
    groups = [list(range(offset, channels, 2)) for offset in (0, 1)]
    return "pan=stereo|" + "|".join(
        f"c{side}=" + "+".join(f"{1 / len(group):.12g}*c{index}" for index in group)
        for side, group in enumerate(groups)
    )


def _audio_options(info: dict[str, Any]) -> list[str]:
    options = ["-map", f"0:{info['index']}", "-vn", "-sn", "-dn"]
    mix = _mix_filter(int(info["channels"]))
    if mix:
        options.extend(["-af", mix])
    return [*options, "-ac", str(info["target_channels"]), "-ar", str(info["target_sample_rate"])]


async def _decoded_frames(source: Path, info: dict[str, Any], *, max_duration: float,
                          context: ExecutionContext | None) -> int:
    """Count actual effective samples, with no PCM file or unbounded stdout buffer."""
    _check_cancel(context)
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
            *INPUT_OPTIONS, "-i", str(source.resolve()), *_audio_options(info),
            "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise AsrError(503, "audio_preprocess_unavailable", "ffmpeg is required") from exc
    frame_bytes = int(info["target_channels"]) * 2
    max_bytes = int(max_duration * int(info["target_sample_rate"])) * frame_bytes

    async def collect() -> int:
        assert process.stdout is not None
        errors_task = asyncio.create_task(_read_tail(process.stderr, STDERR_TAIL_BYTES))
        total = 0
        try:
            while block := await process.stdout.read(PCM_BLOCK_BYTES):
                total += len(block)
                _check_cancel(context)
                if total > max_bytes:
                    raise AsrError(422, "duration_limit_exceeded", "decoded audio exceeds the duration limit")
            errors = await errors_task
            await process.wait()
            if process.returncode != 0 or errors or not total or total % frame_bytes:
                raise AsrError(400, "audio_decode_failed", "media could not be decoded completely")
            return total // frame_bytes
        finally:
            if not errors_task.done():
                errors_task.cancel()
            await asyncio.gather(errors_task, return_exceptions=True)

    return await _guard_operation(process, collect(), timeout_seconds=OPERATION_TIMEOUT_SECONDS,
                                  timeout_code="audio_preprocess_timeout", cancel_event=_event(context))


def _mp3_frames(path: Path, *, rate: int, channels: int, require_spec: bool = True) -> dict[str, int]:
    """Walk every layer III frame; never infer CBR from average bit rate.

    Validates frame lengths, optional Xing frame totals, and the entire tail.
    Nonstandard but structurally valid MP3 is transcoded instead of reused.
    """
    with path.open("rb") as stream:
        size = path.stat().st_size
        first = stream.read(10)
        position = 0
        if first[:3] == b"ID3":
            if len(first) != 10 or first[3] not in {2, 3, 4} or any(value & 128 for value in first[6:10]):
                raise ValueError("invalid ID3 tag")
            position = 10 + sum(value << shift for value, shift in zip(first[6:10], (21, 14, 7, 0), strict=True))
            if first[3] == 4 and first[5] & 0x10:
                position += 10
        if position > size:
            raise ValueError("truncated ID3 tag")
        count, expected_frames = 0, None
        samples_per_frame = 1152
        while position < size:
            stream.seek(position)
            header = stream.read(4)
            if size - position == 128 and header[:3] == b"TAG":
                position = size
                break
            if len(header) != 4:
                raise ValueError("truncated MP3 header")
            word = int.from_bytes(header, "big")
            version, layer = (word >> 19) & 3, (word >> 17) & 3
            if (word >> 21) != 0x7FF or version == 1 or layer != 1:
                raise ValueError("invalid layer III frame")
            bitrate_index, rate_index = (word >> 12) & 15, (word >> 10) & 3
            if bitrate_index in {0, 15} or rate_index == 3:
                raise ValueError("unsupported MP3 frame")
            table = ((0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
                     if version == 3 else (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160))
            bitrate = table[bitrate_index] * 1000
            actual_rate = (44_100, 48_000, 32_000)[rate_index] // (1 if version == 3 else 2 if version == 2 else 4)
            actual_channels = 1 if ((word >> 6) & 3) == 3 else 2
            if require_spec and (version != 3 or bitrate != BITRATE_BPS or actual_rate != rate or channels != actual_channels):
                raise ValueError("MP3 is not 192 kbps CBR at the required format")
            length = (144 if version == 3 else 72) * bitrate // actual_rate + ((word >> 9) & 1)
            if position + length > size:
                raise ValueError("truncated MP3 frame")
            if count == 0:
                # Xing/Info's frame count excludes the metadata frame itself.
                side = (17 if actual_channels == 1 else 32) if version == 3 else (9 if actual_channels == 1 else 17)
                crc = 0 if ((word >> 16) & 1) else 2
                stream.seek(position + 4 + crc + side)
                tag = stream.read(12)
                if len(tag) == 12 and tag[:4] in {b"Xing", b"Info"} and int.from_bytes(tag[4:8], "big") & 1:
                    expected_frames = int.from_bytes(tag[8:12], "big")
            position += length
            count += 1
            samples_per_frame = 1152 if version == 3 else 576
        if count < 2 or (expected_frames is not None and count != expected_frames + 1):
            raise ValueError("MP3 frame count is incomplete")
    return {"mpeg_frames": count, "mpeg_samples_per_frame": samples_per_frame}


async def decoder_identity(*, context: ExecutionContext | None = None) -> str:
    """Current FFmpeg build identity, for cross-operation PCM checkpoint guards."""
    output, _ = await _command("ffmpeg", "-version", context=context)
    # Configuration and library versions distinguish builds with identical
    # release numbers; do not include executable locations or user paths.
    return hashlib.sha256(output).hexdigest()


async def check_mp3_encoder(*, context: ExecutionContext | None = None) -> dict[str, str]:
    """Release preflight and runtime check: codec presence plus tool identity."""
    encoders, _ = await _command("ffmpeg", "-hide_banner", "-encoders", context=context)
    if b"libmp3lame" not in encoders:
        raise AsrError(503, "mp3_encoder_unavailable", "ffmpeg requires the libmp3lame MP3 encoder")
    ffmpeg, _ = await _command("ffmpeg", "-version", context=context)
    ffprobe, _ = await _command("ffprobe", "-version", context=context)
    return {"encoder": ffmpeg.decode(errors="replace").splitlines()[0],
            "decoder": ffmpeg.decode(errors="replace").splitlines()[0],
            "probe": ffprobe.decode(errors="replace").splitlines()[0]}


def validate_retained_audio(path: Path, metadata: dict[str, Any]) -> None:
    """Recovery verifies the committed immutable file, including the entire hash."""
    if metadata.get("version") != RETAINED_AUDIO_VERSION:
        raise AsrError(409, "retained_audio_version_mismatch", "saved audio format version differs")
    try:
        valid = (metadata.get("state") == "ready" and metadata.get("codec") == "mp3"
                 and metadata.get("bitrate_bps") == BITRATE_BPS and not path.is_symlink()
                 and path.is_file() and path.stat().st_size == metadata.get("size_bytes")
                 and _hash(path) == metadata.get("sha256"))
        if not valid:
            raise ValueError
        _mp3_frames(path, rate=int(metadata["sample_rate"]), channels=int(metadata["channels"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AsrError(409, "retained_audio_invalid", "saved MP3 is missing or its identity differs") from exc


async def _encode(source: Path, partial: Path, info: dict[str, Any], *, max_output_bytes: int,
                  min_free_disk_bytes: int, context: ExecutionContext | None,
                  output_written: Callable[[int], Awaitable[None]] | None) -> None:
    _check_cancel(context)
    # -fs bounds encoder writes even if the estimate understates actual duration.
    # FFmpeg may exceed -fs by one packet; the exact post-write guard rejects it.
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
        *INPUT_OPTIONS, "-i", str(source.resolve()), *_audio_options(info),
        "-map_metadata", "-1", "-c:a", "libmp3lame", "-b:a", "192k", "-abr", "0",
        "-write_xing", "1", "-id3v2_version", "3", "-fs", str(max_output_bytes),
        "-f", "mp3", "-n", str(partial.resolve()),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )

    async def collect() -> None:
        output_task = asyncio.create_task(_read_tail(process.stdout, 1024))
        errors_task = asyncio.create_task(_read_tail(process.stderr, STDERR_TAIL_BYTES))
        try:
            while process.returncode is None:
                actual_size = partial.stat().st_size if partial.exists() else 0
                if output_written is not None:
                    await output_written(actual_size)
                if actual_size > max_output_bytes:
                    raise AsrError(503, "storage_unavailable", "MP3 exceeds its output byte budget")
                if shutil.disk_usage(partial.parent).free < min_free_disk_bytes:
                    raise AsrError(503, "storage_unavailable", "not enough free disk space")
                await asyncio.wait({errors_task}, timeout=0.05)
                if errors_task.done():
                    await process.wait()
            _, errors = await asyncio.gather(output_task, errors_task)
            if process.returncode != 0 or errors:
                raise AsrError(400, "audio_encode_failed", "MP3 encoding could not complete")
        finally:
            for task in (output_task, errors_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(output_task, errors_task, return_exceptions=True)

    await _guard_operation(process, collect(), timeout_seconds=OPERATION_TIMEOUT_SECONDS,
                           timeout_code="audio_preprocess_timeout", cancel_event=_event(context))
    if output_written is not None:
        await output_written(partial.stat().st_size)
    if partial.stat().st_size > max_output_bytes:
        raise AsrError(503, "storage_unavailable", "MP3 exceeds its output byte budget")


async def prepare_retained_audio(source: Path, target: Path, *, max_duration_seconds: float,
                                 max_output_bytes: int, min_free_disk_bytes: int,
                                 context: ExecutionContext | None = None,
                                 ensure_output_capacity: Callable[[int], Awaitable[None]] | None = None,
                                 output_written: Callable[[int], Awaitable[None]] | None = None) -> dict[str, Any]:
    """Produce/reuse verified CBR audio without ever deleting the original upload."""
    if not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0 or max_duration_seconds > MAX_DURATION_SECONDS:
        raise AsrError(400, "bad_request", "invalid retained audio duration limit")
    if max_output_bytes <= 0 or min_free_disk_bytes < 0 or source.resolve() == target.resolve():
        raise AsrError(400, "bad_request", "invalid retained audio output budget or path")
    if target.exists() or target.is_symlink():
        raise AsrError(409, "retained_audio_exists", "retained audio destination already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".partial" + target.suffix)
    if partial.exists() or partial.is_symlink():
        raise AsrError(409, "retained_audio_preparing", "retained audio has an unfinished preparation")
    try:
        async with _decode_slot():
            tools = await check_mp3_encoder(context=context)
            info = await probe_retained_source(source, context=context)
            if info["codec"] == "mp3":
                try:
                    await asyncio.to_thread(_mp3_frames, source, rate=int(info["sample_rate"]),
                                            channels=int(info["channels"]), require_spec=False)
                except (ValueError, OSError) as exc:
                    raise AsrError(400, "audio_integrity_failed", "source MP3 has incomplete or invalid frames") from exc
            source_frames = await _decoded_frames(source, info, max_duration=max_duration_seconds, context=context)
            rate, channels = int(info["target_sample_rate"]), int(info["target_channels"])
            duration = source_frames / rate
            source_size = source.stat().st_size
            reusable = info["codec"] == "mp3" and info["sample_rate"] == rate and info["channels"] == channels
            if reusable:
                try:
                    await asyncio.to_thread(_mp3_frames, source, rate=rate, channels=channels)
                except (ValueError, OSError):
                    reusable = False
            if reusable:
                if ensure_output_capacity is not None:
                    await ensure_output_capacity(source_size)
                if source_size > max_output_bytes:
                    raise AsrError(503, "storage_unavailable", "MP3 exceeds its output byte budget")
                written = 0
                with source.open("rb") as original, partial.open("xb", buffering=0) as output:
                    while block := original.read(1024 * 1024):
                        _check_cancel(context)
                        if shutil.disk_usage(partial.parent).free - len(block) < min_free_disk_bytes:
                            raise AsrError(503, "storage_unavailable", "not enough free disk space")
                        if ensure_output_capacity is not None:
                            await ensure_output_capacity(written + len(block))
                        output.write(block)
                        written += len(block)
                        if output_written is not None:
                            await output_written(written)
                        await asyncio.sleep(0)
                    output.flush()
                    os.fsync(output.fileno())
            else:
                # Source samples are already measured, so encoder writes can be
                # fully admitted before they start. At most one info frame and
                # two rounded audio frames plus ID3 metadata are needed.
                frame_bound = (math.ceil(source_frames / 1152) + 4) * (144 * BITRATE_BPS // rate + 1) + 4096
                if frame_bound > max_output_bytes:
                    raise AsrError(503, "storage_unavailable", "MP3 exceeds its output byte budget")
                if ensure_output_capacity is not None:
                    await ensure_output_capacity(frame_bound)
                if shutil.disk_usage(partial.parent).free - frame_bound < min_free_disk_bytes:
                    raise AsrError(503, "storage_unavailable", "not enough free disk space")
                await _encode(source, partial, info, max_output_bytes=frame_bound,
                              min_free_disk_bytes=min_free_disk_bytes, context=context,
                              output_written=output_written)
            output_info = await probe_retained_source(partial, context=context)
            try:
                frames_info = await asyncio.to_thread(_mp3_frames, partial, rate=rate, channels=channels)
            except (ValueError, OSError) as exc:
                raise AsrError(400, "audio_integrity_failed", "encoded MP3 has incomplete or invalid frames") from exc
            actual_frames = await _decoded_frames(partial, output_info, max_duration=max_duration_seconds, context=context)
            # Seekable LAME/Xing output carries gapless delay/padding metadata.
            # Both decodes must expose the same effective sample count; a missing
            # tail cannot pass as a shorter successful conversion.
            if abs(actual_frames - source_frames) > 1:
                raise AsrError(400, "audio_integrity_failed", "MP3 effective duration differs from the complete source track")
            if output_info["codec"] != "mp3" or output_info["sample_rate"] != rate or output_info["channels"] != channels:
                raise AsrError(400, "audio_integrity_failed", "MP3 encoding specifications differ")
            _check_cancel(context)
            with partial.open("rb") as output:
                os.fsync(output.fileno())
            digest = await asyncio.to_thread(_hash, partial)
            size = partial.stat().st_size
            if size > max_output_bytes:
                raise AsrError(503, "storage_unavailable", "MP3 exceeds its output byte budget")
            metadata: dict[str, Any] = {
                "state": "ready", "version": RETAINED_AUDIO_VERSION, "path": target.name,
                "sha256": digest, "size_bytes": size, "duration_seconds": duration,
                "sample_rate": rate, "channels": channels, "codec": "mp3", "bitrate_bps": BITRATE_BPS,
                "frame_count": actual_frames, "source_frame_count": source_frames,
                "source_stream_index": info["index"], "source_audio_streams": info["audio_streams"],
                "source_sample_rate": info["sample_rate"], "source_channels": info["channels"],
                "channel_processing": _mix_filter(int(info["channels"])) or "preserved",
                "timeline": {"origin_seconds": 0.0, "source_start_seconds": info["start_time_seconds"],
                             "effective_frames": actual_frames, "decoder_delay_padding_honored": True,
                             "duration_method": "complete_decoded_samples"},
                "encoder": "copy-verified-mp3" if reusable else tools["encoder"],
                "decoder": tools["decoder"], "decoder_identity": await decoder_identity(context=context),
                "probe": tools["probe"], "reused": reusable, **frames_info,
            }
            # Metadata is returned only after a durable atomic file commit. The
            # caller must persist this metadata before deleting its only source.
            partial.rename(target)
            directory = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return metadata
    finally:
        partial.unlink(missing_ok=True)
