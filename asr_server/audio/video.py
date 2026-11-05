"""Bounded video retention with measured progress. Never used as ASR input."""
from __future__ import annotations

import asyncio
import json
import math
import os
from time import perf_counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from asr_server.audio.preprocess import INPUT_OPTIONS, STDERR_TAIL_BYTES, _guard_operation
from asr_server.audio.retained import _command, _hash, _check_cancel, _event
from asr_server.errors import AsrError
from asr_server.observability import ExecutionContext

VIDEO_PROFILE = {"version": "h264-1080-v1", "codec": "h264", "pixel_format": "yuv420p", "crf": 22,
                 "preset": "fast", "max_fps": 30, "max_long_edge": 1920, "max_short_edge": 1080,
                 "audio_codec": "aac", "audio_bitrate_bps": 128000, "threads": 2}
VIDEO_TIMEOUT = 43200.0
VideoProgress = Callable[[dict[str, Any]], Awaitable[None]]
NVENC_OPTIONS = ["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "hq", "-rc", "vbr", "-cq", "23", "-b:v", "0"]


async def select_video_encoder(preference: str, context: ExecutionContext) -> str:
    if preference == "cpu":
        return "libx264"
    try:
        # A listed encoder alone does not prove the driver/device is usable.
        await _command("ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "color=size=320x180:rate=1",
                       "-frames:v", "1", *NVENC_OPTIONS, "-pix_fmt", "yuv420p", "-f", "null", "-", context=context)
        return "h264_nvenc"
    except AsrError:
        if preference == "nvenc":
            raise AsrError(503, "video_encoder_unavailable", "GPU video encoder is unavailable") from None
        return "libx264"


async def _read_progress(stream: asyncio.StreamReader, phase: str, duration: float,
                         progress: VideoProgress | None, frame_rate: float | None = None) -> bytes:
    """Keep bounded errors separate from FFmpeg's machine-readable progress."""
    errors = bytearray()
    values: dict[str, str] = {}
    keys = {"frame", "fps", "bitrate", "total_size", "out_time_us", "out_time_ms", "out_time",
            "dup_frames", "drop_frames", "speed", "progress"}
    while line := await stream.readline():
        key, sep, value = line.decode("utf-8", errors="replace").strip().partition("=")
        if not sep or (key not in keys and not key.startswith("stream_")):
            errors.extend(line)
            del errors[:-STDERR_TAIL_BYTES]
            continue
        values[key] = value
        if key != "progress":
            continue
        try:
            seconds = float(values.get("out_time_us", "N/A")) / 1_000_000
        except ValueError:
            seconds = math.nan
        # copyts + fragmented MP4 can report zero/N/A out_time until the muxer
        # flushes. Our fps filter produces a fixed-rate video, so encoded/copied
        # frames measure actual processed video time without polling media files.
        if frame_rate is not None:
            try:
                frames = float(values.get("frame", "N/A"))
                if math.isfinite(frames):
                    seconds = frames / frame_rate
            except ValueError:
                pass
        if progress is not None and math.isfinite(seconds):
            seconds = max(0.0, min(duration, seconds))
            await progress({"phase": phase, "processed_seconds": seconds, "duration_seconds": duration,
                            "percent": round(min(100.0, seconds / duration * 100), 1)})
        values.clear()
    return bytes(errors)


async def probe_video(path: Path, context: ExecutionContext | None = None) -> dict[str, Any] | None:
    output, _ = await _command("ffprobe", "-v", "error", *INPUT_OPTIONS,
        "-show_entries", "stream=index,codec_type,codec_name,width,height,pix_fmt,start_time,duration,avg_frame_rate,r_frame_rate,color_transfer,sample_aspect_ratio:stream_side_data=rotation:stream_disposition=attached_pic:format=start_time,duration",
        "-of", "json", str(path.resolve()), context=context)
    info = json.loads(output)
    videos = [s for s in info.get("streams", []) if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    if not videos:
        return None
    audios = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not audios:
        raise AsrError(400, "video_audio_missing", "video has no audio track")
    def number(value: Any) -> float:
        try:
            n = float(value)
            return n if math.isfinite(n) else 0.0
        except (TypeError, ValueError):
            return 0.0
    video, audio = videos[0], audios[0]
    duration = number(info.get("format", {}).get("duration"))
    if duration <= 0 or duration > 21601:
        raise AsrError(422, "video_duration_invalid", "video duration is unavailable or exceeds six hours")
    if video.get("color_transfer") in {"smpte2084", "arib-std-b67"}:
        raise AsrError(422, "video_format_unsupported", "HDR video retention requires an SDR source")
    return {"video": video, "audio": audio, "duration": duration,
            "origin": number(info.get("format", {}).get("start_time")),
            "audio_start": number(audio.get("start_time")), "video_start": number(video.get("start_time"))}


async def _file_command(arguments: list[str], context: ExecutionContext | None, *, phase: str,
                        duration: float, progress: VideoProgress | None, frame_rate: float | None = None) -> None:
    if progress is not None:
        await progress({"phase": phase, "processed_seconds": 0.0, "duration_seconds": duration, "percent": 0.0})
    _check_cancel(context)
    process = await asyncio.create_subprocess_exec(arguments[0], "-nostats", "-stats_period", "1", "-progress", "pipe:2",
        *arguments[1:], stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    async def collect() -> None:
        assert process.stderr is not None
        errors = await _read_progress(process.stderr, phase, duration, progress, frame_rate)
        await process.wait()
        if process.returncode or errors:
            raise AsrError(400, "video_integrity_failed", "video could not be processed completely")
    await _guard_operation(process, collect(), timeout_seconds=VIDEO_TIMEOUT,
                           timeout_code="video_preprocess_timeout", cancel_event=_event(context))


async def _read_block(stream: asyncio.StreamReader) -> bytes:
    # read(n) returns whatever is available (often a tiny pipe fragment). Exact
    # bounded batches keep SQLite fsyncs out of the per-packet critical path.
    try:
        return await stream.readexactly(1024 * 1024)
    except asyncio.IncompleteReadError as error:
        return error.partial


async def _gpu_input(source: Path, info: dict[str, Any], context: ExecutionContext) -> list[str]:
    video = info["video"]
    decoder = {"h264": "h264_cuvid", "hevc": "hevc_cuvid", "vp9": "vp9_cuvid", "av1": "av1_cuvid"}.get(video.get("codec_name"))
    if (decoder is None or any(abs(float(item.get("rotation", 0))) > .01 for item in video.get("side_data_list", []))
            or video.get("sample_aspect_ratio", "1:1") not in {"1:1", "0:1", "N/A"}):
        return []
    width, height = video["width"], video["height"]
    factor = min(1., 1920 / max(width, height), 1080 / min(width, height))
    width, height = max(2, int(width * factor) // 2 * 2), max(2, int(height * factor) // 2 * 2)
    options = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", decoder, "-resize", f"{width}x{height}"]
    try:
        await _command("ffmpeg", "-v", "error", "-nostdin", *options, *INPUT_OPTIONS, "-i", str(source.resolve()),
            "-map", f"0:{video['index']}", "-frames:v", "1", "-an", "-f", "null", "-", context=context)
    except AsrError:
        return []
    return options


async def _packet_hash(path: Path, context: ExecutionContext) -> bytes:
    output, _ = await _command("ffmpeg", "-v", "error", "-nostdin", *INPUT_OPTIONS, "-i", str(path.resolve()),
        "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", "-f", "streamhash", "-hash", "sha256", "-",
        context=context, timeout_seconds=VIDEO_TIMEOUT)
    return output


async def prepare_video(source: Path, target: Path, info: dict[str, Any], *, context: ExecutionContext,
                        ensure_capacity: Callable[[int], Awaitable[None]],
                        written: Callable[[int], Awaitable[None]], encoder: str = "libx264",
                        progress: VideoProgress | None = None, max_fps: int = 15) -> dict[str, Any]:
    """Stream encode with pre-write admission, then reserve the complete remux copy."""
    partial = target.with_name("video.partial.mp4")
    remux = target.with_name("video.faststart.partial.mp4")
    if target.exists():
        raise AsrError(409, "video_exists", "video destination already exists")
    if encoder not in {"libx264", "h264_nvenc"}:
        raise ValueError("unsupported video encoder")
    v, a = info["video"], info["audio"]
    started = perf_counter()
    phases: dict[str, float] = {}
    phase = "probe_decoder"
    phase_started = started
    callback = progress
    async def report(value: dict[str, Any]) -> None:
        nonlocal phase, phase_started
        moment = perf_counter()
        if value["phase"] != phase:
            phases[phase] = round(moment - phase_started, 3)
            phase, phase_started = value["phase"], moment
        if callback is not None:
            await callback({**value, "elapsed_seconds": round(moment - started, 1)})
    progress = report
    gpu_input = await _gpu_input(source, info, context) if encoder == "h264_nvenc" else []
    if (gpu_input and info["duration"] > 120 and (os.cpu_count() or 1) >= 8
            and abs(info["audio_start"]) < .001 and abs(info["video_start"]) < .001 and abs(info["origin"]) < .001):
        from asr_server.audio.video_parallel import prepare_parallel_video
        metadata = await prepare_parallel_video(source, target, info, context=context, gpu_input=gpu_input,
            max_fps=max_fps, ensure_capacity=ensure_capacity, written=written, progress=progress)
        metadata["preparation_seconds"] = round(perf_counter() - started, 3)
        return metadata
    paths = (partial, remux)
    try:
        # Software fallback retains the independent source check. Hardware
        # encoding decodes the source strictly once; output is checked in flight.
        if not gpu_input:
            await _file_command(["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-err_detect", "explode", "-threads", "2",
                *INPUT_OPTIONS, "-i", str(source.resolve()), "-map", f"0:{v['index']}", "-map", f"0:{a['index']}",
                "-f", "null", "-"], context, phase="checking_video", duration=info["duration"], progress=progress)
        scale = "scale=w='trunc(iw*min(1,min(1920/max(iw,ih),1080/min(iw,ih)))/2)*2':h='trunc(ih*min(1,min(1920/max(iw,ih),1080/min(iw,ih)))/2)*2'"
        num, _, den = str(v.get("avg_frame_rate", "30/1")).partition("/")
        try:
            fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            fps = 30.0
        frame_rate = min(fps, max_fps) if math.isfinite(fps) and fps > 0 else float(max_fps)
        scale = f"fps={frame_rate:.9g}" if gpu_input else scale + f",fps={frame_rate:.9g}"
        if progress is not None:
            await progress({"phase": "encoding_video", "processed_seconds": 0.0, "duration_seconds": info["duration"], "percent": 0.0})
        options = NVENC_OPTIONS if encoder == "h264_nvenc" else ["-c:v", "libx264", "-preset", "fast", "-crf", "22"]
        process = await asyncio.create_subprocess_exec("ffmpeg", "-v", "error", "-nostdin", "-nostats", "-stats_period", "1", "-progress", "pipe:2", "-xerror", "-err_detect", "explode", "-threads", "2",
            "-copyts", "-start_at_zero", *gpu_input, *INPUT_OPTIONS, "-i", str(source.resolve()),
            "-map", f"0:{v['index']}", "-map", f"0:{a['index']}", "-map_metadata", "-1", "-sn", "-dn",
            "-vf", scale, "-filter_threads", "1", *options, "-threads", "2",
            *([] if gpu_input else ["-pix_fmt", "yuv420p"]), "-fps_mode", "vfr", "-c:a", "aac", "-b:a", "128k", "-ac", "2",
            "-frag_duration", "2000000", "-movflags", "+frag_keyframe+delay_moov+default_base_moof", "-f", "mp4", "pipe:1",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        async def collect() -> None:
            assert process.stdout is not None
            assert process.stderr is not None
            errors = asyncio.create_task(_read_progress(process.stderr, "encoding_video", info["duration"], progress, frame_rate))
            total = 0
            reported = 0
            verifier = None
            validation_errors = None
            if gpu_input:
                verifier = await asyncio.create_subprocess_exec("ffmpeg", "-v", "error", "-nostdin", "-xerror", "-err_detect", "explode",
                    "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", "h264_cuvid",
                    "-protocol_whitelist", "pipe", "-format_whitelist", "mov", "-i", "pipe:0",
                    "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-",
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
                assert verifier.stderr is not None
                validation_errors = asyncio.create_task(_read_progress(verifier.stderr, "verifying_video", info["duration"], None))
            try:
                with partial.open("xb") as output:
                    while block := await _read_block(process.stdout):
                        if errors.done():
                            errors.result()  # Propagate progress persistence/cancellation failures promptly.
                        _check_cancel(context)
                        await ensure_capacity(total + len(block))
                        output.write(block)
                        total += len(block)
                        if total - reported >= 4 * 1024 * 1024:
                            await written(total)
                            reported = total
                        if verifier is not None:
                            assert verifier.stdin is not None
                            try:
                                verifier.stdin.write(block)
                                await verifier.stdin.drain()
                            except (BrokenPipeError, ConnectionResetError) as error:
                                raise AsrError(400, "video_integrity_failed", "encoded video failed full decode") from error
                    await written(total)
                    output.flush()
                    os.fsync(output.fileno())
                await process.wait()
                if process.returncode or await errors:
                    raise AsrError(400, "video_encode_failed", "video encoding failed")
                if verifier is not None:
                    assert verifier.stdin is not None and validation_errors is not None
                    verifier.stdin.close()
                    await verifier.wait()
                    if verifier.returncode or await validation_errors:
                        raise AsrError(400, "video_integrity_failed", "encoded video failed full decode")
            finally:
                if verifier is not None:
                    if verifier.returncode is None:
                        verifier.kill()
                    await verifier.wait()
                if validation_errors is not None:
                    await asyncio.gather(validation_errors, return_exceptions=True)
                if not errors.done():
                    errors.cancel()
                await asyncio.gather(errors, return_exceptions=True)
        await _guard_operation(process, collect(), timeout_seconds=VIDEO_TIMEOUT,
                               timeout_code="video_preprocess_timeout", cancel_event=_event(context))
        # The second file is admitted in full before FFmpeg writes it. Remux only;
        # faststart needs a seekable destination. Reserve generous metadata overhead.
        bound = partial.stat().st_size * 2 + 16 * 1024 * 1024
        await ensure_capacity(bound)
        await _file_command(["ffmpeg", "-v", "error", "-nostdin", "-xerror", *INPUT_OPTIONS,
            "-copyts", "-i", str(partial.resolve()), "-map", "0", "-c", "copy", "-movflags", "+faststart",
            "-fs", str(bound - partial.stat().st_size), "-n", str(remux.resolve())], context,
            phase="finalizing_video", duration=info["duration"], progress=progress, frame_rate=frame_rate)
        await written(partial.stat().st_size + remux.stat().st_size)
        output = await probe_video(remux, context)
        if output is None:
            raise AsrError(400, "video_integrity_failed", "encoded video stream missing")
        if gpu_input:
            # Both encoded tracks were fully decoded while encoding. Verify that
            # the seekable faststart remux contains exactly those tested packets.
            await progress({"phase": "verifying_video", "processed_seconds": 0., "duration_seconds": info["duration"], "percent": 0.})
            if await _packet_hash(partial, context) != await _packet_hash(remux, context):
                raise AsrError(400, "video_integrity_failed", "remux changed validated media packets")
            await progress({"phase": "verifying_video", "processed_seconds": info["duration"], "duration_seconds": info["duration"], "percent": 100.})
        else:
            # Full decode of both retained tracks, never just a metadata probe.
            await _file_command(["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-err_detect", "explode", "-threads", "2",
                *INPUT_OPTIONS, "-i", str(remux.resolve()), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"], context,
                phase="verifying_video", duration=info["duration"], progress=progress)
        encoded = output["video"]
        if (encoded.get("codec_name") != "h264" or encoded.get("pix_fmt") != "yuv420p"
                or max(encoded["width"], encoded["height"]) > 1920 or min(encoded["width"], encoded["height"]) > 1080
                or abs(output["duration"] - info["duration"]) > 0.15):
            raise AsrError(400, "video_integrity_failed", "encoded video specifications or duration differ")
        # Fragmented MP4 may shift both tracks for encoder delay. Preserve that
        # common shift in the transcript-to-video mapping instead of guessing zero.
        expected_relative = info["audio_start"] - info["video_start"]
        actual_relative = output["audio_start"] - output["video_start"]
        if abs(expected_relative - actual_relative) > 0.15:
            raise AsrError(400, "video_timeline_invalid", "encoded audio and video offsets differ")
        _check_cancel(context)
        digest = await asyncio.to_thread(_hash, remux)
        with remux.open("rb") as file:
            os.fsync(file.fileno())
        # AAC output start includes encoder priming; use video PTS shift to map
        # the original effective audio start (which also defines MP3 time zero).
        offset = output["video_start"] + expected_relative
        profile = dict(VIDEO_PROFILE)
        if encoder == "h264_nvenc":
            profile.pop("crf")
            profile.update(version="h264-1080-nvenc-v2", preset="p1", cq=23, rate_control="vbr")
        phases[phase] = round(perf_counter() - phase_started, 3)
        profile["max_fps"] = max_fps
        metadata = {**profile, "hardware_decode": bool(gpu_input), "timings_seconds": phases,
                    "preparation_seconds": round(perf_counter() - started, 3), "source_width": v["width"], "source_height": v["height"], "encoder": encoder, "state": "ready", "sha256": digest, "size_bytes": remux.stat().st_size,
                    "width": encoded["width"], "height": encoded["height"], "duration_seconds": output["duration"],
                    "audio_offset_seconds": offset, "source_stream_index": v["index"], "source_audio_stream_index": a["index"]}
        remux.replace(target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return metadata
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


def validate_video(path: Path, metadata: dict[str, Any]) -> None:
    if not path.is_file() or path.stat().st_size != metadata.get("size_bytes") or _hash(path) != metadata.get("sha256"):
        raise AsrError(409, "retained_video_invalid", "retained video failed identity validation")
