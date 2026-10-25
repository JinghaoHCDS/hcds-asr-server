from __future__ import annotations

import asyncio
import hashlib
import math
import tempfile
import threading
import wave
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Coroutine, TypeVar

from asr_server.audio.assets import AudioSpec, DEFAULT_AUDIO_SPEC, NormalizedAudioAsset
from asr_server.audio.workspace import ResourceLease, service_temporary_root
from asr_server.errors import AsrError
from asr_server.observability import emit, safe_exception


MAX_AUDIO_SECONDS = 21_600.0
PCM_BLOCK_BYTES = 64 * 1024
STDERR_TAIL_BYTES = 8 * 1024
WAV_HEADER_BYTES = 44
# Deliberately exclude playlist/script demuxers (hls, dash, concat, sdp, ...).
# MOV's enable_drefs/use_absolute_path defaults remain disabled. An extension
# is never used to grant a demuxer or protocol permission.
ALLOWED_DEMUXERS = "wav,mp3,flac,aac,ogg,mov,matroska,webm"
INPUT_OPTIONS = ["-protocol_whitelist", "file", "-format_whitelist", ALLOWED_DEMUXERS]
_T = TypeVar("_T")
_decode_slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, weakref.ReferenceType[asyncio.Lock]] = weakref.WeakKeyDictionary()
_decode_slots_lock = threading.Lock()


def _decode_slot() -> asyncio.Lock:
    """Share one decoder per event loop without retaining closed sync loops."""
    loop = asyncio.get_running_loop()
    with _decode_slots_lock:
        reference = _decode_slots.get(loop)
        slot = reference() if reference is not None else None
        if slot is None:
            slot = asyncio.Lock()
            _decode_slots[loop] = weakref.ref(slot)
        return slot


@dataclass(frozen=True)
class NormalizedAudio:
    audio: bytes
    decode_ms: float


@dataclass(frozen=True)
class NormalizedAudioPath:
    path: Path
    decode_ms: float
    asset: NormalizedAudioAsset | None = None


async def _read_tail(stream: asyncio.StreamReader | None, limit: int) -> bytes:
    if stream is None:
        return b""
    tail = bytearray()
    while block := await stream.read(PCM_BLOCK_BYTES):
        tail.extend(block)
        if len(tail) > limit:
            del tail[:-limit]
    return bytes(tail)


async def _discard_stream(stream: asyncio.StreamReader | None) -> None:
    if stream is not None:
        while await stream.read(PCM_BLOCK_BYTES):
            pass


async def _stop_subprocess(process: asyncio.subprocess.Process) -> None:
    # Draining after cancelling the normal pumps prevents asyncio's subprocess
    # waiter from being held by a paused, full pipe when the decoder is killed.
    drains = [asyncio.create_task(_discard_stream(process.stdout)),
              asyncio.create_task(_discard_stream(process.stderr))]
    try:
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await asyncio.wait_for(process.wait(), timeout=5)
    finally:
        for task in drains:
            if not task.done():
                task.cancel()
        await asyncio.gather(*drains, return_exceptions=True)


async def _guard_operation(process: asyncio.subprocess.Process, operation: Coroutine[object, object, _T], *,
                           timeout_seconds: float, timeout_code: str,
                           cancel_event: threading.Event | None) -> _T:
    task = asyncio.create_task(operation)
    try:
        async with asyncio.timeout(timeout_seconds):
            while not task.done():
                if cancel_event is not None and cancel_event.is_set():
                    raise asyncio.CancelledError
                await asyncio.wait({task}, timeout=0.05)
            return await task
    except TimeoutError as exc:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await _stop_subprocess(process)
        raise AsrError(408, timeout_code, "media decoder exceeded its operation deadline") from exc
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await _stop_subprocess(process)
        raise
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise AsrError(400, "bad_request", f"{name} must be finite and greater than zero")


class MediaPreprocessor:
    """One bounded protocol, deadline and PCM-writing policy for every entry."""

    def __init__(self, *, spec: AudioSpec = DEFAULT_AUDIO_SPEC,
                 max_output_bytes: int | None = None,
                 max_duration_seconds: float = MAX_AUDIO_SECONDS) -> None:
        _positive_finite(max_duration_seconds, "max_duration_seconds")
        self.spec = spec
        self.max_frames = int(max_duration_seconds * spec.sample_rate)
        duration_bytes = self.max_frames * spec.frame_bytes + WAV_HEADER_BYTES
        self.max_output_bytes = duration_bytes if max_output_bytes is None else max_output_bytes
        if self.max_output_bytes < WAV_HEADER_BYTES:
            raise AsrError(503, "storage_unavailable", "normalized audio budget cannot hold a WAV header")

    async def probe(self, path: Path, *, timeout_seconds: float = 30.0,
                    cancel_event: threading.Event | None = None) -> float | None:
        started = perf_counter()
        try:
            duration = await self._probe(path, timeout_seconds=timeout_seconds, cancel_event=cancel_event)
        except BaseException as error:
            emit("media.probe_failed", duration_ms=(perf_counter() - started) * 1000,
                 error_code=getattr(error, "code", "media_cancelled" if isinstance(error, asyncio.CancelledError) else "media_probe_failed"),
                 diagnostics=safe_exception(error))
            raise
        try:
            input_bytes: int | None = path.stat().st_size
        except OSError:
            input_bytes = None
        emit("media.probed", duration_ms=(perf_counter() - started) * 1000,
             input_bytes=input_bytes, audio_seconds=duration,
             decoder_exit="ok" if duration is not None else "unsupported")
        return duration

    async def _probe(self, path: Path, *, timeout_seconds: float = 30.0,
                     cancel_event: threading.Event | None = None) -> float | None:
        _positive_finite(timeout_seconds, "timeout_seconds")
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        try:
            process = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "error", *INPUT_OPTIONS,
                "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path.resolve()),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return None

        async def collect() -> tuple[bytes, bytes]:
            values = await asyncio.gather(_read_tail(process.stdout, 4096), _read_tail(process.stderr, STDERR_TAIL_BYTES))
            await process.wait()
            return values[0], values[1]

        stdout, _stderr = await _guard_operation(process, collect(), timeout_seconds=timeout_seconds,
                                                 timeout_code="audio_probe_timeout", cancel_event=cancel_event)
        if process.returncode != 0:
            return None
        try:
            duration = float(stdout.decode("utf-8", errors="replace").strip())
        except ValueError:
            return None
        return duration if math.isfinite(duration) and duration > 0 else None

    async def normalize(self, input_path: Path, output_path: Path, *, timeout_seconds: float = 1800.0,
                        cancel_event: threading.Event | None = None,
                        owner: str = "workspace", resource_lease: ResourceLease | None = None) -> NormalizedAudioPath:
        started = perf_counter()
        queue_started = perf_counter()
        try:
            async with _decode_slot():
                queue_wait_ms = (perf_counter() - queue_started) * 1000
                # _normalize does not return/raise until decoder reaping and
                # partial-file cleanup complete, including cancellation.
                result = await self._normalize(input_path, output_path, timeout_seconds=timeout_seconds,
                                               cancel_event=cancel_event, owner=owner, resource_lease=resource_lease)
        except BaseException as error:
            emit("media.normalization_failed", duration_ms=(perf_counter() - started) * 1000,
                 budget_bytes=self.max_output_bytes,
                 error_code=getattr(error, "code", "media_cancelled" if isinstance(error, asyncio.CancelledError) else "media_decode_failed"),
                 diagnostics=safe_exception(error))
            raise
        asset = result.asset
        emit("media.normalized", duration_ms=result.decode_ms, queue_wait_ms=queue_wait_ms,
             output_bytes=output_path.stat().st_size,
             audio_seconds=asset.duration if asset else None, frames=asset.frame_count if asset else None,
             sample_rate=self.spec.sample_rate, channels=self.spec.channels,
             normalization_version=asset.normalization_version if asset else None,
             budget_bytes=self.max_output_bytes, decoder_exit="ok")
        return result

    async def _normalize(self, input_path: Path, output_path: Path, *, timeout_seconds: float = 1800.0,
                         cancel_event: threading.Event | None = None,
                         owner: str = "workspace", resource_lease: ResourceLease | None = None) -> NormalizedAudioPath:
        _positive_finite(timeout_seconds, "timeout_seconds")
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        if input_path.resolve() == output_path.resolve():
            raise AsrError(400, "bad_request", "normalization output must differ from the source")
        started = perf_counter()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", *INPUT_OPTIONS,
                "-i", str(input_path.resolve()), "-map", "0:a:0", "-ac", str(self.spec.channels),
                "-ar", str(self.spec.sample_rate), "-vn", "-acodec", "pcm_s16le", "-f", "s16le", "pipe:1",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise AsrError(503, "audio_preprocess_unavailable", "ffmpeg is not installed") from exc

        async def write_pcm() -> tuple[int, str]:
            assert process.stdout is not None
            total = 0
            digest = hashlib.sha256()
            pending = b""
            with output_path.open("wb", buffering=0) as output_file, wave.open(output_file, "wb") as target:
                target.setnchannels(self.spec.channels)
                target.setsampwidth(self.spec.sample_width)
                target.setframerate(self.spec.sample_rate)
                # Emit the empty header before the first output block. Every
                # later write, including final header patching, stays in budget.
                target.writeframesraw(b"")
                if resource_lease is not None:
                    await resource_lease.materialize(WAV_HEADER_BYTES)
                while block := await process.stdout.read(PCM_BLOCK_BYTES):
                    if cancel_event is not None and cancel_event.is_set():
                        raise asyncio.CancelledError
                    total += len(block)
                    if total > self.max_frames * self.spec.frame_bytes:
                        raise AsrError(422, "duration_limit_exceeded", "decoded audio exceeds the duration limit")
                    if total + WAV_HEADER_BYTES > self.max_output_bytes:
                        raise AsrError(503, "storage_unavailable", "decoded audio exceeds its output byte budget",
                                       {"max_output_bytes": self.max_output_bytes})
                    data = pending + block
                    aligned_bytes = len(data) - len(data) % self.spec.frame_bytes
                    pcm, pending = data[:aligned_bytes], data[aligned_bytes:]
                    target.writeframesraw(pcm)
                    if resource_lease is not None:
                        await resource_lease.materialize(len(pcm))
                    digest.update(pcm)
                if pending:
                    raise AsrError(400, "audio_decode_failed", "decoder produced an incomplete PCM frame")
            return total // self.spec.frame_bytes, digest.hexdigest()

        async def collect() -> tuple[int, str]:
            # Both pipes are always consumed. stderr is retained only as a
            # bounded internal tail; raw paths/free text never reach clients.
            pcm_task = asyncio.create_task(write_pcm())
            stderr_task = asyncio.create_task(_read_tail(process.stderr, STDERR_TAIL_BYTES))
            try:
                frame_count, digest = await pcm_task
                await stderr_task
                await process.wait()
                if process.returncode != 0:
                    raise AsrError(400, "audio_decode_failed", "audio decode failed")
                return frame_count, digest
            finally:
                for task in (pcm_task, stderr_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(pcm_task, stderr_task, return_exceptions=True)

        try:
            frame_count, digest = await _guard_operation(process, collect(), timeout_seconds=timeout_seconds,
                                                         timeout_code="audio_preprocess_timeout", cancel_event=cancel_event)
            if frame_count == 0:
                raise AsrError(400, "audio_decode_failed", "media has no decodable audio frames")
            asset = NormalizedAudioAsset(output_path, self.spec, frame_count, digest, owner=owner)
            return NormalizedAudioPath(output_path, (perf_counter() - started) * 1000, asset)
        except BaseException:
            output_path.unlink(missing_ok=True)
            raise


def _run_sync(operation: Coroutine[object, object, _T]) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(operation)
    # Compatibility for synchronous library callers already in an event loop.
    # Production async paths call MediaPreprocessor directly.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-media-sync") as executor:
        return executor.submit(asyncio.run, operation).result()


def probe_audio_duration_seconds(audio: bytes) -> float | None:
    with tempfile.TemporaryDirectory(prefix="probe_", dir=service_temporary_root()) as temporary:
        path = Path(temporary) / "upload"
        path.write_bytes(audio)
        return probe_audio_duration_path(path)


def probe_audio_duration_path(path: Path, *, timeout_seconds: float = 30.0,
                              cancel_event: threading.Event | None = None) -> float | None:
    return _run_sync(probe_audio_duration_path_async(path, timeout_seconds=timeout_seconds, cancel_event=cancel_event))


async def probe_audio_duration_path_async(path: Path, *, timeout_seconds: float = 30.0,
                                          cancel_event: threading.Event | None = None) -> float | None:
    return await MediaPreprocessor().probe(path, timeout_seconds=timeout_seconds, cancel_event=cancel_event)


def normalize_audio_to_wav(audio: bytes, *, timeout_seconds: float = 1800.0,
                           max_output_bytes: int | None = None,
                           max_duration_seconds: float = MAX_AUDIO_SECONDS,
                           spec: AudioSpec = DEFAULT_AUDIO_SPEC,
                           cancel_event: threading.Event | None = None) -> NormalizedAudio:
    with tempfile.TemporaryDirectory(prefix="decode_", dir=service_temporary_root()) as temporary:
        input_path, output_path = Path(temporary) / "upload", Path(temporary) / "normalized.wav"
        input_path.write_bytes(audio)
        normalized = normalize_audio_path_to_wav(input_path, output_path, timeout_seconds=timeout_seconds,
                                                max_output_bytes=max_output_bytes, max_duration_seconds=max_duration_seconds,
                                                spec=spec, cancel_event=cancel_event)
        return NormalizedAudio(output_path.read_bytes(), normalized.decode_ms)


def normalize_audio_path_to_wav(input_path: Path, output_path: Path, *, timeout_seconds: float = 1800.0,
                                max_output_bytes: int | None = None,
                                max_duration_seconds: float = MAX_AUDIO_SECONDS,
                                spec: AudioSpec = DEFAULT_AUDIO_SPEC,
                                cancel_event: threading.Event | None = None,
                                owner: str = "workspace") -> NormalizedAudioPath:
    return _run_sync(normalize_audio_path_to_wav_async(input_path, output_path, timeout_seconds=timeout_seconds,
                                                      max_output_bytes=max_output_bytes, max_duration_seconds=max_duration_seconds,
                                                      spec=spec, cancel_event=cancel_event, owner=owner))


async def normalize_audio_path_to_wav_async(input_path: Path, output_path: Path, *, timeout_seconds: float = 1800.0,
                                           max_output_bytes: int | None = None,
                                           max_duration_seconds: float = MAX_AUDIO_SECONDS,
                                           spec: AudioSpec = DEFAULT_AUDIO_SPEC,
                                           cancel_event: threading.Event | None = None,
                                           owner: str = "workspace",
                                           resource_lease: ResourceLease | None = None) -> NormalizedAudioPath:
    return await MediaPreprocessor(spec=spec, max_output_bytes=max_output_bytes,
                                   max_duration_seconds=max_duration_seconds).normalize(
        input_path, output_path, timeout_seconds=timeout_seconds, cancel_event=cancel_event, owner=owner,
        resource_lease=resource_lease)
