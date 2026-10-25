from __future__ import annotations

import math
import threading
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

from asr_server.errors import AsrError


@dataclass(frozen=True)
class QualityChunk:
    index: int
    start: float
    end: float
    core_start: float
    core_end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_quality_chunks(path: Path, target: float = 60.0, padding: float = 0.0, *, cancel_event: threading.Event | None = None) -> tuple[list[QualityChunk], float]:
    """Use bounded low-energy search without ever removing source samples.

    Core intervals tile the source exactly. Optional context is outside these
    ownership intervals and is reconciled using real alignment timestamps.
    A 100ms energy index is bounded to 216,000 values for a six-hour file.
    """
    if target not in {30, 60, 90, 120, 180} or padding not in {0, 1}:
        raise AsrError(400, "bad_request", "unsupported quality split candidate")
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        if source.getsampwidth() != 2 or source.getnchannels() != 1:
            raise AsrError(422, "invalid_audio", "quality splitting expects mono PCM16 WAV")
        duration = source.getnframes() / rate
        if not math.isfinite(duration) or duration <= 0 or duration > 21_600:
            raise AsrError(422, "duration_limit_exceeded", "audio duration must be between zero and six hours")
        frame_samples = max(int(rate / 10), 1)
        energies: list[float] = []
        while block := source.readframes(frame_samples):
            if cancel_event is not None and cancel_event.is_set():
                from asr_server.audio.splitter import SplitCancelled
                raise SplitCancelled()
            samples = array("h", block)
            energies.append(sum(abs(value) for value in samples) / max(len(samples), 1))
    # Context must also fit the model's 180-second input limit.
    core_limit = min(target, 180.0 - 2 * padding)
    boundaries = [0.0]
    while duration - boundaries[-1] > core_limit:
        if cancel_event is not None and cancel_event.is_set():
            from asr_server.audio.splitter import SplitCancelled
            raise SplitCancelled()
        start = boundaries[-1]
        desired = start + core_limit
        left = max(start + core_limit / 2, desired - 5)
        right = min(desired + 5, start + core_limit, duration)
        candidates = range(math.ceil(left * 10), min(math.floor(right * 10) + 1, len(energies)))
        cut_index = min(candidates, key=lambda i: (energies[i], abs(i / 10 - desired)), default=int(desired * 10))
        cut = min(max(cut_index / 10, start + 0.1), start + core_limit)
        boundaries.append(cut)
    boundaries.append(duration)
    chunks = [
        QualityChunk(index, max(0.0, start - padding), min(duration, end + padding), start, end)
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
    ]
    return chunks, duration


def audio_has_energy(path: Path, start: float, end: float, threshold: float = 30.0) -> bool:
    """A conservative acoustic hint, not a speech or accuracy classifier."""
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        source.setpos(min(int(start * rate), source.getnframes()))
        remaining = max(0, int((end - start) * rate))
        while remaining:
            count = min(remaining, rate)
            samples = array("h", source.readframes(count))
            if samples and sum(abs(value) for value in samples) / len(samples) > threshold:
                return True
            remaining -= count
    return False


def empty_audio_evidence(path: Path, start: float, end: float) -> dict[str, object] | None:
    """Validate an ASR-empty interval without treating low energy as no speech.

    Apart from exact digital silence, accept only a source tail with sparse
    PCM16 rounding residue (at most four quantization steps) followed by a
    substantial exact-zero suffix. Constant quiet signals, music, and even a
    single stronger sample remain unresolved. The residual-tail case must be
    exposed for review; this function never skips an ASR call.
    """
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2 or source.getnchannels() != 1:
            return None
        rate, frames = source.getframerate(), source.getnframes()
        first = max(0, min(round(start * rate), frames))
        stop = max(first, min(round(end * rate), frames))
        source.setpos(first)
        total = nonzero = peak = 0
        last_nonzero = -1
        while total < stop - first:
            samples = array("h", source.readframes(min(rate, stop - first - total)))
            if not samples:
                break
            for index, value in enumerate(samples):
                if value:
                    nonzero += 1
                    peak = max(peak, abs(value))
                    last_nonzero = total + index
            total += len(samples)
        if not total or total != stop - first:
            return None
        evidence: dict[str, object] = {
            "peak_pcm16": peak, "nonzero_samples": nonzero, "samples": total,
            "trailing_zero_seconds": (total - last_nonzero - 1) / rate,
        }
        if not nonzero:
            return {"method": "digital_silence", **evidence}
        trailing_zeros = total - last_nonzero - 1
        if stop == frames and peak <= 4 and nonzero * 20 <= total and trailing_zeros >= max(rate, total / 2):
            return {"method": "near_digital_silence_tail", **evidence}
    return None
