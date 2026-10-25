from __future__ import annotations

import io
import wave
from array import array


def make_wav(duration_seconds: float, amplitude: int = 10_000,
             sample_rate: int = 16_000, channels: int = 1) -> bytes:
    """Small deterministic PCM fixture; callers choose silence explicitly."""
    samples = array("h", [amplitude]) * (int(duration_seconds * sample_rate) * channels)
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())
    return output.getvalue()
