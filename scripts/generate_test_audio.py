"""Regenerate the public synthetic speech fixture with the pinned Flite kal voice.

Requires a locally installed/built Flite 2.2 executable, not an ASR model. The
script never downloads a voice, plays audio, or reads a user's recording.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "test-fixtures/audio/test_short.reference.txt"
OUTPUT = ROOT / "test-fixtures/audio/test_short.wav"
FLITE_REVISION = "e9e2e37c329dbe98bfeb27a1828ef9a71fa84f88"
SAMPLE_RATE = 16000


def pcm_16khz(raw: bytes, sample_rate: int) -> bytes:
    if len(raw) % 2:
        raise ValueError("synthesizer returned incomplete PCM samples")
    if sample_rate == SAMPLE_RATE:
        return raw
    if sample_rate != 8000:
        raise ValueError("kal synthesis must be 8000 or 16000 Hz")
    samples = [item[0] for item in struct.iter_unpack("<h", raw)]
    result = bytearray()
    for index, sample in enumerate(samples):
        following = samples[min(index + 1, len(samples) - 1)]
        result.extend(struct.pack("<hh", sample, (sample + following) // 2))
    return bytes(result)


def generate(flite: Path, output: Path, reference: Path = REFERENCE, *, replace: bool = False) -> dict[str, Any]:
    source_record = output.with_suffix(".source.json")
    if output.suffix.lower() != ".wav" or reference.resolve() in {output.resolve(), source_record.resolve()}:
        raise ValueError("fixture output must be a WAV and must not replace the reference")
    for path in (output, source_record):
        if path.is_symlink() or path.exists() and (not replace or not path.is_file()):
            raise ValueError("fixture outputs exist; use --replace deliberately")
    reference_bytes = reference.read_bytes()
    text = " ".join(reference_bytes.decode("utf-8").split())
    if not text or len(text) > 1000:
        raise ValueError("fixture reference must contain at most 1000 characters")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="asr-synthetic-fixture-") as directory:
        synthesized = Path(directory) / "kal.wav"
        subprocess.run([str(flite.resolve()), "-voice", "kal", "-t", text, "-o", str(synthesized)],
                       capture_output=True, check=True, timeout=30)
        with wave.open(str(synthesized), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getcomptype() != "NONE":
                raise ValueError("synthesizer must return mono PCM16")
            if not 0 < source.getnframes() <= source.getframerate() * 30:
                raise ValueError("synthetic fixture must be nonempty and at most 30 seconds")
            native_pcm = source.readframes(source.getnframes())
            if not native_pcm or len(native_pcm) != source.getnframes() * 2:
                raise ValueError("synthesizer returned truncated PCM data")
            pcm = pcm_16khz(native_pcm, source.getframerate())
        staging = Path(directory) / "fixture.wav"
        with wave.open(str(staging), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(SAMPLE_RATE)
            target.writeframes(pcm)
        data = staging.read_bytes()
    info: dict[str, Any] = {
        "kind": "synthetic_speech_fixture", "audio_file": output.name,
        "reference_file": reference.name, "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "generator": "scripts/generate_test_audio.py", "synthesizer": "Flite 2.2", "voice": "kal",
        "synthesizer_source": "https://github.com/festvox/flite", "synthesizer_revision": FLITE_REVISION,
        "synthesizer_binary_sha256": hashlib.sha256(flite.read_bytes()).hexdigest(),
        "revision_verified_by_generator": False,
        "sample_rate": SAMPLE_RATE, "channels": 1, "sample_width_bytes": 2,
        "frames": len(pcm) // 2, "duration_seconds": len(pcm) / (2 * SAMPLE_RATE),
        "sha256": hashlib.sha256(data).hexdigest(), "license": "MIT",
        "third_party_notices": "../../THIRD_PARTY_NOTICES.md",
        "quality_validation": "not_run_on_real_asr_or_forced_alignment",
    }
    # Fixture generation is a deliberate maintenance action. Never write audio
    # from a user's source path, and keep host paths out of the provenance.
    with output.open("wb" if replace else "xb") as stream:
        stream.write(data)
    with source_record.open("w" if replace else "x", encoding="utf-8") as stream:
        stream.write(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flite", type=Path, required=True, help="Flite 2.2 executable built at the documented revision")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        report = generate(args.flite, args.output, args.reference, replace=args.replace)
    except (OSError, ValueError, wave.Error, subprocess.SubprocessError) as error:
        parser.exit(1, f"synthetic fixture generation failed ({type(error).__name__})\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
