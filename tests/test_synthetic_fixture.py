from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest

from scripts.generate_test_audio import FLITE_REVISION, OUTPUT, REFERENCE, generate, pcm_16khz


def test_published_fixture_has_consistent_synthetic_provenance_and_reference() -> None:
    info = json.loads(OUTPUT.with_suffix(".source.json").read_text())
    assert info["kind"] == "synthetic_speech_fixture" and info["voice"] == "kal"
    assert info["synthesizer_revision"] == FLITE_REVISION
    assert len(info["synthesizer_binary_sha256"]) == 64 and not info["revision_verified_by_generator"]
    assert info["sha256"] == hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    assert info["reference_sha256"] == hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    assert info["reference_file"] == REFERENCE.name
    assert info["quality_validation"] in {
        "not_run_on_real_asr_or_forced_alignment", "short_qwen_workbench_gpu_smoke_passed",
    }
    if info["quality_validation"] == "short_qwen_workbench_gpu_smoke_passed":
        record = (OUTPUT.parent / info["validation_record"]).resolve()
        assert record.is_relative_to(OUTPUT.parents[2] / "docs" / "validation")
        assert record.is_file()
    with wave.open(str(OUTPUT), "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)
        assert audio.getnframes() == info["frames"]
        assert audio.getnframes() / audio.getframerate() == info["duration_seconds"]
    assert 1 < info["duration_seconds"] <= 30
    assert OUTPUT.stat().st_size < 1024 * 1024


def test_integer_resampling_preserves_endpoints_and_exact_frame_count() -> None:
    raw = struct.pack("<hhh", -1000, 1000, 3000)
    assert struct.unpack("<hhhhhh", pcm_16khz(raw, 8000)) == (-1000, 0, 1000, 2000, 3000, 3000)
    assert pcm_16khz(raw, 16000) == raw
    with pytest.raises(ValueError, match="incomplete"):
        pcm_16khz(b"x", 8000)


def test_generator_does_not_overwrite_or_record_host_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = tmp_path / "reference.txt"
    reference.write_text("Synthetic fixture.\n")
    reference_digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    flite = tmp_path / "flite"
    flite.write_bytes(b"synthetic executable identity")
    output = tmp_path / "fixture.wav"

    def synthesis(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert command[1:4] == ["-voice", "kal", "-t"]
        assert command[4] == "Synthetic fixture."
        reference.write_text("Changed during synthesis.\n")
        with wave.open(command[-1], "wb") as native:
            native.setnchannels(1)
            native.setsampwidth(2)
            native.setframerate(8000)
            native.writeframes(struct.pack("<hhh", -1000, 1000, 3000))
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr("scripts.generate_test_audio.subprocess.run", synthesis)
    info = generate(flite, output, reference)
    assert info["frames"] == 6 and info["reference_file"] == "reference.txt"
    assert str(tmp_path) not in json.dumps(info)
    assert info["reference_sha256"] == reference_digest
    with pytest.raises(ValueError, match="outputs exist"):
        generate(tmp_path / "flite", output, reference)


def test_generator_rejects_outputs_that_overwrite_the_reference(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_text("Synthetic reference input.\n")
    with pytest.raises(ValueError, match="must not replace"):
        generate(tmp_path / "unused", reference, reference, replace=True)
    assert reference.read_text() == "Synthetic reference input.\n"


def test_generator_rejects_truncated_synthesis_before_publishing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference.txt"
    reference.write_text("Synthetic reference.\n")
    output = tmp_path / "fixture.wav"

    def synthesis(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        target = Path(command[-1])
        with wave.open(str(target), "wb") as native:
            native.setnchannels(1)
            native.setsampwidth(2)
            native.setframerate(8000)
            native.writeframes(struct.pack("<hhh", -1000, 1000, 3000))
        target.write_bytes(target.read_bytes()[:-2])
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr("scripts.generate_test_audio.subprocess.run", synthesis)
    with pytest.raises(ValueError, match="truncated"):
        generate(tmp_path / "unused", output, reference)
    assert not output.exists() and not output.with_suffix(".source.json").exists()
