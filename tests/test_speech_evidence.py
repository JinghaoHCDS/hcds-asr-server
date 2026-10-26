from __future__ import annotations

import hashlib
import importlib.metadata
import sys
import wave
from array import array
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from asr_server.quality import speech_evidence


class Tensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def to(self, **_kwargs: object) -> Tensor:
        return Tensor(self.values.copy())

    def div_(self, denominator: float) -> Tensor:
        self.values = [value / denominator for value in self.values]
        return self

    def __getitem__(self, key: slice) -> Tensor:
        return Tensor(self.values[key])

    def __mul__(self, multiplier: float) -> Tensor:
        return Tensor([value * multiplier for value in self.values])

    def numel(self) -> int:
        return len(self.values)


class Model:
    def __init__(self) -> None:
        self.view = -1
        self.frame = 0
        self.outputs: dict[tuple[int, int], float | Exception] = {}
        self.calls: list[tuple[int, int, list[float]]] = []

    def eval(self) -> Model:
        return self

    def reset_states(self) -> None:
        self.view += 1
        self.frame = 0

    def __call__(self, frame: Tensor, sample_rate: int) -> Any:
        assert sample_rate == 16000 and frame.numel() == 512
        self.calls.append((self.view, self.frame, frame.values.copy()))
        output = self.outputs.get((self.view, self.frame), 0.01)
        self.frame += 1
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(item=lambda: output)


def write_wav(path: Path, samples: array[int], *, rate: int = 16000, channels: int = 1, width: int = 2) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(samples.tobytes())


@pytest.fixture
def detector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASR_QUALITY_VAD_DIR", raising=False)
    model_path = tmp_path / "silero_vad.jit"
    model_path.write_bytes(b"test-only fake model")
    monkeypatch.setattr(speech_evidence, "_MODEL_SHA256", hashlib.sha256(model_path.read_bytes()).hexdigest())
    monkeypatch.setattr(speech_evidence, "_MODEL_CACHE", None)
    distribution = SimpleNamespace(version="6.2.3", locate_file=lambda filename: model_path)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _name: distribution)
    model = Model()
    imports: list[str] = []
    loads: list[tuple[str, str]] = []

    def load(path: str, *, map_location: str) -> Model:
        loads.append((path, map_location))
        return model

    torch = SimpleNamespace(
        int16="int16", float32="float32", no_grad=nullcontext,
        frombuffer=lambda samples, **_kwargs: Tensor([float(value) for value in samples]),
        jit=SimpleNamespace(load=load),
        nn=SimpleNamespace(functional=SimpleNamespace(pad=lambda tensor, padding: Tensor(tensor.values + [0.0] * padding[1]))),
    )

    def import_module(name: str) -> Any:
        imports.append(name)
        assert name == "torch", "The optional package must never be imported for loading its model."
        return torch

    monkeypatch.setattr(importlib, "import_module", import_module)
    source = tmp_path / "audio.wav"
    write_wav(source, array("h", [1000]) * 16000)
    yield SimpleNamespace(path=source, model_path=model_path, distribution=distribution, model=model, imports=imports, loads=loads, torch=torch)


def test_identity_verifies_package_and_artifact_without_loading_torch(detector: SimpleNamespace) -> None:
    identity = speech_evidence.detector_identity()
    assert identity == {
        "available": True, "detector": "silero-vad", "version": "6.2.3",
        "model_sha256": hashlib.sha256(detector.model_path.read_bytes()).hexdigest(),
        "sample_rate": 16000, "frame_samples": 512, "threshold": 0.5,
        "target_peak": 0.9, "max_gain": 100, "timeout_seconds": 60.0,
    }
    assert detector.imports == [] and detector.loads == []


@pytest.mark.parametrize("explicit", [False, True])
def test_isolated_bundle_is_verified_without_adding_import_paths_or_using_global_package(
    detector: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: bool,
) -> None:
    identity = speech_evidence.detector_identity()
    bundle = tmp_path / ("custom-vad" if explicit else "data/quality-vad/silero-vad-6.2.3")
    bundle.mkdir(parents=True)
    if explicit:
        monkeypatch.setenv("ASR_QUALITY_VAD_DIR", str(bundle))
    detector.distribution.metadata = {"Name": "silero_vad"}
    searched: list[list[str]] = []
    global_queries: list[str] = []

    def distributions(*, path: list[str]) -> list[Any]:
        searched.append(path)
        return [detector.distribution]

    def global_distribution(name: str) -> Any:
        global_queries.append(name)
        return detector.distribution

    monkeypatch.setattr(importlib.metadata, "distributions", distributions)
    monkeypatch.setattr(importlib.metadata, "distribution", global_distribution)
    import_paths = sys.path.copy()
    assert speech_evidence.detector_identity() == identity
    assert searched and Path(searched[0][0]).resolve() == bundle.resolve()
    assert sys.path == import_paths and detector.imports == [] and global_queries == []
    detector.model_path.write_bytes(b"broken isolated model")
    assert speech_evidence.detector_identity()["available"] is False
    assert global_queries == []  # An invalid bundle must never fall back globally.


@pytest.mark.parametrize("directory", ["missing-directory", "empty-directory", ""])
def test_explicit_bad_bundle_never_falls_back_to_a_valid_global_install(
    detector: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directory: str,
) -> None:
    assert speech_evidence.detector_identity()["available"] is True
    configured = tmp_path / directory if directory else None
    if directory == "empty-directory":
        assert configured is not None
        configured.mkdir()
    monkeypatch.setenv("ASR_QUALITY_VAD_DIR", str(configured) if configured is not None else "")
    monkeypatch.setattr(importlib.metadata, "distributions", lambda **_kwargs: [])
    assert speech_evidence.detector_identity()["available"] is False
    assert speech_evidence.non_speech_evidence(detector.path, 0, 1) is None
    assert detector.imports == []


@pytest.mark.parametrize("failure", ["missing", "wrong_version", "wrong_hash", "missing_model"])
def test_unavailable_or_unverified_detector_fails_closed(detector: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    if failure == "missing":
        def unavailable(_name: str) -> Any:
            raise importlib.metadata.PackageNotFoundError("silero-vad")
        monkeypatch.setattr(importlib.metadata, "distribution", unavailable)
    elif failure == "wrong_version":
        detector.distribution.version = "6.2.2"
    elif failure == "wrong_hash":
        detector.model_path.write_bytes(b"unverified replacement")
    else:
        detector.model_path.unlink()
    assert speech_evidence.detector_identity()["available"] is False
    assert speech_evidence.non_speech_evidence(detector.path, 0, 1) is None
    assert detector.imports == []


@pytest.mark.parametrize("change", [{"threshold": 0.2}, {"available": False}, {"unknown": True}])
def test_frozen_identity_rejects_unknown_or_changed_configuration(detector: SimpleNamespace, change: dict[str, object]) -> None:
    identity = {**speech_evidence.detector_identity(), **change}
    assert speech_evidence.non_speech_evidence(detector.path, 0, 1, identity) is None
    assert detector.imports == []


@pytest.mark.parametrize("view,probability", [(0, 0.5), (0, 0.50001), (1, 0.5)])
def test_single_speech_frame_in_either_view_prevents_empty_delivery(detector: SimpleNamespace, view: int, probability: float) -> None:
    # One 32ms frame remains decisive; no 250ms duration filter may drop it.
    detector.model.outputs[(view, 2)] = probability
    assert speech_evidence.non_speech_evidence(detector.path, 0, 1) is None


def test_below_threshold_dual_scan_returns_evidence_without_probability_values(detector: SimpleNamespace) -> None:
    detector.model.outputs[(0, 0)] = 0.49999
    detector.model.outputs[(1, 0)] = 0.49999
    identity = speech_evidence.detector_identity()
    evidence = speech_evidence.non_speech_evidence(detector.path, 0, 1, identity)
    assert evidence is not None
    assert evidence["method"] == "silero_no_speech_detected"
    assert evidence["detector_identity"] == identity
    assert evidence["samples"] == 16000 and evidence["frames_per_view"] == 32 and evidence["views"] == 2
    assert evidence["gain"] == pytest.approx(0.9 * 32768 / 1000)
    cpu_ms = evidence["cpu_ms"]
    assert evidence["peak_pcm16"] == 1000 and isinstance(cpu_ms, (int, float)) and cpu_ms >= 0
    assert not any("prob" in key or "confidence" in key for key in evidence)
    assert detector.loads == [(str(detector.model_path), "cpu")]
    assert detector.imports == ["torch"]


@pytest.mark.parametrize("peak,accepted", [(294, False), (295, True), (32767, True), (0, False)])
def test_gain_cap_uses_required_gain_instead_of_clamping(detector: SimpleNamespace, peak: int, accepted: bool) -> None:
    # With target peak .9, adjacent PCM16 values bracket the 100x limit.
    write_wav(detector.path, array("h", [peak]) * 16000)
    evidence = speech_evidence.non_speech_evidence(detector.path, 0, 1)
    assert (evidence is not None) is accepted
    if not accepted:
        assert detector.imports == []
    else:
        assert evidence is not None
        gain = evidence["gain"]
        assert isinstance(gain, (int, float)) and gain <= 100


@pytest.mark.parametrize("output", [float("nan"), float("inf"), -0.001, 1.001, RuntimeError("VAD failed")])
def test_invalid_or_failed_model_output_never_authorizes_empty_audio(detector: SimpleNamespace, output: float | Exception) -> None:
    detector.model.outputs[(0, 0)] = output
    assert speech_evidence.non_speech_evidence(detector.path, 0, 1) is None


def test_last_partial_frame_is_padded_without_dropping_samples_and_views_reset(detector: SimpleNamespace) -> None:
    samples = array("h", [1000]) * 16001
    samples[-1] = -1000
    write_wav(detector.path, samples)
    evidence = speech_evidence.non_speech_evidence(detector.path, 0, 16001 / 16000)
    assert evidence is not None and evidence["samples"] == 16001
    assert len(detector.model.calls) == 64
    assert [call[1] for call in detector.model.calls] == list(range(32)) * 2
    original_last, amplified_last = detector.model.calls[31][2], detector.model.calls[63][2]
    assert original_last[128] == pytest.approx(-1000 / 32768)
    assert amplified_last[128] == pytest.approx(-0.9)
    assert original_last[129:] == amplified_last[129:] == [0] * 383
    assert detector.model.view == 1
    assert speech_evidence.non_speech_evidence(detector.path, 0, 16001 / 16000) is not None
    assert detector.model.view == 3 and len(detector.loads) == 1


@pytest.mark.parametrize("rate,channels,width", [(8000, 1, 2), (16000, 2, 2), (16000, 1, 1)])
def test_only_normalized_mono_pcm16_16khz_is_accepted(detector: SimpleNamespace, rate: int, channels: int, width: int) -> None:
    write_wav(detector.path, array("h", [1000]) * 32000, rate=rate, channels=channels, width=width)
    assert speech_evidence.non_speech_evidence(detector.path, 0, 1) is None
    assert detector.imports == []


@pytest.mark.parametrize("start,end", [(-1, 1), (1, 1), (1, 0), (0, 1.1), (float("nan"), 1), (0, float("inf"))])
def test_invalid_and_out_of_source_intervals_fail_closed(detector: SimpleNamespace, start: float, end: float) -> None:
    assert speech_evidence.non_speech_evidence(detector.path, start, end) is None
    assert detector.imports == []


def test_long_or_truncated_input_is_not_partially_scanned(detector: SimpleNamespace) -> None:
    write_wav(detector.path, array("h", [1000]) * (181 * 16000))
    assert speech_evidence.non_speech_evidence(detector.path, 0, 181) is None
    write_wav(detector.path, array("h", [1000]) * 16000)
    detector.path.write_bytes(detector.path.read_bytes()[:-2])
    assert speech_evidence.non_speech_evidence(detector.path, 0, 1) is None
    assert detector.imports == []


def test_missing_torch_fails_closed(detector: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(_name: str) -> Any:
        raise ModuleNotFoundError("torch")
    monkeypatch.setattr(importlib, "import_module", unavailable)
    assert speech_evidence.detector_identity()["available"] is True
    assert speech_evidence.non_speech_evidence(detector.path, 0, 1) is None


def test_failed_cpu_model_load_fails_closed(detector: SimpleNamespace) -> None:
    def failed_load(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("invalid JIT model")
    detector.torch.jit.load = failed_load
    assert speech_evidence.non_speech_evidence(detector.path, 0, 1) is None


def test_deadline_discards_partial_scan_and_releases_stateful_model_lock(detector: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(speech_evidence, "perf_counter", lambda: clock[0])
    original_call = Model.__call__

    def slow_call(model: Model, frame: Tensor, sample_rate: int) -> Any:
        result = original_call(model, frame, sample_rate)
        clock[0] = 60.0
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Model, "__call__", slow_call)
        assert speech_evidence.non_speech_evidence(detector.path, 0, 1) is None
    assert len(detector.model.calls) == 1
    # The next request resets the cached model, rather than reusing partial state.
    assert speech_evidence.non_speech_evidence(detector.path, 0, 1) is not None
    assert detector.model.view == 2


def test_artifact_replaced_during_scan_invalidates_negative_evidence(detector: SimpleNamespace) -> None:
    def changed_output() -> float:
        detector.model_path.write_bytes(b"changed during inference")
        return 0.01

    original_call = detector.model.__class__.__call__
    # Keep the full dual scan valid while replacing the verified dependency.
    def call(model: Model, frame: Tensor, sample_rate: int) -> Any:
        result = original_call(model, frame, sample_rate)
        if model.view == 1 and model.frame == 32:
            return SimpleNamespace(item=changed_output)
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Model, "__call__", call)
        assert speech_evidence.non_speech_evidence(detector.path, 0, 1) is None


@pytest.mark.parametrize("view", [0, 1])
def test_offline_inspection_reports_probability_hits_without_classifying_sound(detector: SimpleNamespace, view: int) -> None:
    detector.model.outputs[(view, 2)] = 0.8
    detector.model.outputs[(view, 3)] = 0.9
    report = speech_evidence.inspect_speech_evidence(detector.path, 0, 1)
    assert report["status"] == "completed" and report["reason"] == "speech_like_probability_hits"
    assert report["speech_presence"] == "model_candidates" and report["acoustic_content"] == "unclassified"
    assert report["negative_evidence_established"] is False and report["authorizes_delivery"] is False
    assert len(detector.model.calls) == 64  # Both views are complete despite the first threshold hit.
    assert report["views"][view]["above_threshold_frames"] == 2
    intervals = report["views"][view]["above_threshold_intervals"]
    assert len(intervals) == 1 and intervals[0]["frames"] == 2
    assert intervals[0]["start"] == 0.064 and intervals[0]["end"] == 0.128
    assert intervals[0]["max_probability"] == 0.9
    assert all(item["complete"] for item in report["views"])
    assert str(detector.path) not in str(report) and "waveform" not in report


def test_offline_negative_inspection_never_authorizes_delivery_or_labels_music(detector: SimpleNamespace) -> None:
    identity = speech_evidence.detector_identity()
    report = speech_evidence.inspect_speech_evidence(detector.path, 0, 1, identity)
    assert report["status"] == "completed" and report["reason"] == "dual_scan_below_threshold"
    assert report["negative_evidence_established"] is True
    assert report["speech_presence"] == "not_established" and report["acoustic_content"] == "unclassified"
    assert report["authorizes_delivery"] is False
    assert all(item["above_threshold_frames"] == 0 for item in report["views"])
    assert speech_evidence.detector_identity() == identity  # Production identity is unchanged.


@pytest.mark.parametrize("failure,expected", [("gain", "required_gain_exceeds_limit"), ("detector", "detector_unavailable_or_unverified"), ("identity", "detector_identity_mismatch"), ("runtime", "runtime_unavailable"), ("load", "detector_load_failed")])
def test_offline_inspection_separates_unavailable_evidence_from_probability_hits(detector: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str, expected: str) -> None:
    identity = None
    if failure == "gain":
        write_wav(detector.path, array("h", [100]) * 16000)
    elif failure == "detector":
        detector.model_path.write_bytes(b"unverified")
    elif failure == "identity":
        identity = {**speech_evidence.detector_identity(), "threshold": 0.9}
    elif failure == "runtime":
        def unavailable(_name: str) -> Any:
            raise ModuleNotFoundError("torch")
        monkeypatch.setattr(importlib, "import_module", unavailable)
    else:
        def failed_load(*_args: object, **_kwargs: object) -> Any:
            raise RuntimeError("unusable model")
        detector.torch.jit.load = failed_load
    report = speech_evidence.inspect_speech_evidence(detector.path, 0, 1, identity)
    assert report["status"] == "unavailable" and report["reason"] == expected
    assert report["speech_presence"] == "not_established" and report["negative_evidence_established"] is False
    assert report["views"] == [] and report["acoustic_content"] == "unclassified"


def test_offline_inspection_reports_partial_deadline_without_inventing_voice(detector: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(speech_evidence, "perf_counter", lambda: clock[0])
    original_call = Model.__call__
    def slow_call(model: Model, frame: Tensor, sample_rate: int) -> Any:
        result = original_call(model, frame, sample_rate)
        clock[0] = 60.0
        return result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Model, "__call__", slow_call)
        report = speech_evidence.inspect_speech_evidence(detector.path, 0, 1)
    assert report["status"] == "incomplete" and report["reason"] == "deadline_exceeded"
    assert report["negative_evidence_established"] is False and report["speech_presence"] == "not_established"
    assert report["views"][0]["complete"] is False
    assert speech_evidence.non_speech_evidence(detector.path, 0, 1) is not None  # Lock was released.


def test_offline_inspection_measures_digital_silence_without_detector(detector: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    write_wav(detector.path, array("h", [0]) * 16000)
    monkeypatch.setattr(speech_evidence, "_verified_model", lambda: None)
    report = speech_evidence.inspect_speech_evidence(detector.path, 0, 1)
    assert report["status"] == "completed" and report["reason"] == "digital_silence"
    assert report["acoustic_content"] == "digital_silence" and report["authorizes_delivery"] is False
    assert detector.imports == []


def test_offline_inspection_rejects_invalid_model_probability_as_failure(detector: SimpleNamespace) -> None:
    detector.model.outputs[(0, 0)] = float("nan")
    report = speech_evidence.inspect_speech_evidence(detector.path, 0, 1)
    assert report["status"] == "error" and report["reason"] == "model_probability_invalid"
    assert report["speech_presence"] == "not_established" and report["negative_evidence_established"] is False
