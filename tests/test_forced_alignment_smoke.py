from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import qwen_forced_alignment_smoke as smoke


@pytest.mark.parametrize(("requested", "decoded", "expected"), [("zh", None, "zh"), ("auto", "Chinese", "Chinese")])
def test_forced_alignment_smoke_preserves_native_language_prefill_and_alignment_language(
    monkeypatch: pytest.MonkeyPatch, requested: str, decoded: str | None, expected: str,
) -> None:
    calls: dict[str, Any] = {}
    class Inputs(dict[str, Any]):
        def to(self, device: str, dtype: object) -> Inputs:
            return self
    def inputs() -> Inputs:
        return Inputs(input_ids=SimpleNamespace(shape=[1, 2]))
    class Processor:
        def apply_transcription_request(self, **options: Any) -> Inputs:
            calls["native_request"] = options
            return inputs()
        def apply_chat_template(self, messages: object, **options: Any) -> Inputs:
            calls["automatic_template"] = {"messages": messages, **options}
            return inputs()
        def decode(self, tokens: object, **options: Any) -> list[dict[str, object]]:
            return [{"transcription": "你好", "language": decoded}]
    processor = Processor()
    class AlignerProcessor:
        def prepare_forced_aligner_inputs(self, **options: Any) -> tuple[Inputs, list[object]]:
            calls["alignment"] = options
            return inputs(), []
        def decode_forced_alignment(self, **options: Any) -> list[list[dict[str, object]]]:
            return [[{"text": "你好", "start_time": 0.0, "end_time": 0.2}]]
    class Processors:
        @classmethod
        def from_pretrained(cls, repository: str, **options: Any) -> Processor | AlignerProcessor:
            return AlignerProcessor() if repository == smoke.ALIGNER_MODEL else processor
    class Tokens:
        def __getitem__(self, item: object) -> str:
            return "generated"
    class Model:
        device = "cuda"
        dtype = "bf16"
        config = SimpleNamespace(timestamp_token_id=1)
        @classmethod
        def from_pretrained(cls, repository: str, **options: Any) -> Model:
            return cls()
        def to(self, device: str) -> Model:
            return self
        def eval(self) -> Model:
            return self
        def generate(self, **options: Any) -> Tokens:
            return Tokens()
        def __call__(self, **options: Any) -> SimpleNamespace:
            return SimpleNamespace(logits="logits")
    torch = SimpleNamespace(bfloat16="bf16", float16="fp16", inference_mode=nullcontext,
                            cuda=SimpleNamespace(reset_peak_memory_stats=lambda device: None,
                                                 max_memory_allocated=lambda device: 0,
                                                 max_memory_reserved=lambda device: 0))
    transformers = SimpleNamespace(AutoProcessor=Processors, AutoModelForMultimodalLM=Model,
                                   AutoModelForTokenClassification=Model)
    monkeypatch.setattr(smoke, "require_cuda_torch", lambda: torch)
    def import_module(name: str) -> object:
        assert name == "transformers"
        return transformers
    monkeypatch.setattr("scripts.qwen_forced_alignment_smoke.importlib.import_module", import_module)
    def prefill(value: object) -> str:
        assert value is processor
        calls["prefill"] = True
        return "shared-language-prefill"
    monkeypatch.setattr(smoke, "_qwen_prefill_chat_template", prefill)
    args = argparse.Namespace(asr_model="Qwen/Qwen3-ASR-0.6B-hf", asr_revision=None,
                              audio=Path("fixture.wav"), language=requested, prompt="domain terms",
                              device="cuda", dtype="auto", max_new_tokens=32)
    smoke.run(args)
    if requested == "zh":
        assert calls["native_request"] == {"audio": "fixture.wav", "language": "zh", "prompt": "domain terms",
                                           "chat_template": "shared-language-prefill"}
        assert calls["prefill"] and "automatic_template" not in calls
    else:
        assert calls["automatic_template"]["add_generation_prompt"] is True
        assert "native_request" not in calls and "prefill" not in calls
    assert calls["alignment"] == {"audio": "fixture.wav", "transcript": "你好", "language": expected}
