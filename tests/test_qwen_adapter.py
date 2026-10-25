from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from pathlib import Path

import pytest

from asr_server.adapters.base import AudioPath, TimestampItem
from asr_server.adapters.qwen import QwenAsrAdapter, _HfNativeQwenModel, _QwenWorkerBackend
from asr_server.errors import AsrError
from asr_server.registry import MODEL_SOURCES


class OomModel:
    def transcribe(self, **kwargs: object) -> list[object]:
        del kwargs
        raise RuntimeError("CUDA out of memory")


class EmptyModel:
    def transcribe(self, **kwargs: object) -> list[object]:
        del kwargs
        return []


class TextResult:
    text = "ok"
    language = "en"


class EmptyTextResult:
    text = ""
    language = "en"


class EmptyTextModel:
    def transcribe(self, **kwargs: object) -> list[EmptyTextResult]:
        del kwargs
        return [EmptyTextResult()]


class RecordingModel:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def transcribe(self, **kwargs: object) -> list[TextResult]:
        del kwargs
        self.calls.append("transcribe")
        return [TextResult()]


class FakeWorker:
    def is_alive(self) -> bool:
        return True


class FakeConnection:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    def send(self, payload: dict[str, object]) -> None:
        self.sent.append(payload)

    def recv(self) -> dict[str, object]:
        request = self.sent[-1]
        return {
            "id": request["id"],
            "ok": True,
            "result": {
                "text": "worker text",
                "duration": 1.25,
                "language": "en",
                "warnings": ["from_worker"],
                "timings": {
                    "total_ms": 10.0,
                    "load_ms": 0.0,
                    "decode_ms": 1.0,
                    "inference_ms": 8.0,
                    "postprocess_ms": 1.0,
                },
            },
        }


class FakeTransport:
    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn

    def request(self, op: str, *, timeout_seconds: float, **payload: object) -> object:
        del timeout_seconds
        self.conn.send({"id": len(self.conn.sent) + 1, "op": op, **payload})
        return self.conn.recv()["result"]


class CleanupModel:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class ModelWithInnerModule(CleanupModel):
    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self.model = SimpleNamespace(to=lambda device: calls.append(f"to:{device}"))


class FailingCleanupModel:
    def close(self) -> None:
        raise RuntimeError("close failed")


class FakeHfInputs(dict[str, Any]):
    def to(self, device: str, dtype: object) -> "FakeHfInputs":
        self["device"] = device
        self["dtype"] = dtype
        return self



@pytest.mark.parametrize("model_id", ["qwen3-asr-0.6b", "qwen3-asr-1.7b"])
async def test_qwen_load_uses_hf_native_transformers_without_qwen_asr(monkeypatch: pytest.MonkeyPatch, model_id: str) -> None:
    calls: list[tuple[str, object]] = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, repo_id: str, *, revision: str) -> "FakeProcessor":
            calls.append(("processor_repo", repo_id))
            calls.append(("processor_revision", revision))
            return cls()

        def apply_transcription_request(
            self,
            *,
            audio: str,
            language: str | None,
            prompt: str | None = None,
        ) -> FakeHfInputs:
            calls.append(("audio", audio))
            calls.append(("language", language))
            calls.append(("prompt", prompt))
            return FakeHfInputs(input_ids=SimpleNamespace(shape=[1, 2]))

        def apply_chat_template(self, messages: object, **kwargs: object) -> FakeHfInputs:
            calls.append(("messages", messages))
            calls.append(("chat_template_kwargs", kwargs))
            return FakeHfInputs(input_ids=SimpleNamespace(shape=[1, 2]))

        def decode(self, generated_ids: object, return_format: str) -> list[dict[str, str]]:
            del generated_ids
            calls.append(("return_format", return_format))
            return [{"language": "English", "transcription": "hello"}]

    class FakeModel:
        device = "cuda"
        dtype = "bf16"

        @classmethod
        def from_pretrained(cls, repo_id: str, *, revision: str, dtype: object) -> "FakeModel":
            calls.append(("model_repo", repo_id))
            calls.append(("model_revision", revision))
            calls.append(("dtype", dtype))
            return cls()

        def to(self, device: str) -> "FakeModel":
            calls.append(("device", device))
            return self

        def eval(self) -> "FakeModel":
            calls.append(("eval", True))
            return self

        def generate(self, **kwargs: object) -> object:
            calls.append(("generate", kwargs["max_new_tokens"]))

            class FakeOutput:
                def __getitem__(self, key: object) -> str:
                    calls.append(("slice", key))
                    return "generated"

            return FakeOutput()

    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    fake_transformers = SimpleNamespace(AutoProcessor=FakeProcessor, AutoModelForMultimodalLM=FakeModel)

    def fake_import_module(name: str) -> Any:
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        if name == "qwen_asr":
            raise AssertionError("HF native load must not import qwen_asr")
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("asr_server.adapters.qwen.importlib.import_module", fake_import_module)
    repo, revision = MODEL_SOURCES[model_id]
    adapter = _QwenWorkerBackend(model_id, repository=repo, revision=revision)

    await adapter.load("transformers", "cuda", "auto", max_new_tokens=123)
    result = await adapter.transcribe(
        b"audio",
        model_id=model_id,
        backend="transformers",
        language="auto",
        context="domain terms",
        max_new_tokens=77,
    )

    assert result.text == "hello"
    assert result.language == "English"
    assert result.warnings == ["qwen_auto_language_uses_chat_template_generation_prompt"]
    assert result.context.applied is True
    assert result.context.method == "qwen3_asr_system_prompt"
    messages = next(value for key, value in calls if key == "messages")
    assert isinstance(messages, list)
    assert messages[0] == {
        "role": "system",
        "content": [{"type": "text", "text": "domain terms"}],
    }
    assert "qwen_auto_language_uses_chat_template_generation_prompt" in result.warnings
    assert ("processor_repo", repo) in calls
    assert ("model_repo", repo) in calls
    assert ("processor_revision", revision) in calls
    assert ("model_revision", revision) in calls
    assert ("dtype", "bf16") in calls
    assert ("device", "cuda") in calls
    assert ("generate", 77) in calls
    assert not any(key == "audio" for key, _value in calls), "auto must keep using the generation-prompt path"
    template_kwargs = next(value for key, value in calls if key == "chat_template_kwargs")
    assert template_kwargs == {"add_generation_prompt": True, "tokenize": True, "return_dict": True}


@pytest.mark.parametrize("context", ["", "domain terms"])
@pytest.mark.parametrize(
    ("language", "decoded_language", "expected_language"),
    [("zh", None, "zh"), ("en", None, "en"), ("Chinese", None, "Chinese"), ("zh", "English", "English")],
)
async def test_explicit_language_preserves_native_request_and_alignment_language(
    monkeypatch: pytest.MonkeyPatch, language: str, decoded_language: str | None,
    expected_language: str, context: str,
) -> None:
    calls: dict[str, object] = {}

    class Processor:
        chat_template = "checkpoint template which does not render assistant turns"

        def apply_transcription_request(
            self, *, audio: str, language: str, prompt: str | None = None,
            chat_template: str | None = None,
        ) -> FakeHfInputs:
            calls.update(audio=audio, language=language, prompt=prompt, chat_template=chat_template)
            return FakeHfInputs(input_ids=SimpleNamespace(shape=[1, 2]))

        def apply_chat_template(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("explicit language must retain the native helper and its language normalization")

        def decode(self, _ids: object, *, return_format: str) -> list[dict[str, str | None]]:
            assert return_format == "parsed"
            return [{"language": decoded_language, "transcription": "hello"}]

    class Model:
        device = "cuda"
        dtype = "bf16"

        def generate(self, **kwargs: object) -> object:
            assert kwargs["max_new_tokens"] == 42

            class Output:
                def __getitem__(self, _key: object) -> str:
                    return "generated"

            return Output()

    processor = Processor()
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b", enable_alignment=True)
    adapter._model = _HfNativeQwenModel(processor=processor, model=Model(), torch=SimpleNamespace(), max_new_tokens=42)
    alignment_languages: list[str] = []

    def align(_audio: str, transcript: str, detected_language: str, *, requested_granularity: str) -> tuple[list[TimestampItem], list[str]]:
        assert transcript == "hello" and requested_granularity == "auto"
        alignment_languages.append(detected_language)
        return [TimestampItem("hello", 0, 1, "word")], []

    monkeypatch.setattr(adapter, "_align", align)
    result = await adapter.transcribe(b"audio", model_id="qwen3-asr-0.6b", backend="transformers",
                                      language=language, context=context, timestamps="auto")
    assert calls["language"] == language and calls["prompt"] == (context or None)
    assert isinstance(calls["chat_template"], str) and calls["chat_template"] != processor.chat_template
    assert result.text == "hello" and result.language == expected_language
    assert alignment_languages == [expected_language]
    assert result.timestamps == [TimestampItem("hello", 0, 1, "word")]


@pytest.mark.parametrize("original_renders_assistant", [False, True])
def test_prefill_template_keeps_context_audio_and_one_open_assistant_turn(original_renders_assistant: bool) -> None:
    # Transformers provides Jinja in WSL; keep the Mac base environment lightweight.
    jinja2 = pytest.importorskip("jinja2")
    from asr_server.adapters.qwen import _qwen_prefill_chat_template

    original_template = (
        "{%- for message in messages %}"
        "{%- if message.role in ['system', 'user'] or render_assistant %}"
        "{{ '<|im_start|>' + message.role + '\\n' }}"
        "{%- for part in message.content %}"
        "{%- if part.type == 'text' %}{{ part.text }}"
        "{%- elif part.type == 'audio' %}{{ '<|audio_start|><|audio_pad|><|audio_end|>' }}{% endif %}"
        "{%- endfor %}{{ '<|im_end|>\\n' }}"
        "{%- endif %}{%- endfor %}"
        "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "domain terms"}]},
        {"role": "user", "content": [{"type": "audio", "audio": "speech.wav"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "language Chinese<asr_text>"}]},
    ]
    processor = SimpleNamespace(chat_template=original_template)
    rendered = jinja2.Environment().from_string(_qwen_prefill_chat_template(processor)).render(
        messages=messages, add_generation_prompt=False, render_assistant=original_renders_assistant,
    )
    assert rendered == (
        "<|im_start|>system\ndomain terms<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\nlanguage Chinese<asr_text>"
    )


async def test_qwen_load_rejects_unsupported_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        float16="fp16",
        version=SimpleNamespace(cuda="12.8"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )

    def fake_import_module(name: str) -> Any:
        if name == "torch":
            return fake_torch
        raise AssertionError(f"unexpected import after dtype rejection: {name}")

    monkeypatch.setattr("asr_server.adapters.qwen.importlib.import_module", fake_import_module)
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")

    with pytest.raises(AsrError) as exc_info:
        await adapter.load("transformers", "cuda", "float32")

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "capability_not_supported"


async def test_qwen_transcribe_maps_cuda_oom() -> None:
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
    adapter._model = OomModel()

    with pytest.raises(AsrError) as exc_info:
        await adapter.transcribe(
            b"audio",
            model_id="qwen3-asr-0.6b",
            backend="transformers",
            language="auto",
            context="",
            max_new_tokens=None,
        )

    assert exc_info.value.code == "gpu_unavailable"
    assert exc_info.value.details["phase"] == "inference"


async def test_qwen_transcribe_rejects_empty_results() -> None:
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
    adapter._model = EmptyModel()

    with pytest.raises(AsrError) as exc_info:
        await adapter.transcribe(
            b"audio",
            model_id="qwen3-asr-0.6b",
            backend="transformers",
            language="auto",
            context="",
            max_new_tokens=None,
        )

    assert exc_info.value.code == "empty_transcription"


async def test_qwen_batch_distinguishes_empty_transcription_from_inference_failure() -> None:
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
    adapter._model = EmptyTextModel()

    with pytest.raises(AsrError) as exc_info:
        await adapter.transcribe_batch(
            [b"audio"],
            model_id="qwen3-asr-0.6b",
            backend="transformers",
            language="auto",
            context="",
            max_new_tokens=None,
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "empty_transcription"


async def test_qwen_transcribe_uses_inference_mode_when_torch_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeInferenceMode:
        def __enter__(self) -> None:
            calls.append("enter_inference_mode")

        def __exit__(self, *args: object) -> None:
            calls.append("exit_inference_mode")

    fake_torch = SimpleNamespace(inference_mode=lambda: FakeInferenceMode())

    def fake_import_module(name: str) -> Any:
        assert name == "torch"
        return fake_torch

    monkeypatch.setattr("asr_server.adapters.qwen.importlib.import_module", fake_import_module)
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
    adapter._model = RecordingModel(calls)

    result = await adapter.transcribe(
        b"audio",
        model_id="qwen3-asr-0.6b",
        backend="transformers",
        language="auto",
        context="",
        max_new_tokens=None,
    )

    assert result.text == "ok"
    assert calls == ["enter_inference_mode", "transcribe", "exit_inference_mode"]


async def test_qwen_transcribe_falls_back_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_import_module(name: str) -> Any:
        assert name == "torch"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("asr_server.adapters.qwen.importlib.import_module", fake_import_module)
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
    adapter._model = RecordingModel(calls)

    result = await adapter.transcribe(
        b"audio",
        model_id="qwen3-asr-0.6b",
        backend="transformers",
        language="auto",
        context="",
        max_new_tokens=None,
    )

    assert result.text == "ok"
    assert calls == ["transcribe"]


async def test_qwen_adapter_transcribe_uses_worker_protocol() -> None:
    adapter = QwenAsrAdapter("qwen3-asr-0.6b")
    conn = FakeConnection()
    adapter._transport = FakeTransport(conn)  # type: ignore[assignment]

    result = await adapter.transcribe(
        b"audio",
        model_id="qwen3-asr-0.6b",
        backend="transformers",
        language="auto",
        context="ctx",
        max_new_tokens=128,
    )

    assert result.text == "worker text"
    assert result.timings.inference_ms == 8.0
    assert conn.sent == [
        {
            "id": 1,
            "op": "transcribe",
            "audio": b"audio",
            "language": "auto",
            "context": "ctx",
            "timestamps": "none",
            "max_new_tokens": 128,
        }
    ]


async def test_qwen_worker_protocol_sends_path_descriptor_not_audio_bytes(tmp_path: Path) -> None:
    audio_path = tmp_path / "normalized.wav"
    audio_path.write_bytes(b"wav")
    adapter = QwenAsrAdapter("qwen3-asr-0.6b")
    conn = FakeConnection()
    adapter._transport = FakeTransport(conn)  # type: ignore[assignment]

    await adapter.transcribe(
        AudioPath(audio_path, start=1.0, end=2.0),
        model_id="qwen3-asr-0.6b",
        backend="transformers",
        language="auto",
        context="",
        max_new_tokens=128,
    )

    assert isinstance(conn.sent[0]["audio"], AudioPath)


@pytest.mark.parametrize("failure", [None, "model_close", "cuda_synchronize"])
async def test_qwen_unload_releases_references_and_cache_despite_local_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, failure: str | None,
) -> None:
    calls: list[str] = []
    model = ModelWithInnerModule(calls) if failure != "model_close" else FailingCleanupModel()
    aligner = CleanupModel()
    adapter = _QwenWorkerBackend("qwen3-asr-0.6b")
    adapter._model, adapter._aligner_model = model, aligner
    adapter.loaded_backend = "transformers"

    def synchronize() -> None:
        calls.append("synchronize")
        if failure == "cuda_synchronize":
            raise RuntimeError("sync failed")
    fake_cuda = SimpleNamespace(is_available=lambda: True, synchronize=synchronize,
                                empty_cache=lambda: calls.append("empty_cache"),
                                ipc_collect=lambda: calls.append("ipc_collect"))
    def fake_import(name: str) -> Any:
        assert name == "torch"
        return SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setattr("asr_server.adapters.qwen.importlib.import_module", fake_import)
    await adapter.unload(cuda_empty_cache=True)

    assert all(getattr(adapter, name) is None for name in ("_model", "_aligner_model", "_aligner_processor"))
    assert getattr(adapter, "loaded_backend") is None
    assert aligner.closed
    assert "empty_cache" in calls and "ipc_collect" in calls
    assert calls.index("synchronize") < calls.index("empty_cache")
    # CPU transfer must precede cache release; unrelated helper order is free.
    if isinstance(model, ModelWithInnerModule):
        assert model.closed
        assert calls.index("to:cpu") < calls.index("empty_cache")
