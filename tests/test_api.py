from __future__ import annotations

import asyncio
import io
import wave
from array import array
from collections.abc import AsyncIterator
from dataclasses import replace

import pytest
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from httpx import ASGITransport, AsyncClient

from asr_server import main as main_module
from asr_server.adapters.base import AudioComposition, AudioInput, TranscriptionResult, TranscriptionSegment
from asr_server.adapters.mock import MockAsrAdapter
from asr_server.audio import splitter as splitter_module
from asr_server.config import Settings
from asr_server.execution import MOSS_EXECUTION_POLICY
from asr_server.errors import validation_error_handler
from asr_server.main import create_app
from tests.helpers import make_wav


pytestmark = pytest.mark.integration


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        yield client


async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_validation_error_drops_input_context_and_free_text_message() -> None:
    marker = "private-validation-marker"
    error = RequestValidationError([{
        "type": "value_error", "loc": ("body", "text"),
        "msg": f"Value error, {marker}", "input": marker,
        "ctx": {"error": ValueError(marker)},
    }])
    response = await validation_error_handler(Request({"type": "http"}), error)

    assert response.status_code == 400
    assert marker.encode() not in response.body
    assert b'"code":"bad_request"' in response.body
    assert b'"type":"value_error"' in response.body
    assert b'"loc":["body","text"]' in response.body
    assert b'"msg":"Invalid value"' in response.body
    assert b'"input"' not in response.body and b'"ctx"' not in response.body


async def test_documentation_routes_are_accessible_and_openapi_is_parseable(client: AsyncClient) -> None:
    docs = await client.get("/docs")
    openapi = await client.get("/openapi.json")
    tutorial = await client.get("/docs/mcp")
    assert docs.status_code == openapi.status_code == tutorial.status_code == 200
    assert docs.headers["content-type"].startswith("text/html")
    assert openapi.headers["content-type"].startswith("application/json")
    assert tutorial.headers["content-type"].startswith("text/markdown")
    schema = openapi.json()
    assert schema["openapi"].startswith("3.")
    assert "/v1/audio/transcriptions" in schema["paths"]
    assert tutorial.text.strip()


@pytest.mark.parametrize("enable_moss", [False, True])
async def test_model_listing_matches_enabled_registry(enable_moss: bool) -> None:
    app = create_app(settings=Settings(enable_moss=enable_moss))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.get("/v1/models")
    assert response.status_code == 200
    models = response.json()["models"]
    definitions = {model_id: runtime.definition for model_id, runtime in app.state.manager._runtimes.items()}
    assert {model["id"] for model in models} == set(definitions)
    for model in models:
        definition = definitions[model["id"]]
        assert model["capabilities"] == definition.capabilities.to_api()
        assert model["revision"] == definition.revision
        assert model["default"] == definition.default


async def test_disabled_moss_model_returns_model_not_found(client: AsyncClient) -> None:
    response = await client.get("/v1/models/moss-transcribe-diarize-0.9b/status")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


async def test_moss_load_status_and_unload_when_enabled() -> None:
    app = create_app(settings=Settings(enable_moss=True))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        load_response = await client.post(
            "/v1/models/moss-transcribe-diarize-0.9b/load",
            json={"backend": "transformers", "device": "cuda", "dtype": "auto"},
        )
        status_response = await client.get("/v1/models/moss-transcribe-diarize-0.9b/status")
        unload_response = await client.request(
            "DELETE",
            "/v1/models/moss-transcribe-diarize-0.9b",
            json={"mode": "after_current_requests", "reject_new_requests": True, "cuda_empty_cache": True},
        )

    assert load_response.status_code == 200
    assert load_response.json()["status"] == "loaded"
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "loaded"
    assert status_response.json()["backend"] == "transformers"
    assert status_response.json()["device"] == "cuda"
    assert status_response.json()["dtype"] == "auto"
    assert unload_response.status_code == 200
    assert unload_response.json()["status"] == "unloaded"


async def test_unknown_model_uses_error_envelope(client: AsyncClient) -> None:
    response = await client.get("/v1/models/missing/status")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


async def test_load_and_unload_model(client: AsyncClient) -> None:
    load_response = await client.post(
        "/v1/models/qwen3-asr-1.7b/load",
        json={"backend": "transformers", "device": "cuda", "dtype": "auto"},
    )
    assert load_response.status_code == 200
    assert load_response.json()["status"] == "loaded"

    status_response = await client.get("/v1/models/qwen3-asr-1.7b/status")
    assert status_response.json()["status"] == "loaded"
    assert status_response.json()["backend"] == "transformers"
    assert status_response.json()["device"] == "cuda"
    assert status_response.json()["dtype"] == "auto"

    unload_response = await client.request(
        "DELETE",
        "/v1/models/qwen3-asr-1.7b",
        json={"mode": "after_current_requests", "reject_new_requests": True, "cuda_empty_cache": True},
    )
    assert unload_response.status_code == 200
    assert unload_response.json()["status"] == "unloaded"



async def test_transcription_auto_loads_default_model(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
        data={"language": "auto", "backend": "transformers"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "qwen3-asr-1.7b"
    assert body["backend"] == "transformers"
    assert body["text"]
    assert body["timings"]["total_ms"] >= 0
    assert body["timings"]["decode_ms"] >= 0
    assert body["timings"]["load_ms"] >= 0
    assert body["timings"]["inference_ms"] >= 0
    assert body["usage"]["audio_seconds"] == body["duration"]
    assert not any(warning.startswith("context_received:") for warning in body["warnings"])


async def test_transcription_passes_context_and_hotwords_to_adapter(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
        data={
            "language": "auto",
            "backend": "transformers",
            "context": "专有名词：Qwen3-ASR 和 Silero VAD",
            "hotwords": '["Hugging Face", "RTX 5070 Ti", "uv"]',
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["text"]
    assert any(warning.startswith("context_received:") for warning in body["warnings"])
    assert "Qwen3-ASR" not in body["text"]


async def test_transcription_truncates_context_over_hard_limit(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
        data={"context": "x" * 4001},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["context"]["requested"] is True
    assert body["context"]["applied"] is True
    assert body["context"]["truncated"] is True
    assert "context_truncated:body" in body["warnings"]


async def test_text_response_format_still_supports_context(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
        data={
            "response_format": "text",
            "context": "Qwen3-ASR",
            "hotwords": "Silero VAD, Hugging Face",
        },
    )

    assert response.status_code == 200
    assert response.text
    assert response.headers["content-type"].startswith("text/plain")


async def test_transcription_passes_max_new_tokens_to_adapter(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
        data={"max_new_tokens": "256"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "max_new_tokens_received:256" in body["warnings"]
    assert "max_new_tokens_override:256" in body["warnings"]


async def test_transcription_uses_default_max_new_tokens_without_override_warning(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "max_new_tokens_received:512" in body["warnings"]
    assert "max_new_tokens_override:512" not in body["warnings"]


async def test_transcription_rejects_max_new_tokens_over_server_limit(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
        data={"max_new_tokens": "4097"},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "bad_request"
    assert body["error"]["details"]["max_new_tokens_limit"] == 4096


async def test_moss_accepts_its_model_specific_max_new_tokens_limit() -> None:
    app = create_app(settings=Settings(enable_moss=True))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
            data={
                "model": "moss-transcribe-diarize-0.9b",
                "response_format": "verbose_json",
                "max_new_tokens": "65536",
            },
        )

    assert response.status_code == 200
    assert "max_new_tokens_received:65536" in response.json()["warnings"]


async def test_transcription_reports_zero_load_time_when_model_is_already_loaded(client: AsyncClient) -> None:
    load_response = await client.post(
        "/v1/models/qwen3-asr-1.7b/load",
        json={"backend": "transformers", "device": "cuda", "dtype": "auto"},
    )
    assert load_response.status_code == 200

    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
        data={"model": "qwen3-asr-1.7b", "language": "auto", "backend": "transformers"},
    )

    assert response.status_code == 200
    timings = response.json()["timings"]
    assert timings["total_ms"] >= 0
    assert timings["load_ms"] == 0


async def test_transcription_supports_each_declared_backend(client: AsyncClient) -> None:
    for model in ("qwen3-asr-0.6b", "qwen3-asr-1.7b"):
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
            data={"model": model, "backend": "transformers"},
        )

        assert response.status_code == 200
        assert response.json()["model"] == model
        assert response.json()["backend"] == "transformers"
        assert response.json()["text"]


async def test_moss_verbose_json_returns_segments_when_enabled() -> None:
    app = create_app(settings=Settings(enable_moss=True))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
            data={
                "model": "moss-transcribe-diarize-0.9b",
                "backend": "transformers",
                "response_format": "verbose_json",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "moss-transcribe-diarize-0.9b"
    assert body["backend"] == "transformers"
    assert "max_new_tokens_received:2048" in body["warnings"]
    assert body["execution"] == {
        "mode": "native_long_form",
        "requested_split_strategy": "auto",
        "resolved_split_strategy": "none",
        "speaker_scope": "global",
        "automatic_chunk_fallback": False,
    }
    assert body["generation"]["max_new_tokens"] == 2_048
    assert body["generation"]["invocation_count"] == 1
    assert body["generation"]["truncated"] is False
    assert len(body["segments"]) == 1
    assert body["segments"][0]["start"] == 0.0
    assert body["segments"][0]["end"] > 0.0
    assert body["segments"][0]["speaker"] == "S01"
    assert body["segments"][0]["speaker_label"] == "S01"
    assert body["segments"][0]["speaker_scope"] == "global"
    assert body["segments"][0]["text"] == body["text"]


async def test_moss_json_response_keeps_segments_empty_by_default() -> None:
    app = create_app(settings=Settings(enable_moss=True))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
            data={
                "model": "moss-transcribe-diarize-0.9b",
                "backend": "transformers",
                "response_format": "json",
            },
        )

    assert response.status_code == 200
    assert response.json()["segments"] == []


async def test_moss_text_response_returns_plain_text_when_enabled() -> None:
    app = create_app(settings=Settings(enable_moss=True))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
            data={
                "model": "moss-transcribe-diarize-0.9b",
                "backend": "transformers",
                "response_format": "text",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text


async def test_moss_split_transcription_warns_speaker_labels_are_chunk_local() -> None:
    app = create_app(settings=Settings(enable_moss=True))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.11), "audio/wav")},
            data={
                "model": "moss-transcribe-diarize-0.9b",
                "backend": "transformers",
                "response_format": "verbose_json",
                "split_strategy": "fixed",
                "max_chunk_seconds": "0.04",
                "overlap_seconds": "0.01",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["split"]["chunk_count"] == 4
    assert body["execution"]["mode"] == "chunked"
    assert body["execution"]["speaker_scope"] == "chunk"
    assert "moss_speaker_labels_are_chunk_local" in body["warnings"]
    assert len(body["segments"]) == 4
    assert body["segments"][0]["start"] == pytest.approx(0.0)
    assert body["segments"][1]["start"] == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("requested_max_new_tokens", "expected_max_new_tokens"),
    [(None, 24_000), (8_000, 8_000)],
)
async def test_moss_anchor_replay_returns_four_global_speakers(
    requested_max_new_tokens: int | None,
    expected_max_new_tokens: int,
) -> None:
    class FourSpeakerAdapter(MockAsrAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.max_new_tokens_values: list[int | None] = []

        async def transcribe(
            self,
            audio: AudioInput,
            *,
            model_id: str,
            backend: str,
            language: str,
            context: str,
            timestamps: str = "none",
            max_new_tokens: int | None = None,
        ) -> TranscriptionResult:
            del model_id, backend, language, context, timestamps
            self.calls += 1
            self.max_new_tokens_values.append(max_new_tokens)
            if self.calls == 1:
                return TranscriptionResult(
                    text="甲乙丙丁",
                    duration=10.0,
                    language="zh",
                    warnings=[],
                    segments=[
                        TranscriptionSegment(0.0, 2.5, "甲", "S01"),
                        TranscriptionSegment(2.5, 5.0, "乙", "S02"),
                        TranscriptionSegment(5.0, 7.5, "丙", "S03"),
                        TranscriptionSegment(7.5, 10.0, "丁", "S04"),
                    ],
                )
            assert isinstance(audio, AudioComposition)
            prefix = audio.prefix_duration
            return TranscriptionResult(
                text="锚点和正文",
                duration=prefix + 10.0,
                language="zh",
                warnings=[],
                segments=[
                    TranscriptionSegment(0.0, 2.5, "甲锚", "D"),
                    TranscriptionSegment(3.0, 5.5, "乙锚", "B"),
                    TranscriptionSegment(6.0, 8.5, "丙锚", "A"),
                    TranscriptionSegment(9.0, 11.5, "丁锚", "C"),
                    TranscriptionSegment(prefix + 0.5, prefix + 3.0, "丁继续", "C"),
                    TranscriptionSegment(prefix + 3.5, prefix + 6.0, "甲继续", "D"),
                ],
            )

    app = create_app(settings=Settings(enable_moss=True))
    adapter = FourSpeakerAdapter()
    app.state.manager.runtime_for("moss-transcribe-diarize-0.9b").adapter = adapter
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        data = {
            "model": "moss-transcribe-diarize-0.9b",
            "response_format": "verbose_json",
            "split_strategy": "fixed",
            "max_chunk_seconds": "10",
            "overlap_seconds": "0",
            "speaker_resolution": "auto",
        }
        if requested_max_new_tokens is not None:
            data["max_new_tokens"] = str(requested_max_new_tokens)
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(20.0), "audio/wav")},
            data=data,
        )

    assert response.status_code == 200
    body = response.json()
    assert body["diarization"] == {
        "method": "moss_anchor_replay",
        "status": "complete",
        "speaker_scope": "global",
        "speaker_count": 4,
        "unresolved_segments": 0,
        "conflicts": 0,
        "anchor_budget_limited": False,
        "candidate_speakers": 0,
        "missing_segment_chunks": 0,
        "max_speakers": 4,
        "speaker_capacity_exceeded": 0,
    }
    assert body["execution"]["speaker_scope"] == "global"
    assert body["segments"][-2]["speaker"] == "speaker-0004"
    assert body["segments"][-2]["source_speaker"] == "chunk-0001:C"
    assert body["segments"][-1]["speaker"] == "speaker-0001"
    assert body["generation"]["max_new_tokens"] == expected_max_new_tokens
    assert adapter.max_new_tokens_values == [expected_max_new_tokens, expected_max_new_tokens]
    assert "moss_speaker_labels_are_chunk_local" not in body["warnings"]


async def test_speaker_resolution_rejects_invalid_mode_and_non_diarization_model() -> None:
    app = create_app(settings=Settings(enable_moss=True))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        invalid = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
            data={"model": "moss-transcribe-diarize-0.9b", "speaker_resolution": "guess"},
        )
        unsupported = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
            data={"model": "qwen3-asr-1.7b", "speaker_resolution": "auto"},
        )
        hidden_segments = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
            data={"model": "moss-transcribe-diarize-0.9b", "speaker_resolution": "auto"},
        )

    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "bad_request"
    assert unsupported.status_code == 422
    assert unsupported.json()["error"]["code"] == "capability_not_supported"
    assert hidden_segments.status_code == 422
    assert hidden_segments.json()["error"]["details"]["recommended_response_format"] == "verbose_json"


async def test_required_speaker_resolution_rejects_an_anchor_identity_conflict() -> None:
    class ConflictingAnchorAdapter(MockAsrAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def transcribe(
            self,
            audio: AudioInput,
            *,
            model_id: str,
            backend: str,
            language: str,
            context: str,
            timestamps: str = "none",
            max_new_tokens: int | None = None,
        ) -> TranscriptionResult:
            del model_id, backend, language, context, timestamps, max_new_tokens
            self.calls += 1
            if self.calls == 1:
                return TranscriptionResult(
                    text="甲乙",
                    duration=10.0,
                    language="zh",
                    warnings=[],
                    segments=[
                        TranscriptionSegment(0.0, 3.0, "甲", "S01"),
                        TranscriptionSegment(4.0, 7.0, "乙", "S02"),
                    ],
                )
            assert isinstance(audio, AudioComposition)
            prefix = audio.prefix_duration
            return TranscriptionResult(
                text="冲突",
                duration=prefix + 10.0,
                language="zh",
                warnings=[],
                segments=[
                    TranscriptionSegment(0.0, 3.0, "甲锚", "SAME"),
                    TranscriptionSegment(3.5, 6.5, "乙锚", "SAME"),
                    TranscriptionSegment(prefix + 1.0, prefix + 3.0, "正文", "SAME"),
                ],
            )

    app = create_app(settings=Settings(enable_moss=True))
    app.state.manager.runtime_for("moss-transcribe-diarize-0.9b").adapter = ConflictingAnchorAdapter()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(20.0), "audio/wav")},
            data={
                "model": "moss-transcribe-diarize-0.9b",
                "response_format": "verbose_json",
                "split_strategy": "fixed",
                "max_chunk_seconds": "10",
                "overlap_seconds": "0",
                "speaker_resolution": "required",
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "speaker_resolution_incomplete"
    assert response.json()["error"]["details"]["conflicts"] == 1


async def test_required_native_resolution_rejects_text_without_speaker_segments() -> None:
    class MissingSegmentsAdapter(MockAsrAdapter):
        async def transcribe(
            self,
            audio: AudioInput,
            *,
            model_id: str,
            backend: str,
            language: str,
            context: str,
            timestamps: str = "none",
            max_new_tokens: int | None = None,
        ) -> TranscriptionResult:
            del audio, model_id, backend, language, context, timestamps, max_new_tokens
            return TranscriptionResult(text="有正文但解析不到说话人段", duration=1.0, language="zh", warnings=[])

    app = create_app(settings=Settings(enable_moss=True))
    app.state.manager.runtime_for("moss-transcribe-diarize-0.9b").adapter = MissingSegmentsAdapter()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(1.0), "audio/wav")},
            data={
                "model": "moss-transcribe-diarize-0.9b",
                "response_format": "verbose_json",
                "speaker_resolution": "required",
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "speaker_resolution_incomplete"
    assert response.json()["error"]["details"]["missing_segment_chunks"] == 1


async def test_moss_native_long_form_rejects_silently_ignored_chunk_options() -> None:
    app = create_app(settings=Settings(enable_moss=True))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
            data={
                "model": "moss-transcribe-diarize-0.9b",
                "split_strategy": "auto",
                "max_chunk_seconds": "120",
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "capability_not_supported"
    assert response.json()["error"]["details"]["recommended_split_strategy"] == "fixed"


async def test_moss_auto_fallback_is_explicit_and_chunk_scoped() -> None:
    app = create_app(settings=Settings(enable_moss=True))
    runtime = app.state.manager.runtime_for("moss-transcribe-diarize-0.9b")
    runtime.definition = replace(
        runtime.definition,
        execution_policy=replace(
            MOSS_EXECUTION_POLICY,
            validated_native_max_seconds=0.05,
            fallback_chunk_seconds=0.04,
            chunk_hard_seconds=0.06,
        ),
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.11), "audio/wav")},
            data={"model": "moss-transcribe-diarize-0.9b", "response_format": "verbose_json"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["execution"]["mode"] == "chunked"
    assert body["execution"]["automatic_chunk_fallback"] is True
    assert body["execution"]["fallback_reason"] == "duration_exceeds_validated_native_limit"
    assert body["execution"]["speaker_scope"] == "chunk"
    assert body["split"]["strategy"] == "fixed"
    assert body["split"]["chunk_count"] == 3
    assert "moss_native_long_form_fallback:duration_exceeds_validated_native_limit" in body["warnings"]
    assert "moss_speaker_labels_are_chunk_local" in body["warnings"]



async def test_transcription_can_return_chunk_metadata(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.11), "audio/wav")},
        data={
            "model": "qwen3-asr-1.7b",
            "backend": "transformers",
            "split_strategy": "fixed",
            "max_chunk_seconds": "0.04",
            "overlap_seconds": "0.01",
            "preserve_segments": "true",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["split"]["strategy"] == "fixed"
    assert body["split"]["chunk_count"] == 4
    assert len(body["chunks"]) == 4
    assert body["chunks"][0]["index"] == 0
    assert body["chunks"][1]["start"] == pytest.approx(0.03)
    assert body["chunks"][1]["raw_text"]
    assert body["chunks"][1]["deduped_prefix_chars"] >= 0
    assert body["chunks"][1]["timestamp_source"] == "vad_chunk_window"
    assert body["duration"] == pytest.approx(0.11)
    assert body["text"]
    assert "mock_batch_adapter" in body["warnings"]


async def test_batch_result_count_mismatch_returns_inference_failed(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def bad_batch(
        self: MockAsrAdapter,
        audio_chunks: list[bytes],
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        timestamps: str = "none",
        max_new_tokens: int | None = None,
    ) -> list[TranscriptionResult]:
        del self, audio_chunks, model_id, backend, language, context, timestamps, max_new_tokens
        return []

    monkeypatch.setattr(MockAsrAdapter, "transcribe_batch", bad_batch)

    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.11), "audio/wav")},
        data={
            "model": "qwen3-asr-1.7b",
            "backend": "transformers",
            "split_strategy": "fixed",
            "max_chunk_seconds": "0.04",
            "overlap_seconds": "0.01",
        },
    )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "inference_failed"
    assert body["error"]["details"]["expected"] == 1


async def test_transcription_uses_energy_fallback_when_silero_is_unavailable(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: object, **kwargs: object) -> list[tuple[float, float]]:
        raise splitter_module._SileroUnavailable("test unavailable")

    monkeypatch.setattr(splitter_module, "_silero_intervals", unavailable)

    samples = array("h")
    samples.extend([0] * int(0.2 * 16_000))
    samples.extend([10_000] * int(0.4 * 16_000))
    samples.extend([0] * int(0.5 * 16_000))
    samples.extend([10_000] * int(0.4 * 16_000))
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(samples.tobytes())

    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", audio.getvalue(), "audio/wav")},
        data={
            "model": "qwen3-asr-1.7b",
            "backend": "transformers",
            "max_chunk_seconds": "0.8",
            "overlap_seconds": "0.03",
            "preserve_segments": "true",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["split"]["strategy"] == "energy"
    assert body["split"]["requested_strategy"] == "auto"
    assert body["split"]["vad_backend"] == "energy"
    assert body["split"]["chunk_count"] == 2
    assert "silero_streaming_not_validated_fallback_to_energy" in body["warnings"]
    assert len(body["chunks"]) == 2
    assert body["duration"] == pytest.approx(1.5)


async def test_transcription_rejects_invalid_split_overlap(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", make_wav(0.11), "audio/wav")},
        data={
            "split_strategy": "fixed",
            "max_chunk_seconds": "0.04",
            "overlap_seconds": "0.04",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


async def test_transcription_maps_audio_decode_failure(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.txt", b"not actually audio", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "audio_decode_failed"


@pytest.mark.parametrize("endpoint", ["transcriptions", "transcription-jobs"])
async def test_upload_size_limit_rejects_both_entries_and_releases_resources(endpoint: str) -> None:
    app = create_app(settings=Settings(max_upload_mb=1, min_free_disk_mb=0))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        response = await client.post(f"/v1/audio/{endpoint}",
                                     files={"file": ("large.wav", b"x" * (1024 * 1024 + 1), "audio/wav")})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "audio_too_large"
    assert app.state.workspace_manager._reserved_upload_bytes == 0


@pytest.mark.parametrize(("model", "endpoint", "option", "value"), [
    ("qwen3-asr-1.7b", "load", "backend", "vllm"),
    ("qwen3-asr-1.7b", "transcriptions", "backend", "vllm"),
    ("qwen3-asr-1.7b", "transcription-jobs", "backend", "vllm"),
    ("moss-transcribe-diarize-0.9b", "transcriptions", "backend", "vllm"),
    ("qwen3-asr-1.7b", "transcriptions", "timestamps", "word"),
    ("moss-transcribe-diarize-0.9b", "transcriptions", "timestamps", "word"),
    ("moss-transcribe-diarize-0.9b", "transcriptions", "language", "en"),
])
async def test_unsupported_capability_is_rejected_before_audio_decode(
    monkeypatch: pytest.MonkeyPatch, model: str, endpoint: str, option: str, value: str,
) -> None:
    app = create_app(settings=Settings(enable_moss=True))
    called = False

    def fail_decode(audio: bytes) -> object:
        nonlocal called
        called = True
        raise AssertionError("decode should not run for unsupported capabilities")

    monkeypatch.setattr(main_module, "run_transcription_path", fail_decode)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        if endpoint == "load":
            response = await client.post(f"/v1/models/{model}/load", json={option: value})
        else:
            response = await client.post(f"/v1/audio/{endpoint}",
                                         files={"file": ("sample.wav", b"not actually audio", "audio/wav")},
                                         data={"model": model, option: value})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "capability_not_supported"
    assert called is False


async def test_unload_waits_for_active_request_and_rejects_new_requests() -> None:
    app = create_app(adapter_delay_seconds=0.2)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        transcription_task = asyncio.create_task(
            client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
                data={"model": "qwen3-asr-1.7b", "backend": "transformers"},
            )
        )
        for _ in range(30):
            status_response = await client.get("/v1/models/qwen3-asr-1.7b/status")
            if status_response.json()["active_requests"] == 1:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("transcription did not enter active request state")

        unload_response = await client.request(
            "DELETE",
            "/v1/models/qwen3-asr-1.7b",
            json={"mode": "after_current_requests", "reject_new_requests": True, "cuda_empty_cache": True},
        )
        assert unload_response.status_code == 200
        assert unload_response.json()["status"] == "unloading_scheduled"
        assert unload_response.json()["active_requests"] == 1

        rejected_response = await client.post(
            "/v1/audio/transcriptions",
            files={"file": ("sample.wav", make_wav(0.1), "audio/wav")},
            data={"model": "qwen3-asr-1.7b", "backend": "transformers"},
        )
        assert rejected_response.status_code == 409
        assert rejected_response.json()["error"]["code"] == "model_unloading_scheduled"

        completed_response = await transcription_task
        assert completed_response.status_code == 200

        status_response = await client.get("/v1/models/qwen3-asr-1.7b/status")
        assert status_response.json()["status"] == "unloaded"
        assert status_response.json()["active_requests"] == 0
