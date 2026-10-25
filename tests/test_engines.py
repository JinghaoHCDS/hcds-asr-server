"""A third basic engine exercises the generic boundaries without GPU packages."""
from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from asr_server.adapters.base import (
    AudioInput, ContextApplication, GenerationMetrics, TimestampItem,
    TranscriptionResult, TranscriptionSegment, TranscriptionTimings,
)
from asr_server.config import Settings
from asr_server.engines import EngineRegistration, create_registered_adapter
from asr_server.errors import AsrError
from asr_server.execution import ModelExecutionPolicy
from asr_server.main import create_app
from asr_server.registry import ModelCapabilities, ModelDefinition, default_models
from asr_server.workbench.service import WorkbenchService
from asr_server.workers.audio import audio_duration_seconds
from asr_server.workers.codec import result_from_payload, result_to_payload


class BasicEngineAdapter:
    """No align_text, model-specific branching, GPU import or production entry."""

    def __init__(self, definition: ModelDefinition) -> None:
        self.definition = definition
        self.calls = 0

    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        pass

    async def unload(self, cuda_empty_cache: bool) -> None:
        pass

    async def abort(self) -> None:
        pass

    async def transcribe(self, audio: AudioInput, *, model_id: str, backend: str, language: str, context: str,
                         timestamps: str = "none", max_new_tokens: int | None = None) -> TranscriptionResult:
        self.calls += 1
        return TranscriptionResult(text="基础引擎输出", duration=audio_duration_seconds(audio), language="zh", warnings=[])

    async def transcribe_batch(self, audio_chunks: list[AudioInput], *, model_id: str, backend: str, language: str,
                               context: str, timestamps: str = "none", max_new_tokens: int | None = None) -> list[TranscriptionResult]:
        return [await self.transcribe(audio, model_id=model_id, backend=backend, language=language, context=context,
                                      timestamps=timestamps, max_new_tokens=max_new_tokens) for audio in audio_chunks]


def basic_definition() -> ModelDefinition:
    return ModelDefinition(
        id="test-basic", provider="test", default=True, revision="test-revision",
        capabilities=ModelCapabilities(transcription=True, streaming=False, timestamps=[], forced_alignment=False,
                                       languages=["auto", "zh"], chinese_dialects=[], backends=["transformers"]),
        execution_policy=ModelExecutionPolicy("chunked", 128, 1024), engine_id="test-basic-engine",
        repository="test/frozen-model", pipelines=("legacy",), pipeline_version="test-basic-v1",
        resource_type="cpu",
    )


def short_wav() -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 16000)
    return stream.getvalue()


async def test_third_engine_works_through_api_and_workbench_and_rejects_capabilities_before_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASR_TEMP_DIR", str(tmp_path / "temporary"))
    definition = basic_definition()
    variant = replace(definition, id="test-variant", default=False, revision="test-other-revision",
                      capabilities=replace(definition.capabilities, context_prompt=False))
    adapters: dict[str, BasicEngineAdapter] = {}

    def factory(model: ModelDefinition, settings: Settings) -> BasicEngineAdapter:
        adapter = BasicEngineAdapter(model)
        adapters[model.id] = adapter
        return adapter

    engine = EngineRegistration("test-basic-engine", factory, "test:basic")
    app = create_app(Settings(adapter="qwen", workbench_data_dir=tmp_path / "workbench", min_free_disk_mb=0),
                     models={definition.id: definition, variant.id: variant}, engines={engine.engine_id: engine})
    service: WorkbenchService = app.state.workbench
    service.store.create_user("test-user", "test-password")
    await app.state.job_manager.start()
    await service.start()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            models = (await client.get("/v1/models")).json()["models"]
            assert {model["id"] for model in models} == {definition.id, variant.id}
            assert models[0]["engine_id"] == engine.engine_id and models[0]["pipelines"] == ["legacy"]
            for options in ({"timestamps": "auto"}, {"quality_profile": "quality_v1"}, {"model": variant.id, "context": "unsupported prompt"}):
                rejected = await client.post("/v1/audio/transcriptions", files={"file": ("invalid.wav", b"not-media")}, data=options)
                assert rejected.status_code == 422 and rejected.json()["error"]["code"] == "capability_not_supported"
            assert adapters[definition.id].calls == 0
            assert adapters[variant.id].calls == 0
            response = await client.post("/v1/audio/transcriptions", files={"file": ("short.wav", short_wav())})
            assert response.status_code == 200, response.text
            assert response.json()["text"] == "基础引擎输出" and response.json()["model"] == definition.id
            assert adapters[definition.id].definition.repository == definition.repository
            assert (await client.post("/api/auth/login", json={"username": "test-user", "password": "test-password"})).status_code == 200
            capabilities = (await client.get("/api/capabilities")).json()
            assert capabilities["default_model"] == definition.id and capabilities["default_quality_profile"] == "legacy"

            async def upload() -> str:
                content = short_wav()
                record = (await client.post("/api/uploads", json={"filename": "source.wav", "size_bytes": len(content)})).json()
                appended = await client.patch(f"/api/uploads/{record['id']}", content=content, headers={"Upload-Offset": "0"})
                assert appended.status_code == 200
                return str(record["id"])

            upload_id = await upload()
            rejected = await client.post(f"/api/uploads/{upload_id}/complete", json={"model": definition.id, "quality_profile": "quality_v1"})
            assert rejected.status_code == 422
            submitted = await client.post(f"/api/uploads/{upload_id}/complete", json={"model": definition.id, "quality_profile": "legacy", "idempotency_key": "same-audio"})
            assert submitted.status_code == 202, submitted.text
            job_id = submitted.json()["id"]
            result: dict[str, Any] = {}
            for _ in range(200):
                result = (await client.get(f"/api/jobs/{job_id}")).json()
                if result["status"] in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.01)
            assert result["status"] == "completed", result
            assert result["result"]["text"] == "基础引擎输出" and result["result"]["alignment_status"] == "unavailable"
            assert (await client.get(f"/api/jobs/{job_id}/export?format=txt")).status_code == 200
            assert (await client.get(f"/api/jobs/{job_id}/export?format=srt")).status_code == 422
            another_upload = await upload()
            changed_model = await client.post(f"/api/uploads/{another_upload}/complete", json={"model": variant.id, "quality_profile": "legacy", "idempotency_key": "same-audio"})
            assert changed_model.status_code == 409 and changed_model.json()["error"]["code"] == "idempotency_conflict"
    finally:
        await service.shutdown()
        await app.state.job_manager.shutdown()
        await app.state.manager.shutdown()
        service.close()
    assert "test-basic" not in default_models()


def test_shared_codec_preserves_model_results_and_runtime_profiles_fail_closed() -> None:
    result = TranscriptionResult(
        text="hello", duration=2, language="en", warnings=["review"],
        segments=[TranscriptionSegment(0, 2, "hello", "S01", "chunk-0001:S01", "resolved")],
        timestamps=[TimestampItem("hello", 0, 2, "word", source="test-aligner")],
        timings=TranscriptionTimings(20, 1, 2, 15, 2),
        generation=GenerationMetrics(prompt_tokens=3, generated_tokens=4, max_new_tokens=128,
                                     peak_vram_allocated_mb=1024, segment_coverage_end_seconds=2,
                                     segment_coverage_ratio=1, audible_speech_end_seconds=2,
                                     uncovered_speech_intervals=0, truncated=False, finish_reason="eos"),
        context=ContextApplication(True, True, "prompt", 10, 10, 1, False),
    )
    assert result_from_payload(result_to_payload(result)) == result
    definition = replace(basic_definition(), runtime_profile_id="unavailable-runtime")
    with pytest.raises(AsrError) as rejected:
        create_registered_adapter(definition, Settings(), engines={definition.engine_id: EngineRegistration(
            definition.engine_id, lambda model, settings: BasicEngineAdapter(model), "test:basic")})
    assert rejected.value.code == "capability_not_supported"
    assert default_models("moss-transcribe-diarize-0.9b", enable_moss=True)["moss-transcribe-diarize-0.9b"].default
    with pytest.raises(ValueError, match="enabled model"):
        default_models("moss-transcribe-diarize-0.9b")
