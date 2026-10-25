from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal

from asr_server.execution import MOSS_EXECUTION_POLICY, QWEN_EXECUTION_POLICY, ModelExecutionPolicy
from asr_server.audio.assets import AudioSpec

Backend = Literal["auto", "transformers", "vllm"]
ModelStatus = Literal["unloaded", "loading", "loaded", "unloading_scheduled", "unloading", "error"]
MOSS_MODEL_ID = "moss-transcribe-diarize-0.9b"

# One identity source for adapters, capability discovery and smoke tools.
MODEL_SOURCES: dict[str, tuple[str, str]] = {
    "qwen3-asr-1.7b": ("Qwen/Qwen3-ASR-1.7B-hf", "057a3b044fcd31c433e7971ab40d68d20e7eae6d"),
    "qwen3-asr-0.6b": ("Qwen/Qwen3-ASR-0.6B-hf", "6aa69c382e2b426eee1f5870d4c95859a74b6445"),
    MOSS_MODEL_ID: ("OpenMOSS-Team/MOSS-Transcribe-Diarize", "d7231bbae2587a4af278735eb765b318c4f64edd"),
}
ALIGNER_SOURCE = ("Qwen/Qwen3-ForcedAligner-0.6B-hf", "c07281df297b9905d24a508279258cccf987a064")


QWEN_LANGUAGES = [
    "auto",
    "zh",
    "en",
    "yue",
    "ar",
    "de",
    "fr",
    "es",
    "pt",
    "id",
    "it",
    "ko",
    "ru",
    "th",
    "vi",
    "ja",
    "tr",
    "hi",
    "ms",
    "nl",
    "sv",
    "da",
    "fi",
    "pl",
    "cs",
    "fil",
    "fa",
    "el",
    "hu",
    "mk",
    "ro",
]

QWEN_CHINESE_DIALECTS = [
    "Anhui",
    "Dongbei",
    "Fujian",
    "Gansu",
    "Guizhou",
    "Hebei",
    "Henan",
    "Hubei",
    "Hunan",
    "Jiangxi",
    "Ningxia",
    "Shandong",
    "Shaanxi",
    "Shanxi",
    "Sichuan",
    "Tianjin",
    "Yunnan",
    "Zhejiang",
    "Cantonese-Hong-Kong-accent",
    "Cantonese-Guangdong-accent",
    "Wu",
    "Minnan",
]


@dataclass(frozen=True)
class ModelCapabilities:
    transcription: bool
    streaming: bool
    timestamps: list[str]
    forced_alignment: bool
    languages: list[str]
    chinese_dialects: list[str]
    backends: list[str]
    diarization: bool = False
    segment_timestamps: bool = False
    execution_modes: list[str] = field(default_factory=lambda: ["chunked"])
    auto_execution_mode: str = "chunked"
    max_new_tokens: int = 4_096
    speaker_scopes: list[str] = field(default_factory=list)
    validated_native_max_seconds: float | None = None
    automatic_fallback_chunk_seconds: float | None = None
    speaker_resolution_modes: list[str] = field(default_factory=lambda: ["off"])
    speaker_resolution_method: str | None = None
    validated_anchor_replay_speakers: int | None = None
    context_prompt: bool = True
    batch_mode: str = "serial"
    native_batch_max_items: int = 1
    native_batch_max_audio_seconds: float = 300.0

    def to_api(self) -> dict[str, object]:
        return {
            "transcription": self.transcription,
            "streaming": self.streaming,
            "timestamps": self.timestamps,
            "forced_alignment": self.forced_alignment,
            "languages": self.languages,
            "chinese_dialects": self.chinese_dialects,
            "backends": self.backends,
            "diarization": self.diarization,
            "segment_timestamps": self.segment_timestamps,
            "execution_modes": self.execution_modes,
            "auto_execution_mode": self.auto_execution_mode,
            "max_new_tokens": self.max_new_tokens,
            "speaker_scopes": self.speaker_scopes,
            "validated_native_max_seconds": self.validated_native_max_seconds,
            "automatic_fallback_chunk_seconds": self.automatic_fallback_chunk_seconds,
            "speaker_resolution_modes": self.speaker_resolution_modes,
            "speaker_resolution_method": self.speaker_resolution_method,
            "validated_anchor_replay_speakers": self.validated_anchor_replay_speakers,
            "context_prompt": self.context_prompt,
            "batch_mode": self.batch_mode,
            "native_batch_max_items": self.native_batch_max_items,
            "native_batch_max_audio_seconds": self.native_batch_max_audio_seconds,
        }


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    provider: str
    default: bool
    capabilities: ModelCapabilities
    revision: str
    execution_policy: ModelExecutionPolicy
    engine_id: str = "qwen"
    repository: str = ""
    audio_spec: AudioSpec = field(default_factory=AudioSpec)
    pipelines: tuple[str, ...] = ("legacy",)
    pipeline_version: str = "legacy-v1"
    engine_version: str = "1"
    default_pipeline: str = "legacy"
    resource_type: str = "gpu"
    runtime_profile_id: str = "managed-python"

    def supports_pipeline(self, profile: str) -> bool:
        return profile in self.pipelines

    def identity(self) -> dict[str, object]:
        """Frozen execution identity for idempotency and checkpoint owners."""
        return {
            "model": self.id, "revision": self.revision, "repository": self.repository,
            "engine_id": self.engine_id, "pipeline_version": self.pipeline_version,
            "engine_version": self.engine_version, "default_pipeline": self.default_pipeline,
            "audio_spec": asdict(self.audio_spec), "runtime_profile_id": self.runtime_profile_id,
        }

    def discovery(self) -> dict[str, object]:
        return {"engine_id": self.engine_id, "pipelines": list(self.pipelines),
                "audio_spec": asdict(self.audio_spec), "runtime_profile_id": self.runtime_profile_id,
                "revision": self.revision, "pipeline_version": self.pipeline_version,
                "engine_version": self.engine_version, "default_pipeline": self.default_pipeline,
                "resource_type": self.resource_type}


def default_models(
    default_model_id: str = "qwen3-asr-1.7b",
    *,
    enable_moss: bool = False,
    enable_qwen_alignment: bool = False,
) -> dict[str, ModelDefinition]:
    qwen_capabilities = ModelCapabilities(
        transcription=True,
        streaming=False,
        timestamps=["auto", "word", "char"] if enable_qwen_alignment else [],
        forced_alignment=enable_qwen_alignment,
        languages=QWEN_LANGUAGES,
        chinese_dialects=QWEN_CHINESE_DIALECTS,
        backends=["transformers"],
    )
    models: dict[str, ModelDefinition] = {
        "qwen3-asr-1.7b": ModelDefinition(
            id="qwen3-asr-1.7b",
            provider="QwenLM",
            default=default_model_id == "qwen3-asr-1.7b",
            capabilities=qwen_capabilities,
            revision=MODEL_SOURCES["qwen3-asr-1.7b"][1],
            execution_policy=QWEN_EXECUTION_POLICY,
            engine_id="qwen", repository=MODEL_SOURCES["qwen3-asr-1.7b"][0],
            pipelines=("legacy", "quality_v1"), pipeline_version="qwen-v1",
        ),
        "qwen3-asr-0.6b": ModelDefinition(
            id="qwen3-asr-0.6b",
            provider="QwenLM",
            default=default_model_id == "qwen3-asr-0.6b",
            capabilities=qwen_capabilities,
            revision=MODEL_SOURCES["qwen3-asr-0.6b"][1],
            execution_policy=QWEN_EXECUTION_POLICY,
            engine_id="qwen", repository=MODEL_SOURCES["qwen3-asr-0.6b"][0],
            pipelines=("legacy", "quality_v1"), pipeline_version="qwen-v1",
        ),
    }
    if enable_moss:
        models[MOSS_MODEL_ID] = ModelDefinition(
            id=MOSS_MODEL_ID,
            provider="OpenMOSS-Team",
            default=False,
            capabilities=ModelCapabilities(
                transcription=True,
                streaming=False,
                timestamps=[],
                forced_alignment=False,
                languages=["auto"],
                chinese_dialects=[],
                backends=["transformers"],
                diarization=True,
                segment_timestamps=True,
                execution_modes=["native_long_form", "chunked"],
                auto_execution_mode="native_long_form",
                max_new_tokens=MOSS_EXECUTION_POLICY.max_new_tokens,
                speaker_scopes=["global", "chunk", "mixed"],
                validated_native_max_seconds=MOSS_EXECUTION_POLICY.validated_native_max_seconds,
                automatic_fallback_chunk_seconds=MOSS_EXECUTION_POLICY.fallback_chunk_seconds,
                speaker_resolution_modes=["off", "auto", "required"],
                speaker_resolution_method="moss_anchor_replay",
                validated_anchor_replay_speakers=4,
                native_batch_max_audio_seconds=MOSS_EXECUTION_POLICY.native_hard_seconds,
            ),
            revision=MODEL_SOURCES[MOSS_MODEL_ID][1],
            execution_policy=MOSS_EXECUTION_POLICY,
            engine_id="moss", repository=MODEL_SOURCES[MOSS_MODEL_ID][0],
            pipelines=("legacy",), pipeline_version="moss-v1",
        )
    if default_model_id not in models:
        raise ValueError("ASR_DEFAULT_MODEL must name an enabled model")
    return {model_id: replace(model, default=model_id == default_model_id) for model_id, model in models.items()}
