"""Explicit, internal ASR engine registration; no dynamic third-party plugins."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from asr_server.adapters.base import AsrAdapter
from asr_server.config import Settings
from asr_server.errors import AsrError
from asr_server.registry import ModelDefinition


AdapterFactory = Callable[[ModelDefinition, Settings], AsrAdapter]


@dataclass(frozen=True)
class EngineRegistration:
    engine_id: str
    adapter_factory: AdapterFactory
    worker_entry: str
    supported_pipelines: tuple[str, ...] = ("legacy",)
    runtime_profile_id: str = "managed-python"
    batch_mode: str = "serial"


def _qwen(definition: ModelDefinition, settings: Settings) -> AsrAdapter:
    from asr_server.adapters.qwen import QwenAsrAdapter

    return QwenAsrAdapter(
        definition.id, enable_alignment=settings.enable_qwen_alignment,
        repository=definition.repository, revision=definition.revision,
        startup_timeout_seconds=settings.worker_startup_timeout_seconds,
        load_timeout_seconds=settings.worker_load_timeout_seconds,
        inference_timeout_seconds=settings.worker_inference_timeout_seconds,
        shutdown_timeout_seconds=settings.worker_shutdown_timeout_seconds,
    )


def _moss(definition: ModelDefinition, settings: Settings) -> AsrAdapter:
    from asr_server.adapters.moss import MossTranscribeDiarizeAdapter

    return MossTranscribeDiarizeAdapter(
        definition.id, repository=definition.repository, revision=definition.revision,
        startup_timeout_seconds=settings.worker_startup_timeout_seconds,
        load_timeout_seconds=settings.worker_load_timeout_seconds,
        inference_timeout_seconds=settings.worker_inference_timeout_seconds,
        shutdown_timeout_seconds=settings.worker_shutdown_timeout_seconds,
    )


def default_engines() -> dict[str, EngineRegistration]:
    return {
        "qwen": EngineRegistration("qwen", _qwen, "asr_server.adapters.qwen:_qwen_worker_main", ("legacy", "quality_v1")),
        "moss": EngineRegistration("moss", _moss, "asr_server.adapters.moss:_moss_worker_main"),
    }


def create_registered_adapter(definition: ModelDefinition, settings: Settings, *,
                              engines: Mapping[str, EngineRegistration] | None = None,
                              adapter_delay_seconds: float = 0.0) -> AsrAdapter:
    registration = (engines if engines is not None else default_engines()).get(definition.engine_id)
    if registration is None:
        raise AsrError(422, "capability_not_supported", "model engine is not registered")
    if definition.runtime_profile_id != registration.runtime_profile_id or definition.runtime_profile_id != "managed-python":
        raise AsrError(422, "capability_not_supported", "model runtime profile is not enabled")
    if not set(definition.pipelines).issubset(registration.supported_pipelines):
        raise AsrError(422, "capability_not_supported", "model pipeline is not supported by its engine")
    if definition.default_pipeline not in definition.pipelines:
        raise AsrError(422, "capability_not_supported", "model default pipeline is not enabled")
    if settings.adapter == "mock":
        from asr_server.adapters.mock import MockAsrAdapter

        return MockAsrAdapter(delay_seconds=adapter_delay_seconds)
    return registration.adapter_factory(definition, settings)
