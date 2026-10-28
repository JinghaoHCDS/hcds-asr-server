from __future__ import annotations

import asyncio
import json
import logging
import threading

import pytest

from asr_server.adapters.base import AudioInput, TranscriptionResult
from asr_server.lifecycle import ModelLifecycleManager
from asr_server.observability import event_record
from asr_server.registry import default_models


class FailingLoadAdapter:
    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        del backend, device, dtype, max_new_tokens
        raise RuntimeError("load failed")

    async def unload(self, cuda_empty_cache: bool) -> None:
        del cuda_empty_cache

    async def abort(self) -> None:
        return

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
        raise AssertionError("transcribe should not run when load fails")

    async def transcribe_batch(
        self,
        audio_chunks: list[AudioInput],
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        timestamps: str = "none",
        max_new_tokens: int | None = None,
    ) -> list[TranscriptionResult]:
        del audio_chunks, model_id, backend, language, context, timestamps, max_new_tokens
        raise AssertionError("transcribe_batch should not run when load fails")


class RecordingAdapter:
    def __init__(self) -> None:
        self.loads: list[int | None] = []
        self.unloads = 0
        self.active_transcribes = 0
        self.max_parallel_transcribes = 0
        self.unload_while_transcribing = False
        self.load_while_transcribing = False
        self.transcribe_started = threading.Event()

    async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
        del backend, device, dtype
        if self.active_transcribes > 0:
            self.load_while_transcribing = True
        self.loads.append(max_new_tokens)

    async def unload(self, cuda_empty_cache: bool) -> None:
        del cuda_empty_cache
        self.unloads += 1
        if self.active_transcribes > 0:
            self.unload_while_transcribing = True

    async def abort(self) -> None:
        return

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
        del audio, model_id, backend, context, timestamps, max_new_tokens
        self.active_transcribes += 1
        self.transcribe_started.set()
        self.max_parallel_transcribes = max(self.max_parallel_transcribes, self.active_transcribes)
        await asyncio.sleep(0.03)
        self.active_transcribes -= 1
        return TranscriptionResult(text="ok", duration=1.0, language="zh" if language == "auto" else language, warnings=[])

    async def transcribe_batch(
        self,
        audio_chunks: list[AudioInput],
        *,
        model_id: str,
        backend: str,
        language: str,
        context: str,
        timestamps: str = "none",
        max_new_tokens: int | None = None,
    ) -> list[TranscriptionResult]:
        del audio_chunks, model_id, backend, language, context, timestamps, max_new_tokens
        raise AssertionError("batch path is not used in this test")


async def test_load_failure_sets_model_error_state() -> None:
    manager = ModelLifecycleManager(default_models(), lambda _model_id: FailingLoadAdapter())

    with pytest.raises(RuntimeError, match="load failed"):
        await manager.load_model("qwen3-asr-1.7b", backend="transformers")

    assert manager.runtime_for("qwen3-asr-1.7b").status == "error"


async def test_same_model_transcriptions_are_serialized_without_generation_reload() -> None:
    adapter = RecordingAdapter()
    manager = ModelLifecycleManager(default_models(), lambda _model_id: adapter)

    first = asyncio.create_task(
        manager.transcribe_chunks(
            [b"first"],
            model_id="qwen3-asr-1.7b",
            backend="transformers",
            language="auto",
            timestamps="none",
            context="",
            max_new_tokens=512,
            batch_size=1,
        )
    )
    await asyncio.sleep(0.01)
    second = asyncio.create_task(
        manager.transcribe_chunks(
            [b"second"],
            model_id="qwen3-asr-1.7b",
            backend="transformers",
            language="auto",
            timestamps="none",
            context="",
            max_new_tokens=256,
            batch_size=1,
        )
    )

    await asyncio.gather(first, second)

    assert adapter.loads == [None]
    assert adapter.max_parallel_transcribes == 1
    assert adapter.unload_while_transcribing is False


async def test_explicit_load_waits_for_active_transcription() -> None:
    adapter = RecordingAdapter()
    manager = ModelLifecycleManager(default_models(), lambda _model_id: adapter)

    transcription = asyncio.create_task(
        manager.transcribe_chunks(
            [b"audio"],
            model_id="qwen3-asr-1.7b",
            backend="transformers",
            language="auto",
            timestamps="none",
            context="",
            max_new_tokens=512,
            batch_size=1,
        )
    )
    assert await asyncio.to_thread(adapter.transcribe_started.wait, 1.0)

    load = asyncio.create_task(
        manager.load_model("qwen3-asr-1.7b", backend="transformers")
    )

    await asyncio.gather(transcription, load)

    assert adapter.load_while_transcribing is False
    assert adapter.loads == [None]


async def test_model_unloads_after_idle_timeout() -> None:
    adapter = RecordingAdapter()
    manager = ModelLifecycleManager(default_models(), lambda _model_id: adapter, idle_unload_seconds=0.01)

    await manager.transcribe_chunks(
        [b"audio"],
        model_id="qwen3-asr-1.7b",
        backend="transformers",
        language="auto",
        timestamps="none",
        context="",
        max_new_tokens=512,
        batch_size=1,
    )

    assert manager.runtime_for("qwen3-asr-1.7b").status == "loaded"
    await asyncio.sleep(0.05)

    runtime = manager.runtime_for("qwen3-asr-1.7b")
    assert runtime.status == "unloaded"
    assert runtime.backend is None
    assert adapter.unloads == 1


async def test_shutdown_unloads_loaded_model() -> None:
    adapter = RecordingAdapter()
    manager = ModelLifecycleManager(default_models(), lambda _model_id: adapter, idle_unload_seconds=180)

    await manager.load_model("qwen3-asr-1.7b", backend="transformers")

    assert manager.runtime_for("qwen3-asr-1.7b").status == "loaded"

    await manager.shutdown()

    runtime = manager.runtime_for("qwen3-asr-1.7b")
    assert runtime.status == "unloaded"
    assert runtime.backend is None
    assert adapter.unloads == 1


@pytest.mark.parametrize("operation", ["shutdown", "abort"])
async def test_shutdown_and_abort_log_only_content_free_diagnostics(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, operation: str,
) -> None:
    marker = "private-model-error-marker /private/input.wav"
    records: list[dict[str, object]] = []

    def record_event(event: str, *, level: int = logging.INFO, **fields: object) -> None:
        records.append(event_record(event, level=level, **fields))

    class FailingCleanupAdapter(RecordingAdapter):
        async def unload(self, cuda_empty_cache: bool) -> None:
            if operation == "shutdown":
                raise RuntimeError(marker)
            await super().unload(cuda_empty_cache)

        async def abort(self) -> None:
            if operation == "abort":
                raise RuntimeError(marker)

    adapter = FailingCleanupAdapter()
    manager = ModelLifecycleManager(default_models(), lambda _model_id: adapter)
    await manager.load_model("qwen3-asr-1.7b", backend="transformers")
    monkeypatch.setattr("asr_server.lifecycle.emit", record_event)
    if operation == "shutdown":
        await manager.shutdown()
        expected_event = "model.shutdown_unload_failed"
    else:
        await manager.abort_all_workers()
        expected_event = "model.abort_failed"

    failures = [record for record in records if record["event"] == expected_event]
    assert failures and all(record["error_type"] == "RuntimeError" for record in failures)
    assert all(record["error_chain"] for record in failures)
    assert marker not in json.dumps(records) and marker not in caplog.text
    assert manager.runtime_for("qwen3-asr-1.7b").status == "error"


async def test_new_transcription_resets_idle_unload_timer() -> None:
    adapter = RecordingAdapter()
    manager = ModelLifecycleManager(default_models(), lambda _model_id: adapter, idle_unload_seconds=0.05)

    async def transcribe_once() -> None:
        await manager.transcribe_chunks(
            [b"audio"],
            model_id="qwen3-asr-1.7b",
            backend="transformers",
            language="auto",
            timestamps="none",
            context="",
            max_new_tokens=512,
            batch_size=1,
        )

    await transcribe_once()
    await asyncio.sleep(0.02)
    await transcribe_once()
    await asyncio.sleep(0.03)

    assert manager.runtime_for("qwen3-asr-1.7b").status == "loaded"
    assert adapter.unloads == 0

    await asyncio.sleep(0.05)

    assert manager.runtime_for("qwen3-asr-1.7b").status == "unloaded"
    assert adapter.unloads == 1


async def test_different_models_share_one_global_gpu_operation() -> None:
    monitor_lock = threading.Lock()
    active = 0
    maximum = 0

    class MonitoredAdapter(RecordingAdapter):
        async def load(self, backend: str, device: str, dtype: str, max_new_tokens: int | None = None) -> None:
            nonlocal active, maximum
            with monitor_lock:
                active += 1
                maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            with monitor_lock:
                active -= 1
            await super().load(backend, device, dtype, max_new_tokens)

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
            nonlocal active, maximum
            with monitor_lock:
                active += 1
                maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.03)
                return TranscriptionResult(text="ok", duration=1.0, language="auto", warnings=[])
            finally:
                with monitor_lock:
                    active -= 1

    manager = ModelLifecycleManager(default_models(), lambda _model_id: MonitoredAdapter())

    async def transcribe(model_id: str) -> None:
        await manager.transcribe_chunks(
            [b"audio"],
            model_id=model_id,
            backend="transformers",
            language="auto",
            timestamps="none",
            context="",
            max_new_tokens=512,
            batch_size=1,
        )

    await asyncio.gather(transcribe("qwen3-asr-0.6b"), transcribe("qwen3-asr-1.7b"))

    assert maximum == 1
    assert manager.runtime_for("qwen3-asr-0.6b").status == "unloaded"
    assert manager.runtime_for("qwen3-asr-1.7b").status == "loaded"


async def test_cancelled_request_keeps_active_ownership_until_adapter_finishes() -> None:
    adapter = RecordingAdapter()
    manager = ModelLifecycleManager(default_models(), lambda _model_id: adapter, idle_unload_seconds=60)
    request = asyncio.create_task(
        manager.transcribe_chunks(
            [b"audio"],
            model_id="qwen3-asr-1.7b",
            backend="transformers",
            language="auto",
            timestamps="none",
            context="",
            max_new_tokens=512,
            batch_size=1,
        )
    )
    assert await asyncio.to_thread(adapter.transcribe_started.wait, 1.0)
    request.cancel()
    await asyncio.sleep(0.005)

    assert manager.runtime_for("qwen3-asr-1.7b").active_requests == 1
    assert not request.done()

    with pytest.raises(asyncio.CancelledError):
        await request
    assert manager.runtime_for("qwen3-asr-1.7b").active_requests == 0
