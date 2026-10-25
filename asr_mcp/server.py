from __future__ import annotations

from collections.abc import Awaitable
from pathlib import Path
from typing import Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from asr_mcp.client import AsrHttpClient, BridgeSettings
from asr_mcp.errors import AsrGatewayError


Backend = Literal["auto", "transformers", "vllm"]
ResponseFormat = Literal["json", "verbose_json"]
Timestamps = Literal["none", "auto", "word", "char"]
SplitStrategy = Literal["auto", "none", "fixed", "silero", "energy", "vad"]
SpeakerResolution = Literal["off", "auto", "required"]
Purpose = Literal["transcription", "oral_analysis", "podcast"]

READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
CREATE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)
DESTRUCTIVE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=False,
)


async def _call(operation: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    try:
        return await operation
    except (AsrGatewayError, ValueError, httpx.HTTPError) as exc:
        raise ToolError(str(exc)) from exc


def create_mcp_server(client: AsrHttpClient) -> FastMCP:
    mcp = FastMCP(
        "HCDS ASR Server",
        instructions=(
            "Discover model capabilities before transcribing. Use asynchronous jobs for long audio. "
            "Local file paths must be inside ASR_MCP_ALLOWED_ROOTS."
        ),
    )

    @mcp.tool(annotations=READ_ONLY)
    async def asr_list_models() -> dict[str, Any]:
        """List runtime ASR models, capabilities, backends and load states."""
        return await _call(client.list_models())

    @mcp.tool(annotations=READ_ONLY)
    async def asr_get_model_status(model_id: str) -> dict[str, Any]:
        """Get one model's lifecycle status and active request count."""
        return await _call(client.get_model_status(model_id))

    @mcp.tool(annotations=CREATE)
    async def asr_transcribe_file(
        path: str,
        model: str | None = None,
        language: str = "auto",
        response_format: ResponseFormat | None = None,
        timestamps: Timestamps | None = None,
        backend: Backend = "auto",
        max_new_tokens: int | None = None,
        context: str = "",
        hotwords: list[str] | None = None,
        split_strategy: SplitStrategy = "auto",
        max_chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
        preserve_segments: bool | None = None,
        speaker_resolution: SpeakerResolution | None = None,
        purpose: Purpose = "transcription",
        pause_threshold_seconds: float = 0.8,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Upload and transcribe a local audio file; long audio may return a queued job."""
        return await _call(
            client.transcribe_file(
                Path(path),
                model=model,
                language=language,
                response_format=response_format,
                timestamps=timestamps,
                backend=backend,
                max_new_tokens=max_new_tokens,
                context=context,
                hotwords=hotwords,
                split_strategy=split_strategy,
                max_chunk_seconds=max_chunk_seconds,
                overlap_seconds=overlap_seconds,
                preserve_segments=preserve_segments,
                speaker_resolution=speaker_resolution,
                purpose=purpose,
                pause_threshold_seconds=pause_threshold_seconds,
                idempotency_key=idempotency_key,
            )
        )

    @mcp.tool(annotations=CREATE)
    async def asr_create_job(
        path: str,
        model: str | None = None,
        language: str = "auto",
        response_format: ResponseFormat | None = None,
        timestamps: Timestamps | None = None,
        backend: Backend = "auto",
        max_new_tokens: int | None = None,
        context: str = "",
        hotwords: list[str] | None = None,
        split_strategy: SplitStrategy = "auto",
        max_chunk_seconds: float | None = None,
        overlap_seconds: float | None = None,
        preserve_segments: bool | None = None,
        speaker_resolution: SpeakerResolution | None = None,
        purpose: Purpose = "transcription",
        pause_threshold_seconds: float = 0.8,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Upload a local audio file and enqueue a FIFO asynchronous transcription job."""
        return await _call(
            client.create_job(
                Path(path),
                model=model,
                language=language,
                response_format=response_format,
                timestamps=timestamps,
                backend=backend,
                max_new_tokens=max_new_tokens,
                context=context,
                hotwords=hotwords,
                split_strategy=split_strategy,
                max_chunk_seconds=max_chunk_seconds,
                overlap_seconds=overlap_seconds,
                preserve_segments=preserve_segments,
                speaker_resolution=speaker_resolution,
                purpose=purpose,
                pause_threshold_seconds=pause_threshold_seconds,
                idempotency_key=idempotency_key,
            )
        )

    @mcp.tool(annotations=READ_ONLY)
    async def asr_get_job(job_id: str, detail: Literal["compact", "full"] = "compact") -> dict[str, Any]:
        """Get compact progress by default; request full only when fetching the final result."""
        return await _call(client.get_job(job_id, detail=detail))

    @mcp.tool(annotations=DESTRUCTIVE)
    async def asr_cancel_job(job_id: str) -> dict[str, Any]:
        """Request safe cancellation; active inference stops only at a safe chunk boundary."""
        return await _call(client.cancel_job(job_id))

    @mcp.tool(annotations=CREATE)
    async def asr_load_model(
        model_id: str,
        backend: Backend = "auto",
        device: Literal["cuda"] = "cuda",
        dtype: str = "auto",
    ) -> dict[str, Any]:
        """Start loading a declared model into the WSL GPU worker."""
        return await _call(client.load_model(model_id, backend=backend, device=device, dtype=dtype))

    @mcp.tool(annotations=DESTRUCTIVE)
    async def asr_unload_model(model_id: str, cuda_empty_cache: bool = True) -> dict[str, Any]:
        """Safely unload a model after active requests finish and reject new requests meanwhile."""
        return await _call(client.unload_model(model_id, cuda_empty_cache=cuda_empty_cache))

    return mcp


def main() -> None:
    settings = BridgeSettings.from_env()
    create_mcp_server(AsrHttpClient(settings)).run(transport="stdio")


if __name__ == "__main__":
    main()
