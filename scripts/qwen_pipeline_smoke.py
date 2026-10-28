from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr_server.config import Settings
from asr_server.main import create_app
from scripts.validation_output import add_report_arguments, emit_report, numeric_metrics, text_metrics, validate_report_output


async def run(args: argparse.Namespace) -> None:
    app = create_app(
        Settings(
            adapter="qwen",
            default_model=args.model,
            enable_qwen_alignment=True,
            idle_unload_seconds=0,
        )
    )
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://pipeline-smoke",
            timeout=args.timeout,
        ) as client:
            with args.audio.open("rb") as source:
                response = await client.post(
                    "/v1/audio/transcriptions",
                    files={"file": (args.audio.name, source, "application/octet-stream")},
                    data={
                        "purpose": "oral_analysis",
                        "language": args.language,
                        "context": args.context,
                        "hotwords": json.dumps(args.keyword, ensure_ascii=False),
                    },
                )
            if response.status_code != 200:
                raise RuntimeError(f"pipeline returned HTTP {response.status_code}")
            body = response.json()
            text = str(body["text"])
            if args.reject_hallucinated_term and args.reject_hallucinated_term.casefold() in text.casefold():
                raise RuntimeError("negative Context check failed: absent keyword appeared in transcript")
            emit_report({"model": body["model"], **text_metrics(text),
                         "timestamps_count": len(body["timestamps"]),
                         "timing": numeric_metrics(body["timing"]),
                         "generation": numeric_metrics(body["generation"]),
                         "timings": numeric_metrics(body["timings"]),
                         "warnings_count": len(body["warnings"])},
                        output=getattr(args, "output", None), content=body,
                        include_content=getattr(args, "include_content", False))
    finally:
        await app.state.manager.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise the full local Qwen upload/context/alignment pipeline.")
    parser.add_argument("--audio", type=Path, default=Path("test-fixtures/audio/test_short.wav"))
    parser.add_argument("--model", default="qwen3-asr-0.6b", choices=["qwen3-asr-0.6b", "qwen3-asr-1.7b"])
    parser.add_argument("--language", default="auto")
    parser.add_argument("--context", default="Reference vocabulary only; transcribe only speech present.")
    parser.add_argument("--keyword", action="append", default=[])
    parser.add_argument("--reject-hallucinated-term")
    parser.add_argument("--timeout", type=float, default=300.0)
    add_report_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        validate_report_output(args.output, include_content=args.include_content)
        if not args.audio.is_file():
            raise ValueError("audio must be a file")
        asyncio.run(run(args))
    except Exception as exc:
        raise SystemExit(f"Qwen pipeline smoke failed ({type(exc).__name__})") from None


if __name__ == "__main__":
    main()
