"""Exercise login, resumable upload, Qwen, edit alignment, exports and deletion."""
from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from httpx import ASGITransport, AsyncClient
from asr_server.config import Settings
from asr_server.main import create_app
from scripts.validation_output import add_report_arguments, emit_report, numeric_metrics, text_metrics, validate_report_output


async def run(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="asr-workbench-smoke-") as directory:
        app = create_app(Settings(adapter="mock" if args.mock else "qwen", default_model=args.model,
            enable_qwen_alignment=True, workbench_data_dir=Path(directory), min_free_disk_mb=0,
            idle_unload_seconds=0))
        service = app.state.workbench
        password = secrets.token_urlsafe(24)
        service.store.create_user("smoke-admin", password, True)
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://workbench-smoke", timeout=30) as client:
                signed_in = await client.post("/api/auth/login", json={"username": "smoke-admin", "password": password})
                assert signed_in.status_code == 200, "smoke login failed"
                caps = (await client.get("/api/capabilities")).json()
                audio = args.audio.read_bytes()
                upload = await client.post("/api/uploads", json={"filename": args.audio.name, "size_bytes": len(audio)})
                assert upload.status_code == 201, "smoke upload creation failed"
                upload_id = upload.json()["id"]
                half = len(audio) // 2
                first = await client.patch(f"/api/uploads/{upload_id}", content=audio[:half], headers={"Upload-Offset": "0"})
                assert first.status_code == 200, "smoke upload part failed"
                replay = await client.patch(f"/api/uploads/{upload_id}", content=audio[:half], headers={"Upload-Offset": "0"})
                assert replay.status_code == 200 and replay.json()["offset"] == half
                second = await client.patch(f"/api/uploads/{upload_id}", content=audio[half:], headers={"Upload-Offset": str(half)})
                assert second.status_code == 200, "smoke upload completion failed"
                started = perf_counter()
                created = await client.post(f"/api/uploads/{upload_id}/complete", json={"domain": "ai", "hotwords": "Qwen3-ASR,ABSENTTERMZXQ", "idempotency_key": "smoke"})
                assert created.status_code == 202, "smoke job creation failed"
                job_id = created.json()["id"]
                phases: list[str] = []
                while perf_counter() - started < args.timeout:
                    job = (await client.get(f"/api/jobs/{job_id}")).json()
                    if not phases or phases[-1] != job["status"]:
                        phases.append(job["status"])
                    if job["status"] in {"failed", "cancelled", "completed"}:
                        break
                    await asyncio.sleep(.2)
                assert job["status"] == "completed", "smoke transcription did not complete"
                result = job["result"]
                assert result["text"].strip()
                assert "ABSENTTERMZXQ" not in result["text"]
                hidden = await client.get(f"/v1/jobs/{job_id}")
                assert hidden.status_code == 404
                exports = {}
                for format_ in ["txt", "srt", "vtt", "json"]:
                    response = await client.get(f"/api/jobs/{job_id}/export", params={"format": format_, "version": 1})
                    exports[format_] = response.status_code
                    if not args.mock or format_ in {"txt", "json"}:
                        assert response.status_code == 200, "smoke export failed"
                segment = next(s for s in result["segments"] if s["text"].strip())
                edit = await client.patch(f"/api/jobs/{job_id}/segments/{segment['id']}", json={"text": segment["text"] + "。", "base_version": 1})
                assert edit.status_code == 200, "smoke edit failed"
                pending_export = await client.get(f"/api/jobs/{job_id}/export", params={"format": "srt", "version": 2})
                for _ in range(300):
                    edited = (await client.get(f"/api/jobs/{job_id}")).json()["result"]
                    if edited["alignment_status"] != "pending":
                        break
                    await asyncio.sleep(.2)
                assert edited["alignment_status"] == "aligned", "smoke edit alignment failed"
                ranged = await client.get(f"/api/jobs/{job_id}/audio", headers={"Range": "bytes=0-31"})
                assert ranged.status_code == 206 and len(ranged.content) == 32
                report = {"mock": args.mock, "model": args.model, "profile": result.get("quality_profile"),
                    "phases": phases, "text": result["text"], "raw_text": result.get("raw_text"),
                    "segments": len(result["segments"]), "timestamps": len(result.get("timestamps", [])),
                    "timestamp_sources": sorted({t["source"] for t in result.get("timestamps", [])}),
                    "quality": result.get("quality"), "timings": result.get("timings"),
                    "exports": exports, "edit_version": edited["version"], "edit_alignment": edited["alignment_status"],
                    "pending_export_status": pending_export.status_code, "range_status": ranged.status_code,
                    "capability_models": [m["id"] for m in caps["models"]]}
                deleted = await client.delete(f"/api/jobs/{job_id}")
                assert deleted.status_code == 204
                assert (await client.get(f"/api/jobs/{job_id}")).status_code == 404
                report["deleted"] = True
                return report


def report_summary(report: dict[str, Any]) -> dict[str, Any]:
    """The in-process run() result is private; only this projection is printed."""
    fields = ("mock", "model", "profile", "phases", "segments", "timestamps", "timestamp_sources",
              "exports", "edit_version", "edit_alignment", "pending_export_status", "range_status",
              "capability_models", "deleted")
    return {**{field: report[field] for field in fields if field in report},
            **text_metrics(str(report.get("text") or "")),
            **text_metrics(str(report.get("raw_text") or ""), "raw_text"),
            "quality": numeric_metrics(report.get("quality")), "timings": numeric_metrics(report.get("timings"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=Path("test-fixtures/audio/test_short.wav"))
    parser.add_argument("--model", choices=["qwen3-asr-0.6b", "qwen3-asr-1.7b"], default="qwen3-asr-1.7b")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    add_report_arguments(parser)
    args = parser.parse_args()
    try:
        validate_report_output(args.output, include_content=args.include_content)
        report = asyncio.run(run(args))
        emit_report(report_summary(report), output=args.output, content=report, include_content=args.include_content)
    except Exception as exc:
        parser.exit(1, f"workbench smoke failed ({type(exc).__name__})\n")


if __name__ == "__main__":
    main()
