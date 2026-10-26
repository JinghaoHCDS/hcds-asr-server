"""Exercise the MP3 workbench in a new, isolated directory without opening a port.

Default mode uses the mock adapter and real FFmpeg media processing. For CUDA
acceptance, configure the installed runtime and pass --real-gpu. The synthetic
retry transition checks recovery plumbing and is not a simulated GPU failure.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import secrets
import sys
import time
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from httpx import ASGITransport, AsyncClient, Response

from asr_server.config import load_settings
from asr_server.main import create_app
from asr_server.workbench.service import WorkbenchService


PUBLIC_FIXTURE_SHA256 = "9121c5126488b541dc75aaced93ffe28a8e80207c87aa040a74e10ef1cec73d2"
WAIT_SECONDS = 7200.0


class SmokeFailure(RuntimeError):
    pass


def require(condition: bool, label: str) -> None:
    if not condition:
        raise SmokeFailure(label)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def expected(response: Response, status: int, operation: str) -> dict[str, Any]:
    if response.status_code != status:
        try:
            code = response.json().get("error", {}).get("code", "unknown_error")
        except ValueError:
            code = "non_json_error"
        raise SmokeFailure(f"{operation}: HTTP {response.status_code}, code={code}, expected={status}")
    if not response.content:
        return {}
    value: dict[str, Any] = response.json()
    return value


async def wait_job(client: AsyncClient, service: WorkbenchService, job_id: str, *,
                   alignment_version: int | None = None) -> tuple[dict[str, Any], list[str]]:
    deadline = time.monotonic() + WAIT_SECONDS
    phases: list[str] = []
    while time.monotonic() < deadline:
        job = expected(await client.get(f"/api/jobs/{job_id}"), 200, "job detail")
        phase = str(job.get("progress", {}).get("phase") or job["status"])
        if not phases or phases[-1] != phase:
            phases.append(phase)
            print(f"workbench smoke: {phase}", file=sys.stderr)
        if alignment_version is None:
            if job["status"] in {"failed", "cancelled"}:
                raise SmokeFailure(f"transcription ended {job['status']}: code={(job.get('error') or {}).get('code', 'unknown')}")
            finished = job["status"] == "completed"
        else:
            result = job.get("result") or {}
            state = result.get("alignment_status")
            if state == "failed":
                raise SmokeFailure(f"edit alignment failed: code={(result.get('alignment_error') or {}).get('code', 'unknown')}")
            finished = result.get("version") == alignment_version and state != "pending"
        if finished and job_id not in service.active and job_id not in service.operation_leases:
            return job, phases
        await asyncio.sleep(0.1)
    raise SmokeFailure("workbench operation timed out")


async def storage(client: AsyncClient) -> dict[str, Any]:
    value = expected(await client.get("/api/storage"), 200, "storage")
    require(value["used_bytes"] == value["saved_bytes"] + value["text_bytes"] + value["temporary_bytes"],
            "storage categories do not sum to used bytes")
    require(value["allocated_bytes"] == value["used_bytes"] + value["reserved_bytes"],
            "storage reservations counted incorrectly")
    require(value["available_bytes"] == max(0, value["quota_bytes"] - value["allocated_bytes"]),
            "storage available quota differs")
    return value


async def check_clean(service: WorkbenchService, client: AsyncClient, job_id: str, *,
                      mp3_sha256: str | None = None) -> dict[str, Any]:
    root = service.job_dir(job_id)
    internal = await service.store.run(service.store.job, job_id)
    asset = internal.get("audio_asset") or {}
    require(asset.get("state") == "ready", "MP3 asset is not ready")
    path = root / "audio.mp3"
    require(path.is_file(), "ready MP3 is missing")
    actual_hash = await asyncio.to_thread(digest, path)
    require(actual_hash == asset.get("sha256"), "MP3 digest differs from metadata")
    if mp3_sha256 is not None:
        require(actual_hash == mp3_sha256, "MP3 changed between operations")
    require(path.stat().st_size == asset.get("size_bytes"), "MP3 size differs from metadata")
    require(not (root / "source").exists(), "source remained after MP3 ready")
    require(not (root / "work").exists() and not (root / "operation").exists(), "operation directory remained after terminal state")
    leftovers = [item for item in service.store.root.rglob("*") if item.is_file()
                 and (item.suffix.lower() == ".wav" or ".partial" in item.name)]
    require(not leftovers, "PCM or partial files remained after operation")
    temp_files = [item for item in (service.store.root / "temporary").rglob("*") if item.is_file()]
    require(not temp_files, "service temporary files remained after operation")
    usage = await storage(client)
    require(usage["temporary_bytes"] == 0 and usage["reserved_bytes"] == 0,
            "temporary bytes or outstanding reservation remained")
    require(not service.operation_leases, "operation resource lease remained")
    require(service.app.state.workspace_manager.pending_disk_bytes(service.store.root) == 0,
            "workspace disk reservation remained")
    pcm = internal.get("normalized_asset") or {}
    require(bool(pcm.get("pcm_sha256")), "normalized PCM identity missing")
    return {"audio": {key: asset.get(key) for key in
                      ("state", "version", "sha256", "size_bytes", "duration_seconds", "sample_rate", "channels",
                       "codec", "bitrate_bps", "source_stream_index", "source_audio_streams", "reused")},
            "pcm_sha256": pcm["pcm_sha256"], "pcm_frames": pcm.get("frame_count"), "storage": usage,
            "source_removed": True, "pcm_and_partial_removed": True, "leases_released": True}


async def exports(client: AsyncClient, job_id: str, version: int, *, aligned: bool) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for format_ in ("txt", "json", "srt", "vtt"):
        response = await client.get(f"/api/jobs/{job_id}/export", params={"format": format_, "version": version})
        wanted = 200 if aligned or format_ in {"txt", "json"} else 409
        require(response.status_code == wanted, f"{format_} export: HTTP {response.status_code}, expected={wanted}")
        if wanted == 200:
            require(bool(response.content), f"{format_} export is empty")
            require(response.headers.get("x-transcript-version") == str(version), f"{format_} export version differs")
            require("private" in response.headers.get("cache-control", ""), "export cache is not private")
        report[format_] = {"status": response.status_code, "size_bytes": len(response.content),
                           "sha256": hashlib.sha256(response.content).hexdigest()}
    return report


async def run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    settings = load_settings()
    if args.real_gpu:
        require(settings.adapter != "mock", "--real-gpu requires ASR_ADAPTER=qwen or registered")
        require(settings.enable_qwen_alignment, "--real-gpu requires ASR_ENABLE_QWEN_ALIGNMENT=1")
        torch = importlib.import_module("torch")
        require(torch.version.cuda is not None and torch.cuda.is_available(), "CUDA-enabled torch is required")
        report["runtime"] = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                             "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))}
    else:
        settings = replace(settings, adapter="mock", enable_qwen_alignment=True)
        report["runtime"] = {"adapter": "mock", "quality_evidence": False}
    settings = replace(settings, workbench_enabled=True, workbench_data_dir=args.data_dir,
                       api_token=secrets.token_urlsafe(32), log_dir=args.data_dir / "logs", log_level="WARNING")
    report["parameters"] = {"audio_filename": args.audio.name, "source_sha256": digest(args.audio),
                            "source_bytes": args.audio.stat().st_size, "adapter": settings.adapter,
                            "real_gpu": args.real_gpu, "quality_profile": "quality_v1"}
    previous_temp = os.environ.get("ASR_TEMP_DIR")
    os.environ["ASR_TEMP_DIR"] = str(args.data_dir / "temporary")
    try:
        app = create_app(settings)
        service: WorkbenchService = app.state.workbench
        transport = ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            password = secrets.token_urlsafe(24)
            username = "smoke_" + secrets.token_hex(8)
            other_username = "other_" + secrets.token_hex(8)
            await service.store.run(service.store.create_user, username, password)
            await service.store.run(service.store.create_user, other_username, password)
            async with AsyncClient(transport=transport, base_url="http://smoke.local", timeout=60) as client:
                expected(await client.post("/api/auth/login", json={"username": username, "password": password}), 200, "login")
                initial = await storage(client)
                require(initial["allocated_bytes"] == 0, "fresh workbench usage is nonzero")
                caps = expected(await client.get("/api/capabilities"), 200, "capabilities")
                model = str(caps.get("default_model") or settings.default_model)
                require(caps.get("default_quality_profile") == "quality_v1", "default model does not support quality_v1")
                report["parameters"]["model"] = model
                models = expected(await client.get("/v1/models", headers={"Authorization": f"Bearer {settings.api_token}"}),
                                  200, "models")
                require(any(item["id"] == model and item["capabilities"]["forced_alignment"] for item in models["models"]),
                        "default model does not declare forced alignment")
                upload = expected(await client.post("/api/uploads", json={"filename": args.audio.name,
                                                                         "size_bytes": args.audio.stat().st_size}), 201, "create upload")
                offset = 0
                chunks = 0
                # Deliberately use several API chunks even for the short fixture.
                with args.audio.open("rb") as source:
                    while block := source.read(min(int(upload["chunk_size"]), 1024 * 1024)):
                        updated = expected(await client.patch(f"/api/uploads/{upload['id']}", content=block,
                                                             headers={"Upload-Offset": str(offset)}), 200, "upload chunk")
                        offset += len(block)
                        require(updated["offset"] == offset, "server upload offset differs")
                        chunks += 1
                report["upload"] = {"chunks": chunks, "bytes": offset}
                started = time.monotonic()
                submitted = expected(await client.post(f"/api/uploads/{upload['id']}/complete", json={
                    "model": model, "quality_profile": "quality_v1", "language": "auto", "domain": "general",
                    "idempotency_key": "isolated-audio-smoke"}), 202, "submit")
                job_id = str(submitted["id"])
                job, phases = await wait_job(client, service, job_id)
                result = job["result"]
                require(bool(result.get("text", "").strip()), "transcript is empty")
                initial_check = await check_clean(service, client, job_id)
                initial_check["seconds"] = round(time.monotonic() - started, 3)
                initial_check["phases"] = phases
                initial_check["transcript_chars"] = len(result["text"])
                if report["parameters"]["source_sha256"] == PUBLIC_FIXTURE_SHA256:
                    initial_check["public_fixture_text_prefix"] = result["text"][:200]
                report["transcription"] = initial_check
                mp3_hash = str(initial_check["audio"]["sha256"])
                pcm_hash = initial_check["pcm_sha256"]
                response = await client.get(f"/api/jobs/{job_id}/audio", headers={"Range": "bytes=0-63"})
                require(response.status_code == 206 and len(response.content) == 64, "MP3 Range response differs")
                require(response.headers.get("content-type", "").startswith("audio/mpeg"), "MP3 media type differs")
                require("private" in response.headers.get("cache-control", ""), "audio cache is not private")
                report["playback"] = {"status": 206, "content_type": response.headers["content-type"],
                                      "content_range": response.headers.get("content-range"), "private_cache": True}
                async with AsyncClient(transport=transport, base_url="http://smoke.local", timeout=60) as other:
                    expected(await other.post("/api/auth/login", json={"username": other_username, "password": password}), 200, "other login")
                    for suffix in ("", "/audio", "/export?format=txt"):
                        require((await other.get(f"/api/jobs/{job_id}{suffix}")).status_code == 404, "cross-user task access was not denied")
                report["playback"]["cross_user_404"] = True
                initial_aligned = result.get("alignment_status") == "aligned"
                if args.real_gpu:
                    require(initial_aligned, "real GPU transcript was not aligned")
                report["exports_initial"] = await exports(client, job_id, int(result["version"]), aligned=initial_aligned)
                segment = next((item for item in result["segments"] if item["text"].strip()), None)
                require(segment is not None, "no editable transcript segment")
                assert segment is not None
                edited_text = str(segment["text"]).rstrip("。.!！") + "。"
                if edited_text == segment["text"]:
                    edited_text += "。"
                version = int(result["version"]) + 1
                started = time.monotonic()
                expected(await client.patch(f"/api/jobs/{job_id}/segments/{segment['id']}", json={
                    "text": edited_text, "base_version": int(result["version"])}), 200, "edit")
                edited, edit_phases = await wait_job(client, service, job_id, alignment_version=version)
                require(edited["result"]["alignment_status"] == "aligned", "edited transcript did not align")
                alignment_check = await check_clean(service, client, job_id, mp3_sha256=mp3_hash)
                require(alignment_check["pcm_sha256"] == pcm_hash, "editing rebuilt different PCM")
                alignment_check.update({"seconds": round(time.monotonic() - started, 3), "phases": edit_phases,
                                        "version": version, "pcm_rebuilt_with_same_identity": True})
                report["edit_alignment"] = alignment_check
                report["exports_edited"] = await exports(client, job_id, version, aligned=True)
                # Controlled engineering state change: no claim of a GPU error.
                async with service.lock:
                    internal = await service.store.run(service.store.job, job_id)
                    service.terminal(internal, "failed")
                    internal["error"] = {"code": "smoke_retry_fixture", "message": "engineering retry check", "details": {}}
                    await service.store.run(service.store.save_job, internal)
                started = time.monotonic()
                expected(await client.post(f"/api/jobs/{job_id}/retry"), 202, "retry")
                _retried, retry_phases = await wait_job(client, service, job_id)
                retry_check = await check_clean(service, client, job_id, mp3_sha256=mp3_hash)
                require(retry_check["pcm_sha256"] == pcm_hash, "retry rebuilt different PCM")
                retry_check.update({"seconds": round(time.monotonic() - started, 3), "phases": retry_phases,
                                    "engineering_status_transition": "completed_to_failed", "pcm_rebuilt_with_same_identity": True})
                report["retry"] = retry_check
                expected(await client.delete(f"/api/jobs/{job_id}"), 204, "delete")
                require(not service.job_dir(job_id).exists(), "deleted job directory remained")
                require(not service.upload_path(str(upload["id"])).exists(), "deleted upload file remained")
                final = await storage(client)
                require(final["allocated_bytes"] == 0, "deleted task storage did not return to zero")
                require((await client.get(f"/api/jobs/{job_id}")).status_code == 404, "deleted job remained visible")
                report["deletion"] = {"task_removed": True, "storage": final}
                # Remove credentials for this completed isolated run.
                await client.post("/api/auth/logout")
                await service.transaction(lambda db: db.execute("DELETE FROM sessions"))
                await service.transaction(lambda db: db.execute("DELETE FROM users"))
        report["status"] = "passed"
    finally:
        if previous_temp is None:
            os.environ.pop("ASR_TEMP_DIR", None)
        else:
            os.environ["ASR_TEMP_DIR"] = previous_temp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="new or empty isolated workbench directory")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON evidence file")
    parser.add_argument("--real-gpu", action="store_true")
    args = parser.parse_args()
    args.data_dir = args.data_dir.absolute()
    args.audio = args.audio.resolve()
    args.output = args.output.absolute()
    if args.data_dir.is_symlink() or (args.data_dir.exists() and (not args.data_dir.is_dir() or any(args.data_dir.iterdir()))):
        parser.error("--data-dir must be new or empty and must not be a symlink")
    if not args.audio.is_file():
        parser.error("--audio must be an existing file")
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be a new file")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"status": "failed", "kind": "isolated_workbench_audio_smoke", "opened_port": False}
    started = time.monotonic()
    try:
        with redirect_stdout(sys.stderr):
            asyncio.run(run(args, report))
    except (Exception, KeyboardInterrupt) as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc) if isinstance(exc, SmokeFailure) else "runtime failure; inspect stderr"}
        print(f"workbench smoke: {type(exc).__name__}: {exc}", file=sys.stderr)
    report["seconds"] = round(time.monotonic() - started, 3)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({"status": report["status"], "seconds": report["seconds"], "mode": "real_gpu" if args.real_gpu else "mock"}, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
