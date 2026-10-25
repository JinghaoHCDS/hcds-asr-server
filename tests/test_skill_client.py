from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest


@pytest.fixture
def skill() -> ModuleType:
    name = "lan_asr_skill_client"
    path = Path(__file__).resolve().parents[1] / "skills/lan-asr/scripts/asr_batch.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def model(*, diarization: bool = False, alignment: bool = True) -> dict[str, Any]:
    return {
        "id": "moss" if diarization else "qwen", "default": not diarization,
        "pipelines": ["legacy"] if diarization else ["legacy", "quality_v1"],
        "capabilities": {
            "transcription": True, "forced_alignment": alignment,
            "timestamps": ["auto", "word", "char"] if alignment else [],
            "diarization": diarization, "languages": ["auto", "en"],
            "speaker_resolution_modes": ["off", "auto", "required"] if diarization else ["off"],
            "max_new_tokens": 65536 if diarization else 4096, "context_prompt": True,
        },
    }


@pytest.fixture
def gateway() -> Iterator[tuple[str, list[tuple[str, bytes, str | None]]]]:
    records: list[tuple[str, bytes, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def send_json(self, payload: dict[str, Any], code: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            records.append((self.path, b"", self.headers.get("Authorization")))
            if self.path == "/v1/models":
                self.send_json({"models": [model()]})
            elif "detail=compact" in self.path:
                self.send_json({"status": "completed"})
            else:
                self.send_json({"status": "completed", "result": {"text": "真实返回文本", "warnings": ["needs_review"]}})

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            records.append((self.path, body, self.headers.get("Authorization")))
            self.send_json({"id": "job_1", "status": "queued"}, 202)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", records
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def batch(skill: ModuleType, tmp_path: Path, base_url: str = "http://test", *options: str) -> tuple[sqlite3.Connection, str]:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (source / "audio.wav").write_bytes(b"fixture-audio")
    args = skill.parser().parse_args(["run", str(source), "--base-url", base_url, *options])
    db = skill.open_db(tmp_path / "batches.sqlite")
    return db, skill.create_batch(db, args)


def test_old_database_migration_keeps_batches_and_defaults(skill: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE batches (id TEXT PRIMARY KEY, created_at TEXT, source_dir TEXT, output_dir TEXT, base_url TEXT, model TEXT, language TEXT, cancelled INTEGER)")
        db.execute("INSERT INTO batches VALUES ('old', 'date', 'source', 'output', 'http://old', 'qwen', 'auto', 0)")
    with skill.open_db(path) as db:
        row = db.execute("SELECT * FROM batches WHERE id='old'").fetchone()
        assert row["base_url"] == "http://old"
        assert row["purpose"] == "transcription"
        assert row["quality_profile"] == "legacy"
        assert row["max_new_tokens"] is None
        assert "source_sha256" in {item[1] for item in db.execute("PRAGMA table_info(files)")}


@pytest.mark.parametrize("options", [
    {"purpose": "podcast", "speaker_resolution": "off"},
    {"purpose": "oral_analysis", "alignment": False},
    {"quality_profile": "quality_v1", "alignment": False},
    {"language": "unsupported"},
    {"max_new_tokens": 8192},
])
def test_explicit_model_rejects_missing_capabilities(skill: ModuleType, monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]) -> None:
    options = dict(options)
    alignment = options.pop("alignment", True)
    monkeypatch.setattr(skill, "request_json", lambda *args, **kwargs: {"models": [model(alignment=alignment), model(diarization=True)]})
    request = {"timestamps": "none", "speaker_resolution": "off", **options}
    with pytest.raises(RuntimeError, match="qwen does not declare"):
        skill.choose_model("http://test", "qwen", **request)


def test_podcast_auto_selection_uses_diarization_even_with_speaker_resolution_off(skill: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skill, "request_json", lambda *args, **kwargs: {"models": [model(), model(diarization=True)]})
    assert skill.choose_model("http://test", None, purpose="podcast", timestamps="none", speaker_resolution="off") == "moss"
    with pytest.raises(RuntimeError):
        skill.choose_model("http://test", "moss", timestamps="none", speaker_resolution="off", quality_profile="quality_v1")


def test_batch_http_flow_persists_profile_and_saves_complete_result(skill: ModuleType, tmp_path: Path, gateway: tuple[str, list[tuple[str, bytes, str | None]]], monkeypatch: pytest.MonkeyPatch) -> None:
    base_url, records = gateway
    monkeypatch.setenv("ASR_TOKEN", "test-token")
    db, batch_id = batch(skill, tmp_path, base_url, "--quality-profile", "quality_v1", "--max-new-tokens", "4096")
    db.close()
    db = skill.open_db(tmp_path / "batches.sqlite")
    original_write = skill.write_result

    def write(output: Path, relative: Path, result: dict[str, Any]) -> tuple[Path, Path]:
        assert db.execute("SELECT status FROM files").fetchone()[0] == "submitted"
        return cast(tuple[Path, Path], original_write(output, relative, result))

    monkeypatch.setattr(skill, "write_result", write)
    skill.process_batch(db, batch_id, 0.001)
    row = db.execute("SELECT * FROM files").fetchone()
    assert row["status"] == "completed"
    assert Path(row["output_text"]).read_text() == "真实返回文本"
    assert json.loads(Path(row["output_json"]).read_text())["warnings"] == ["needs_review"]
    assert skill.batch_exit_code(db, batch_id) == 0
    paths = [item[0] for item in records]
    assert paths[-2:] == ["/v1/jobs/job_1?detail=compact", "/v1/jobs/job_1?detail=full"]
    assert paths.count("/v1/jobs/job_1?detail=full") == 1
    post = next(item for item in records if item[0] == "/v1/audio/transcription-jobs")
    assert b'name="quality_profile"\r\n\r\nquality_v1' in post[1]
    assert b'name="max_new_tokens"\r\n\r\n4096' in post[1]
    assert all(item[2] == "Bearer test-token" for item in records)
    assert not list((tmp_path / "source/asr-transcripts").glob(".lan-asr-*"))
    db.close()


@pytest.mark.parametrize("failure", ["source", "capability"])
def test_lost_job_rechecks_source_and_capabilities(skill: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    monkeypatch.setattr(skill, "request_json", lambda *args, **kwargs: {"models": [model()]})
    db, batch_id = batch(skill, tmp_path)
    db.execute("UPDATE files SET status='submitted', job_id='lost', attempts=1")
    db.commit()
    if failure == "source":
        path = tmp_path / "source/audio.wav"
        original = path.stat()
        path.write_bytes(b"changed-audio")
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))

    def request(base: str, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if path == "/v1/models":
            return {"models": [] if failure == "capability" else [model()]}
        raise skill.ApiError(404, "job_not_found", "lost")

    monkeypatch.setattr(skill, "request_json", request)
    monkeypatch.setattr(skill, "submit_job", lambda *args, **kwargs: pytest.fail("must not resubmit incompatible or changed source"))
    skill.process_batch(db, batch_id, 0.001)
    row = db.execute("SELECT * FROM files").fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] == 1
    assert skill.batch_exit_code(db, batch_id) == 1
    db.close()


def test_result_write_failure_is_not_marked_completed(skill: ModuleType, tmp_path: Path, gateway: tuple[str, list[tuple[str, bytes, str | None]]], monkeypatch: pytest.MonkeyPatch) -> None:
    db, batch_id = batch(skill, tmp_path, gateway[0])

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(skill.os, "replace", fail_replace)
    skill.process_batch(db, batch_id, 0.001)
    row = db.execute("SELECT * FROM files").fetchone()
    assert row["status"] == "failed"
    assert "disk full" in row["error"]
    assert row["output_text"] is None
    assert not list((tmp_path / "source/asr-transcripts").glob(".lan-asr-*"))
    assert skill.batch_exit_code(db, batch_id) == 1
    db.close()


def test_lost_job_resubmission_preserves_frozen_profile_and_stops_after_one_retry(skill: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skill, "request_json", lambda *args, **kwargs: {"models": [model()]})
    db, batch_id = batch(skill, tmp_path, "http://test", "--quality-profile", "quality_v1", "--max-new-tokens", "1234")
    db.execute("UPDATE files SET status='submitted', job_id='lost', attempts=1")
    db.commit()
    submissions: list[dict[str, Any]] = []

    def request(base: str, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if path == "/v1/models":
            return {"models": [model()]}
        raise skill.ApiError(404, "job_not_found", "lost")

    def submit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        submissions.append(kwargs)
        return {"id": "lost-again"}

    monkeypatch.setattr(skill, "request_json", request)
    monkeypatch.setattr(skill, "submit_job", submit)
    skill.process_batch(db, batch_id, 0.001)
    assert len(submissions) == 1
    assert submissions[0]["quality_profile"] == "quality_v1"
    assert submissions[0]["max_new_tokens"] == 1234
    assert db.execute("SELECT status, attempts FROM files").fetchone()[:] == ("failed", 2)
    db.close()


def test_cancelled_batch_never_reports_success(skill: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skill, "request_json", lambda *args, **kwargs: {"models": [model()]})
    db, batch_id = batch(skill, tmp_path)
    skill.cancel_batch(db, batch_id)
    skill.process_batch(db, batch_id, 0.001)
    assert db.execute("SELECT status FROM files").fetchone()[0] == "cancelled"
    assert skill.batch_exit_code(db, batch_id) == 1
    db.close()


def test_base_url_precedence_and_localhost_fallback(skill: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.json"
    monkeypatch.setattr(skill, "DEFAULT_CONFIG", config)
    monkeypatch.delenv("ASR_BASE_URL", raising=False)
    assert skill.resolve_base_url(None) == "http://127.0.0.1:18080"
    config.write_text('{"base_url":"http://configured:18080", "token":"ignored"}')
    assert skill.resolve_base_url(None) == "http://configured:18080"
    monkeypatch.setenv("ASR_BASE_URL", "http://environment:18080")
    assert skill.resolve_base_url(None) == "http://environment:18080"
    assert skill.resolve_base_url("http://explicit:18080/") == "http://explicit:18080"
    config.write_text("broken JSON")
    assert skill.resolve_base_url(None) == "http://environment:18080"
    assert skill.resolve_base_url("http://explicit:18080") == "http://explicit:18080"


def test_bundled_deployment_address_wins_over_old_machine_default(
    skill: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundled = tmp_path / "server.json"
    local = tmp_path / "config.json"
    bundled.write_text('{"base_url":"http://downloaded-server:18080"}')
    local.write_text('{"base_url":"http://old-server:18080"}')
    monkeypatch.setattr(skill, "BUNDLED_CONFIG", bundled)
    monkeypatch.setattr(skill, "DEFAULT_CONFIG", local)
    monkeypatch.delenv("ASR_BASE_URL", raising=False)
    assert skill.resolve_base_url(None) == "http://downloaded-server:18080"
    monkeypatch.setenv("ASR_BASE_URL", "http://environment:18080")
    assert skill.resolve_base_url(None) == "http://environment:18080"
    assert skill.resolve_base_url("http://explicit:18080") == "http://explicit:18080"
    monkeypatch.delenv("ASR_BASE_URL")
    bundled.write_text("broken JSON")
    with pytest.raises(ValueError, match="invalid JSON"):
        skill.resolve_base_url(None)


@pytest.mark.parametrize("content", ['{"base_url":', '[]', '{"base_url":null}', '{"base_url":"file:///audio"}'])
def test_invalid_config_fails_only_when_run_needs_it(skill: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    config = tmp_path / "config.json"
    config.write_text(content)
    monkeypatch.setattr(skill, "DEFAULT_CONFIG", config)
    monkeypatch.delenv("ASR_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="ASR client config|ASR base_url"):
        skill.resolve_base_url(None)
    for arguments in (["status", "--batch", "batch_1"], ["resume", "batch_1"], ["cancel", "batch_1"]):
        assert skill.parser().parse_args(arguments).batch_id == "batch_1"
    with pytest.raises(SystemExit) as outcome:
        skill.parser().parse_args(["run", "--help"])
    assert outcome.value.code == 0
    db_path = tmp_path / "existing.sqlite"
    with skill.open_db(db_path) as db:
        db.execute("INSERT INTO batches (id, created_at, source_dir, output_dir, base_url, model, language, cancelled) VALUES ('old', 'date', 'source', 'output', 'http://old', 'qwen', 'auto', 1)")
    for arguments, expected in ((["status", "--batch", "old"], 0), (["resume", "old"], 1), (["cancel", "old"], 0)):
        monkeypatch.setattr(sys, "argv", ["asr_batch.py", *arguments, "--db", str(db_path)])
        assert skill.main() == expected


def test_old_discovery_without_pipelines_remains_legacy_compatible(skill: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = model()
    entry.pop("pipelines")
    monkeypatch.setattr(skill, "request_json", lambda *args, **kwargs: {"models": [entry]})
    assert skill.choose_model("http://test", "qwen", timestamps="none", speaker_resolution="off") == "qwen"
    with pytest.raises(RuntimeError):
        skill.choose_model("http://test", "qwen", timestamps="none", speaker_resolution="off", quality_profile="quality_v1")
