#!/usr/bin/env python3
"""Small, dependency-free batch client for the LAN ASR gateway."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import mimetypes
import os
import sqlite3
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mkv", ".mov", ".mp3", ".mp4", ".ogg", ".opus", ".wav", ".webm", ".wma"}
DEFAULT_BASE_URL = "http://127.0.0.1:18080"
DEFAULT_DB = Path.home() / ".Agent" / "state" / "lan-asr" / "batches.sqlite"
DEFAULT_CONFIG = DEFAULT_DB.with_name("config.json")
TERMINAL = {"completed", "failed", "cancelled"}


@dataclass
class ApiError(RuntimeError):
    status: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message} (HTTP {self.status})"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_base_url(explicit: str | None) -> str:
    """Read non-secret client configuration only when creating a new batch."""
    if explicit is not None:
        value = explicit
    elif os.getenv("ASR_BASE_URL"):
        value = os.environ["ASR_BASE_URL"]
    else:
        try:
            content = DEFAULT_CONFIG.read_text(encoding="utf-8")
        except FileNotFoundError:
            return DEFAULT_BASE_URL
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"could not read ASR client config: {DEFAULT_CONFIG}") from exc
        try:
            config = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in ASR client config: {DEFAULT_CONFIG}") from exc
        if not isinstance(config, dict):
            raise ValueError("ASR client config must be a JSON object")
        if "base_url" not in config:
            return DEFAULT_BASE_URL
        value = config["base_url"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ASR base_url must be a non-empty http(s) URL")
    value = value.strip()
    try:
        parsed = urlsplit(value)
        valid = parsed.scheme in {"http", "https"} and parsed.hostname and parsed.port != 0
        valid = valid and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
        valid = valid and not any(character.isspace() for character in value)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("ASR base_url must be an http(s) URL without credentials, query or fragment")
    return value.rstrip("/")


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS batches (
          id TEXT PRIMARY KEY, created_at TEXT NOT NULL, source_dir TEXT NOT NULL,
          output_dir TEXT NOT NULL, base_url TEXT NOT NULL, model TEXT NOT NULL,
          language TEXT NOT NULL, cancelled INTEGER NOT NULL DEFAULT 0,
          purpose TEXT NOT NULL DEFAULT 'transcription',
          response_format TEXT NOT NULL DEFAULT 'json',
          timestamps TEXT NOT NULL DEFAULT 'none',
          speaker_resolution TEXT NOT NULL DEFAULT 'off',
          context TEXT NOT NULL DEFAULT '',
          hotwords TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS files (
          id INTEGER PRIMARY KEY, batch_id TEXT NOT NULL, source_path TEXT NOT NULL,
          relative_path TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
          status TEXT NOT NULL DEFAULT 'pending', job_id TEXT, attempts INTEGER NOT NULL DEFAULT 0,
          error TEXT, output_json TEXT, output_text TEXT,
          UNIQUE(batch_id, source_path), FOREIGN KEY(batch_id) REFERENCES batches(id)
        );
        """
    )
    # Migrate databases created by the pre-1.1 client without discarding
    # existing batches.  These columns are deliberately additive so resume
    # and status remain compatible with the shared state database.
    for column, definition in {
        "purpose": "TEXT NOT NULL DEFAULT 'transcription'",
        "response_format": "TEXT NOT NULL DEFAULT 'json'",
        "timestamps": "TEXT NOT NULL DEFAULT 'none'",
        "speaker_resolution": "TEXT NOT NULL DEFAULT 'off'",
        "context": "TEXT NOT NULL DEFAULT ''",
        "hotwords": "TEXT NOT NULL DEFAULT ''",
        "quality_profile": "TEXT NOT NULL DEFAULT 'legacy'",
        "max_new_tokens": "INTEGER",
    }.items():
        existing = {row[1] for row in db.execute("PRAGMA table_info(batches)")}
        if column not in existing:
            db.execute(f"ALTER TABLE batches ADD COLUMN {column} {definition}")
    if "source_sha256" not in {row[1] for row in db.execute("PRAGMA table_info(files)")}:
        db.execute("ALTER TABLE files ADD COLUMN source_sha256 TEXT")
    db.commit()
    return db


def request_json(base_url: str, method: str, path: str, *, body: bytes | None = None, headers: dict[str, str] | None = None, timeout: float = 30.0) -> dict:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("ASR base URL must be an http(s) URL")
    target = (parsed.path.rstrip("/") + path) or "/"
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    request_headers = {"Accept": "application/json", **(headers or {})}
    token = os.getenv("ASR_TOKEN")
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        request_headers.setdefault("Content-Length", str(len(body)))
    try:
        connection.request(method, target, body=body, headers=request_headers)
        response = connection.getresponse()
        payload_bytes = response.read()
    except OSError as exc:
        raise ApiError(0, "network_error", str(exc)) from exc
    finally:
        connection.close()
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as exc:
        raise ApiError(response.status, "invalid_json", "ASR gateway returned non-JSON data") from exc
    if response.status >= 400:
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        raise ApiError(response.status, str(error.get("code", "http_error")), str(error.get("message", payload)))
    if not isinstance(payload, dict):
        raise ApiError(response.status, "invalid_json", "ASR gateway returned a non-object JSON response")
    return payload


def submit_job(
    base_url: str,
    audio_path: Path,
    *,
    model: str,
    language: str,
    purpose: str,
    response_format: str,
    timestamps: str,
    speaker_resolution: str,
    context: str,
    hotwords: str,
    idempotency_key: str,
    quality_profile: str = "legacy",
    max_new_tokens: int | None = None,
) -> dict:
    boundary = f"----lan-asr-{uuid.uuid4().hex}"
    fields = {
        "model": model,
        "language": language,
        "response_format": response_format,
        "purpose": purpose,
        "timestamps": timestamps,
        "backend": "auto",
        "speaker_resolution": speaker_resolution,
        "context": context,
        "hotwords": hotwords,
        "idempotency_key": idempotency_key,
        "quality_profile": quality_profile,
    }
    if max_new_tokens is not None:
        fields["max_new_tokens"] = str(max_new_tokens)
    prefix = b"".join(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
        for key, value in fields.items()
    )
    content_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    file_header = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{audio_path.name}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()
    # The gateway allows files up to 512 MB. This client streams the upload instead of base64 encoding it.
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("ASR base URL must be an http(s) URL")
    target = (parsed.path.rstrip("/") + "/v1/audio/transcription-jobs") or "/v1/audio/transcription-jobs"
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=60)
    headers = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(prefix) + len(file_header) + audio_path.stat().st_size + len(suffix)),
    }
    token = os.getenv("ASR_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        connection.putrequest("POST", target)
        for key, value in headers.items():
            connection.putheader(key, value)
        connection.endheaders()
        connection.send(prefix)
        connection.send(file_header)
        with audio_path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                connection.send(block)
        connection.send(suffix)
        response = connection.getresponse()
        raw = response.read()
    except OSError as exc:
        raise ApiError(0, "network_error", str(exc)) from exc
    finally:
        connection.close()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApiError(response.status, "invalid_json", "ASR gateway returned non-JSON data") from exc
    if response.status >= 400:
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        raise ApiError(response.status, str(error.get("code", "http_error")), str(error.get("message", payload)))
    if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
        raise ApiError(response.status, "invalid_job_response", "ASR gateway did not return a job id")
    return payload


def choose_model(
    base_url: str,
    requested: str | None,
    *,
    timestamps: str,
    speaker_resolution: str,
    purpose: str = "transcription",
    language: str = "auto",
    quality_profile: str = "legacy",
    max_new_tokens: int | None = None,
    response_format: str = "json",
    context: str = "",
    hotwords: str = "",
) -> str:
    if speaker_resolution != "off" and response_format != "verbose_json":
        raise ValueError("speaker resolution requires --response-format verbose_json")
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    models = request_json(base_url, "GET", "/v1/models").get("models")
    if not isinstance(models, list):
        raise RuntimeError("ASR gateway did not return a models list")

    def supports(item: dict) -> bool:
        capabilities = item.get("capabilities", {})
        if not isinstance(capabilities, dict):
            return False
        if capabilities.get("transcription") is not True:
            return False
        declared_timestamps = capabilities.get("timestamps", [])
        if timestamps != "none" and timestamps not in declared_timestamps and not (timestamps == "auto" and declared_timestamps):
            return False
        if purpose == "oral_analysis" and capabilities.get("forced_alignment") is not True:
            return False
        if purpose == "podcast" and capabilities.get("diarization") is not True:
            return False
        if language not in capabilities.get("languages", []):
            return False
        if quality_profile not in item.get("pipelines", ["legacy"]):
            return False
        if quality_profile == "quality_v1" and capabilities.get("forced_alignment") is not True:
            return False
        if (context or hotwords) and capabilities.get("context_prompt") is not True:
            return False
        if max_new_tokens is not None:
            limit = capabilities.get("max_new_tokens")
            if not isinstance(limit, int) or max_new_tokens > limit:
                return False
        if speaker_resolution != "off" and speaker_resolution not in capabilities.get("speaker_resolution_modes", []):
            return False
        if speaker_resolution != "off" and capabilities.get("diarization") is not True:
            return False
        return True

    available = [item for item in models if isinstance(item, dict) and isinstance(item.get("capabilities"), dict) and item["capabilities"].get("transcription") is True]
    if requested:
        item = next((item for item in available if item.get("id") == requested), None)
        if item is None:
            raise RuntimeError(f"model is not declared for transcription: {requested}")
        if not supports(item):
            raise RuntimeError(
                f"model {requested} does not declare the requested capabilities: "
                f"purpose={purpose}, language={language}, timestamps={timestamps}, "
                f"speaker_resolution={speaker_resolution}, quality_profile={quality_profile}, "
                f"max_new_tokens={max_new_tokens}"
            )
        return requested

    compatible = [item for item in available if supports(item)]
    default = next((item.get("id") for item in compatible if item.get("default") is True), None)
    fallback = next((item.get("id") for item in compatible), None)
    if not isinstance(default or fallback, str):
        raise RuntimeError("ASR gateway declares no transcription model")
    return default or fallback


def iter_audio_files(source: Path, includes: list[str]) -> list[Path]:
    candidates = {path for pattern in includes for path in source.glob(pattern)} if includes else set(source.rglob("*"))
    return sorted(path for path in candidates if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)


def source_digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def check_source(row: sqlite3.Row) -> str:
    """Reject changed inputs before reusing the frozen request/idempotency key."""
    path = Path(row["source_path"])
    stat = path.stat()
    if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
        raise ValueError("source file changed; create a new batch")
    digest = source_digest(path)
    if row["source_sha256"] and digest != row["source_sha256"]:
        raise ValueError("source file changed; create a new batch")
    return digest


def resolve_request_options(args: argparse.Namespace) -> tuple[str, str, str, str, str, str]:
    purpose = args.purpose
    response_format = args.response_format or ("verbose_json" if purpose in {"oral_analysis", "podcast"} else "json")
    timestamps = args.timestamps or ("char" if purpose == "oral_analysis" else "none")
    speaker_resolution = args.speaker_resolution or ("auto" if purpose == "podcast" else "off")
    return (
        purpose,
        response_format,
        timestamps,
        speaker_resolution,
        args.context or "",
        args.hotwords or "",
    )


def create_batch(db: sqlite3.Connection, args: argparse.Namespace) -> str:
    base_url = resolve_base_url(args.base_url)
    source = args.source.resolve()
    if not source.is_dir():
        raise ValueError(f"source directory does not exist: {source}")
    output = (args.output_dir or source / "asr-transcripts").resolve()
    output.mkdir(parents=True, exist_ok=True)
    purpose, response_format, timestamps, speaker_resolution, context, hotwords = resolve_request_options(args)
    model = choose_model(
        base_url,
        args.model,
        timestamps=timestamps,
        speaker_resolution=speaker_resolution,
        purpose=purpose,
        language=args.language,
        quality_profile=args.quality_profile,
        max_new_tokens=args.max_new_tokens,
        response_format=response_format,
        context=context,
        hotwords=hotwords,
    )
    files = iter_audio_files(source, args.include)
    if not files:
        raise ValueError("no supported audio/video files found")
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    db.execute(
        """
        INSERT INTO batches
          (id, created_at, source_dir, output_dir, base_url, model, language,
           cancelled, purpose, response_format, timestamps, speaker_resolution,
           context, hotwords, quality_profile, max_new_tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            batch_id,
            now(),
            str(source),
            str(output),
            base_url,
            model,
            args.language,
            purpose,
            response_format,
            timestamps,
            speaker_resolution,
            context,
            hotwords,
            args.quality_profile,
            args.max_new_tokens,
        ),
    )
    for path in files:
        stat = path.stat()
        digest = source_digest(path)
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            db.rollback()
            raise ValueError("source file changed while preparing batch; try again")
        db.execute(
            "INSERT INTO files (batch_id, source_path, relative_path, size, mtime_ns, source_sha256) VALUES (?, ?, ?, ?, ?, ?)",
            (batch_id, str(path), str(path.relative_to(source)), stat.st_size, stat.st_mtime_ns, digest),
        )
    db.commit()
    print(
        f"batch: {batch_id}\nfiles: {len(files)}\nmodel: {model}\n"
        f"purpose: {purpose}\nresponse_format: {response_format}\n"
        f"timestamps: {timestamps}\nspeaker_resolution: {speaker_resolution}\n"
        f"quality_profile: {args.quality_profile}\nmax_new_tokens: {args.max_new_tokens}\n"
        f"output: {output}"
    )
    return batch_id


def write_result(output_dir: Path, relative_path: Path, result: dict) -> tuple[Path, Path]:
    stem = output_dir / relative_path
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = stem.with_suffix(stem.suffix + ".asr.json")
    text_path = stem.with_suffix(stem.suffix + ".txt")
    if not isinstance(result.get("text"), str):
        raise ValueError("completed job returned no result.text")
    staging: list[tuple[Path, Path]] = []
    try:
        for target, content in (
            (json_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n"),
            (text_path, result["text"]),
        ):
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".lan-asr-", dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                staging.append((temporary, target))
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, target in staging:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staging:
            temporary.unlink(missing_ok=True)
    return json_path, text_path


def process_batch(db: sqlite3.Connection, batch_id: str, poll_seconds: float) -> None:
    batch = db.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    if batch is None:
        raise ValueError(f"unknown batch: {batch_id}")
    while True:
        if db.execute("SELECT cancelled FROM batches WHERE id = ?", (batch_id,)).fetchone()[0]:
            print(f"batch cancelled: {batch_id}")
            return
        row = db.execute("SELECT * FROM files WHERE batch_id = ? AND status IN ('pending', 'submitted') ORDER BY id LIMIT 1", (batch_id,)).fetchone()
        if row is None:
            print_status(db, batch_id)
            return
        path = Path(row["source_path"])
        if row["status"] == "pending":
            if not path.is_file():
                db.execute("UPDATE files SET status='failed', error=? WHERE id=?", ("source file no longer exists", row["id"]))
                db.commit()
                continue
            try:
                digest = check_source(row)
                choose_model(
                    batch["base_url"], batch["model"], timestamps=batch["timestamps"],
                    speaker_resolution=batch["speaker_resolution"], purpose=batch["purpose"],
                    language=batch["language"], quality_profile=batch["quality_profile"],
                    max_new_tokens=batch["max_new_tokens"], response_format=batch["response_format"],
                    context=batch["context"], hotwords=batch["hotwords"],
                )
                if not row["source_sha256"]:
                    db.execute("UPDATE files SET source_sha256=? WHERE id=?", (digest, row["id"]))
                    db.commit()
                created = submit_job(
                    batch["base_url"],
                    path,
                    model=batch["model"],
                    language=batch["language"],
                    purpose=batch["purpose"],
                    response_format=batch["response_format"],
                    timestamps=batch["timestamps"],
                    speaker_resolution=batch["speaker_resolution"],
                    context=batch["context"],
                    hotwords=batch["hotwords"],
                    idempotency_key=f"lan-asr:{batch_id}:{row['id']}",
                    quality_profile=batch["quality_profile"],
                    max_new_tokens=batch["max_new_tokens"],
                )
                # The upload streams the source; do not accept a changed input
                # as the result of the original batch item.
                if source_digest(path) != digest:
                    try:
                        request_json(batch["base_url"], "DELETE", f"/v1/jobs/{created['id']}")
                    except ApiError:
                        pass
                    raise ValueError("source file changed during upload; create a new batch")
            except (ApiError, RuntimeError, ValueError, OSError) as exc:
                db.execute("UPDATE files SET status='failed', error=? WHERE id=?", (str(exc), row["id"]))
                db.commit()
                continue
            db.execute("UPDATE files SET status='submitted', job_id=?, attempts=attempts+1, error=NULL WHERE id=?", (created["id"], row["id"]))
            db.commit()
            print(f"submitted: {row['relative_path']} ({created['id']})")
            continue
        try:
            job = request_json(batch["base_url"], "GET", f"/v1/jobs/{row['job_id']}?detail=compact")
        except ApiError as exc:
            if exc.status == 404 and exc.code == "job_not_found" and row["attempts"] < 2:
                db.execute("UPDATE files SET status='pending', job_id=NULL, error=? WHERE id=?", ("job lost after gateway restart; resubmitting", row["id"]))
            else:
                db.execute("UPDATE files SET status='failed', error=? WHERE id=?", (str(exc), row["id"]))
            db.commit()
            continue
        status = job.get("status")
        if status == "completed":
            try:
                full_job = request_json(batch["base_url"], "GET", f"/v1/jobs/{row['job_id']}?detail=full")
            except ApiError as exc:
                db.execute("UPDATE files SET status='failed', error=? WHERE id=?", (f"completed job full result unavailable: {exc}", row["id"]))
                db.commit()
                continue
            if db.execute("SELECT cancelled FROM batches WHERE id=?", (batch_id,)).fetchone()[0]:
                db.execute("UPDATE files SET status='cancelled', error='batch cancelled' WHERE id=?", (row["id"],))
                db.commit()
                continue
            result = full_job.get("result")
            if full_job.get("status") != "completed":
                db.execute("UPDATE files SET status='failed', error='full result was not completed' WHERE id=?", (row["id"],))
                db.commit()
                continue
            if not isinstance(result, dict) or not isinstance(result.get("text"), str):
                db.execute("UPDATE files SET status='failed', error=? WHERE id=?", ("completed job returned no result.text", row["id"]))
            else:
                try:
                    json_path, text_path = write_result(Path(batch["output_dir"]), Path(row["relative_path"]), result)
                except (OSError, ValueError) as exc:
                    db.execute("UPDATE files SET status='failed', error=? WHERE id=?", (f"could not save result: {exc}", row["id"]))
                else:
                    db.execute("UPDATE files SET status='completed', output_json=?, output_text=?, error=NULL WHERE id=?", (str(json_path), str(text_path), row["id"]))
                    print(f"completed: {row['relative_path']}")
            db.commit()
            continue
        if status in {"failed", "cancelled"}:
            error = job.get("error", {})
            message = error.get("message") if isinstance(error, dict) else None
            db.execute("UPDATE files SET status=?, error=? WHERE id=?", (status, message or f"gateway job {status}", row["id"]))
            db.commit()
            continue
        if status not in {"queued", "preprocessing", "splitting", "loading_model", "transcribing", "merging", "cancel_requested"}:
            db.execute("UPDATE files SET status='failed', error=? WHERE id=?", (f"unexpected gateway job status: {status}", row["id"]))
            db.commit()
            continue
        time.sleep(poll_seconds)


def print_status(db: sqlite3.Connection, batch_id: str) -> None:
    batch = db.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    if batch is None:
        raise ValueError(f"unknown batch: {batch_id}")
    counts = {row["status"]: row["count"] for row in db.execute("SELECT status, COUNT(*) AS count FROM files WHERE batch_id=? GROUP BY status", (batch_id,))}
    print(json.dumps({
        "batch": batch_id,
        "source": batch["source_dir"],
        "output": batch["output_dir"],
        "model": batch["model"],
        "purpose": batch["purpose"],
        "response_format": batch["response_format"],
        "timestamps": batch["timestamps"],
        "speaker_resolution": batch["speaker_resolution"],
        "quality_profile": batch["quality_profile"],
        "max_new_tokens": batch["max_new_tokens"],
        "cancelled": bool(batch["cancelled"]),
        "counts": counts,
    }, ensure_ascii=False, indent=2))


def batch_exit_code(db: sqlite3.Connection, batch_id: str) -> int:
    batch = db.execute("SELECT cancelled FROM batches WHERE id=?", (batch_id,)).fetchone()
    if batch is None:
        raise ValueError(f"unknown batch: {batch_id}")
    incomplete = db.execute(
        "SELECT COUNT(*) FROM files WHERE batch_id=? AND status!='completed'",
        (batch_id,),
    ).fetchone()[0]
    return 1 if batch["cancelled"] or incomplete else 0


def cancel_batch(db: sqlite3.Connection, batch_id: str) -> None:
    batch = db.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    if batch is None:
        raise ValueError(f"unknown batch: {batch_id}")
    db.execute("UPDATE batches SET cancelled=1 WHERE id=?", (batch_id,))
    rows = db.execute("SELECT id, job_id FROM files WHERE batch_id=? AND status='submitted'", (batch_id,)).fetchall()
    for row in rows:
        try:
            request_json(batch["base_url"], "DELETE", f"/v1/jobs/{row['job_id']}")
        except ApiError as exc:
            print(f"could not cancel {row['job_id']}: {exc}", file=sys.stderr)
    db.execute("UPDATE files SET status='cancelled', error='batch cancelled' WHERE batch_id=? AND status='pending'", (batch_id,))
    db.commit()
    print(f"cancellation requested: {batch_id}")


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, default=DEFAULT_DB)
    root = argparse.ArgumentParser(description="Batch local files through the LAN ASR gateway.")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", parents=[common])
    run.add_argument("source", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--include", action="append", default=[])
    run.add_argument("--base-url", help="Server URL; overrides ASR_BASE_URL and the optional client config.json")
    run.add_argument("--model")
    run.add_argument("--language", default="auto")
    run.add_argument("--purpose", choices=("transcription", "oral_analysis", "podcast"), default="transcription")
    run.add_argument("--response-format", choices=("json", "verbose_json"))
    run.add_argument("--timestamps", choices=("none", "auto", "word", "char"))
    run.add_argument("--speaker-resolution", choices=("off", "auto", "required"))
    run.add_argument("--quality-profile", choices=("legacy", "quality_v1"), default="legacy")
    run.add_argument("--max-new-tokens", type=int)
    run.add_argument("--context", default="")
    run.add_argument("--hotwords", default="")
    run.add_argument("--poll-seconds", type=float, default=2.0)
    resume = commands.add_parser("resume", parents=[common])
    resume.add_argument("batch_id")
    resume.add_argument("--poll-seconds", type=float, default=2.0)
    status = commands.add_parser("status", parents=[common])
    status.add_argument("--batch", required=True, dest="batch_id")
    cancel = commands.add_parser("cancel", parents=[common])
    cancel.add_argument("batch_id")
    return root


def main() -> int:
    args = parser().parse_args()
    poll_seconds = getattr(args, "poll_seconds", 2.0)
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("--poll-seconds must be finite and positive")
    db = open_db(args.db)
    try:
        if args.command == "run":
            batch_id = create_batch(db, args)
            process_batch(db, batch_id, args.poll_seconds)
            return batch_exit_code(db, batch_id)
        elif args.command == "resume":
            process_batch(db, args.batch_id, args.poll_seconds)
            return batch_exit_code(db, args.batch_id)
        elif args.command == "status":
            print_status(db, args.batch_id)
        else:
            cancel_batch(db, args.batch_id)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ApiError, RuntimeError, ValueError, OSError, sqlite3.Error) as exc:
        print(f"lan-asr: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
