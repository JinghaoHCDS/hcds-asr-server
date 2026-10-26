from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, ParamSpec, TypeVar, cast
from uuid import uuid4

from asr_server.errors import AsrError
from asr_server.workbench.executor import BoundedExecutor
from asr_server.workbench.security import password_hash, token_hash


P = ParamSpec("P")
T = TypeVar("T")
SCHEMA_VERSION = 4


def _summary_sql(payload: str) -> str:
    fields = ["id", "filename", "status", "created_at", "started_at", "completed_at", "expires_at", "model_revision"]
    pairs = [f"'{field}',json_extract({payload},'$.{field}')" for field in fields]
    pairs += [
        f"'model',json_extract({payload},'$.request.model')",
        f"'audio_size_bytes',coalesce(json_extract({payload},'$.audio_asset.size_bytes'),0)",
        "'audio_asset',json_object(" + ",".join(f"'{key}',json_extract({payload},'$.audio_asset.{key}')" for key in
            ["state", "codec", "bitrate_bps", "sample_rate", "channels", "size_bytes", "duration_seconds", "source_stream_index", "source_audio_streams"]) + ")",
        f"'has_result',json_type({payload},'$.result')='object'",
        f"'alignment_status',json_extract({payload},'$.result.alignment_status')",
        f"'version',json_extract({payload},'$.result.version')",
        "'progress',json_object(" + ",".join(f"'{key}',json_extract({payload},'$.progress.{key}')" for key in
            ["phase", "percent", "completed_chunks", "total_chunks", "current_chunk"]) + ")",
        f"'error',CASE WHEN json_type({payload},'$.error')='object' THEN json_object('code',json_extract({payload},'$.error.code'),'message',json_extract({payload},'$.error.message')) ELSE NULL END",
    ]
    return "json_object(" + ",".join(pairs) + ")"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def after_days(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def identifier(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class WorkbenchStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "workbench.sqlite3"
        self.initialized = False
        self._initialize_lock = threading.RLock()
        self.executor = BoundedExecutor(4, 32, "asr-repository", drain_on_cancel=True)

    async def run(self, operation: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Run a complete repository operation, including opening its connection."""
        return await self.executor.run(operation, *args, **kwargs)

    def initialize(self) -> None:
        with self._initialize_lock:
            self._initialize()

    def _initialize(self) -> None:
        if self.initialized:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            schema = int(db.execute("PRAGMA user_version").fetchone()[0])
            if schema > SCHEMA_VERSION:
                raise AsrError(409, "schema_version_unavailable", "database requires a newer workbench version")
            if schema < SCHEMA_VERSION:
                tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if any(table in tables and db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                       for table in ("jobs", "uploads", "versions", "job_chunks", "legacy_job_payloads", "assets")):
                    raise AsrError(409, "schema_reset_required",
                                   "old workbench tasks require offline cleanup before schema 4; run scripts.workbench_reset_tasks")
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL, is_admin INTEGER NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, filename TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL, offset INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, expires_at TEXT NOT NULL, job_id TEXT
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE, owner_id TEXT NOT NULL,
                    status TEXT NOT NULL, reserved_bytes INTEGER NOT NULL,
                    payload TEXT NOT NULL, idempotency_key TEXT, fingerprint TEXT,
                    UNIQUE(owner_id,idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS versions (
                    job_id TEXT NOT NULL, version INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(job_id,version)
                );
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner_id,sequence);
            """)
            db.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)", ("glossary", json.dumps({
                "version": 1,
                "entries": [
                    {"canonical": "Qwen3-ASR", "aliases": [], "domain": "ai", "auto_normalize": False},
                    {"canonical": "ChatGPT", "aliases": ["chatgpt"], "domain": "ai", "auto_normalize": True},
                    {"canonical": "OpenAI", "aliases": ["openai"], "domain": "ai", "auto_normalize": True},
                    {"canonical": "API", "aliases": [], "domain": "general", "auto_normalize": False},
                ],
            }, ensure_ascii=False)))
            if schema < SCHEMA_VERSION:
                self._migrate(db)
        self.initialized = True

    def close(self) -> None:
        self.executor.close()

    def _migrate(self, db: sqlite3.Connection) -> None:
        """Upgrade an empty task store while preserving accounts and glossary."""
        for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
            db.execute(f'DROP TRIGGER "{name}"')
        columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
        for name, definition in {"summary": "TEXT NOT NULL DEFAULT '{}'", "revision": "INTEGER NOT NULL DEFAULT 1"}.items():
            if name not in columns:
                db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")
        db.execute("CREATE INDEX IF NOT EXISTS jobs_status_sequence ON jobs(status,sequence)")
        db.execute("CREATE TABLE IF NOT EXISTS capacity (owner_id TEXT PRIMARY KEY, logical_bytes INTEGER NOT NULL DEFAULT 0, physical_bytes INTEGER NOT NULL DEFAULT 0)")
        db.execute("CREATE TABLE IF NOT EXISTS assets (asset_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, physical_bytes INTEGER NOT NULL)")
        for table, additions in {
            "capacity": {"media_budget_bytes": "INTEGER NOT NULL DEFAULT 0", "temporary_bytes": "INTEGER NOT NULL DEFAULT 0", "text_file_bytes": "INTEGER NOT NULL DEFAULT 0"},
            "assets": {"temporary_bytes": "INTEGER NOT NULL DEFAULT 0", "text_file_bytes": "INTEGER NOT NULL DEFAULT 0"},
        }.items():
            current = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
            for name, definition in additions.items():
                if name not in current:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
        db.execute("CREATE INDEX IF NOT EXISTS assets_owner ON assets(owner_id)")
        db.execute("CREATE TABLE IF NOT EXISTS job_chunks (job_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(job_id,chunk_index))")
        db.execute("CREATE TABLE IF NOT EXISTS legacy_job_payloads (job_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, payload TEXT NOT NULL)")
        db.execute(f"UPDATE jobs SET summary={_summary_sql('payload')}")
        db.execute("DELETE FROM capacity")
        db.execute("INSERT INTO capacity(owner_id) SELECT owner_id FROM jobs UNION SELECT owner_id FROM uploads")
        db.execute("UPDATE capacity SET logical_bytes="
            "coalesce((SELECT sum(reserved_bytes+length(CAST(payload AS BLOB))) FROM jobs WHERE owner_id=capacity.owner_id),0)+"
            "coalesce((SELECT sum(size_bytes) FROM uploads WHERE owner_id=capacity.owner_id AND job_id IS NULL),0)+"
            "coalesce((SELECT sum(length(CAST(v.payload AS BLOB))) FROM versions v JOIN jobs j ON j.id=v.job_id WHERE j.owner_id=capacity.owner_id),0)+"
            "coalesce((SELECT sum(length(CAST(c.payload AS BLOB))) FROM job_chunks c JOIN jobs j ON j.id=c.job_id WHERE j.owner_id=capacity.owner_id),0)+"
            "coalesce((SELECT sum(length(CAST(payload AS BLOB))) FROM legacy_job_payloads WHERE owner_id=capacity.owner_id),0),"
            "physical_bytes=coalesce((SELECT sum(physical_bytes) FROM assets WHERE owner_id=capacity.owner_id),0),"
            "temporary_bytes=coalesce((SELECT sum(temporary_bytes) FROM assets WHERE owner_id=capacity.owner_id),0),"
            "text_file_bytes=coalesce((SELECT sum(text_file_bytes) FROM assets WHERE owner_id=capacity.owner_id),0),"
            "media_budget_bytes=coalesce((SELECT sum(reserved_bytes) FROM jobs WHERE owner_id=capacity.owner_id),0)+"
            "coalesce((SELECT sum(size_bytes) FROM uploads WHERE owner_id=capacity.owner_id AND job_id IS NULL),0)")
        def trigger(name: str, when: str, body: str) -> None:
            db.execute(f"CREATE TRIGGER IF NOT EXISTS {name} {when} BEGIN {body} END")
        trigger("jobs_summary_insert", "AFTER INSERT ON jobs", f"UPDATE jobs SET summary={_summary_sql('new.payload')} WHERE id=new.id;")
        trigger("jobs_summary_update", "AFTER UPDATE OF payload,status ON jobs", f"UPDATE jobs SET summary={_summary_sql('new.payload')},revision=old.revision+1 WHERE id=new.id;")
        trigger("jobs_capacity_insert", "AFTER INSERT ON jobs", "INSERT INTO capacity(owner_id,logical_bytes,media_budget_bytes) VALUES(new.owner_id,new.reserved_bytes+length(CAST(new.payload AS BLOB)),new.reserved_bytes) ON CONFLICT(owner_id) DO UPDATE SET logical_bytes=logical_bytes+excluded.logical_bytes,media_budget_bytes=media_budget_bytes+excluded.media_budget_bytes;")
        trigger("jobs_capacity_update", "AFTER UPDATE OF payload,reserved_bytes ON jobs", "UPDATE capacity SET logical_bytes=logical_bytes+new.reserved_bytes-old.reserved_bytes+length(CAST(new.payload AS BLOB))-length(CAST(old.payload AS BLOB)),media_budget_bytes=media_budget_bytes+new.reserved_bytes-old.reserved_bytes WHERE owner_id=new.owner_id;")
        trigger("jobs_capacity_delete", "AFTER DELETE ON jobs", "UPDATE capacity SET logical_bytes=logical_bytes-old.reserved_bytes-length(CAST(old.payload AS BLOB)),media_budget_bytes=media_budget_bytes-old.reserved_bytes WHERE owner_id=old.owner_id;")
        trigger("uploads_capacity_insert", "AFTER INSERT ON uploads", "INSERT INTO capacity(owner_id,logical_bytes,media_budget_bytes) VALUES(new.owner_id,CASE WHEN new.job_id IS NULL THEN new.size_bytes ELSE 0 END,CASE WHEN new.job_id IS NULL THEN new.size_bytes ELSE 0 END) ON CONFLICT(owner_id) DO UPDATE SET logical_bytes=logical_bytes+excluded.logical_bytes,media_budget_bytes=media_budget_bytes+excluded.media_budget_bytes;")
        trigger("uploads_capacity_update", "AFTER UPDATE OF size_bytes,job_id ON uploads", "UPDATE capacity SET logical_bytes=logical_bytes+(CASE WHEN new.job_id IS NULL THEN new.size_bytes ELSE 0 END)-(CASE WHEN old.job_id IS NULL THEN old.size_bytes ELSE 0 END),media_budget_bytes=media_budget_bytes+(CASE WHEN new.job_id IS NULL THEN new.size_bytes ELSE 0 END)-(CASE WHEN old.job_id IS NULL THEN old.size_bytes ELSE 0 END) WHERE owner_id=new.owner_id;")
        trigger("uploads_capacity_delete", "AFTER DELETE ON uploads", "UPDATE capacity SET logical_bytes=logical_bytes-(CASE WHEN old.job_id IS NULL THEN old.size_bytes ELSE 0 END),media_budget_bytes=media_budget_bytes-(CASE WHEN old.job_id IS NULL THEN old.size_bytes ELSE 0 END) WHERE owner_id=old.owner_id;")
        trigger("versions_capacity_insert", "AFTER INSERT ON versions", "UPDATE capacity SET logical_bytes=logical_bytes+length(CAST(new.payload AS BLOB)) WHERE owner_id=(SELECT owner_id FROM jobs WHERE id=new.job_id);")
        trigger("versions_capacity_delete", "AFTER DELETE ON versions", "UPDATE capacity SET logical_bytes=logical_bytes-length(CAST(old.payload AS BLOB)) WHERE owner_id=(SELECT owner_id FROM jobs WHERE id=old.job_id);")
        trigger("versions_capacity_update", "AFTER UPDATE OF payload ON versions", "UPDATE capacity SET logical_bytes=logical_bytes+length(CAST(new.payload AS BLOB))-length(CAST(old.payload AS BLOB)) WHERE owner_id=(SELECT owner_id FROM jobs WHERE id=new.job_id);")
        trigger("chunks_capacity_insert", "AFTER INSERT ON job_chunks", "UPDATE capacity SET logical_bytes=logical_bytes+length(CAST(new.payload AS BLOB)) WHERE owner_id=(SELECT owner_id FROM jobs WHERE id=new.job_id);")
        trigger("chunks_capacity_update", "AFTER UPDATE OF payload ON job_chunks", "UPDATE capacity SET logical_bytes=logical_bytes+length(CAST(new.payload AS BLOB))-length(CAST(old.payload AS BLOB)) WHERE owner_id=(SELECT owner_id FROM jobs WHERE id=new.job_id);")
        trigger("chunks_capacity_delete", "AFTER DELETE ON job_chunks", "UPDATE capacity SET logical_bytes=logical_bytes-length(CAST(old.payload AS BLOB)) WHERE owner_id=(SELECT owner_id FROM jobs WHERE id=old.job_id);")
        trigger("legacy_capacity_insert", "AFTER INSERT ON legacy_job_payloads", "INSERT INTO capacity(owner_id,logical_bytes) VALUES(new.owner_id,length(CAST(new.payload AS BLOB))) ON CONFLICT(owner_id) DO UPDATE SET logical_bytes=logical_bytes+excluded.logical_bytes;")
        trigger("legacy_capacity_delete", "AFTER DELETE ON legacy_job_payloads", "UPDATE capacity SET logical_bytes=logical_bytes-length(CAST(old.payload AS BLOB)) WHERE owner_id=old.owner_id;")
        trigger("jobs_owned_data_delete", "BEFORE DELETE ON jobs", "DELETE FROM versions WHERE job_id=old.id; DELETE FROM job_chunks WHERE job_id=old.id; DELETE FROM legacy_job_payloads WHERE job_id=old.id;")
        # REPLACE must fire delete triggers, including historical version updates.
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.initialize()
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA recursive_triggers=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_user(self, username: str, password: str, is_admin: bool = False) -> dict[str, Any]:
        username = username.strip()
        if not re.fullmatch(r"[\w.@-]{2,64}", username):
            raise AsrError(400, "bad_request", "username must contain 2 to 64 letters, digits, . @ - or _")
        hashed = password_hash(password)
        user = {"id": identifier("user"), "username": username, "is_admin": is_admin,
                "disabled": False, "created_at": now()}
        try:
            with self.connection() as db:
                db.execute("INSERT INTO users VALUES (?,?,?,?,?,?)", (
                    user["id"], username, hashed, int(is_admin), 0, user["created_at"],
                ))
        except sqlite3.IntegrityError as exc:
            raise AsrError(409, "username_exists", "username already exists") from exc
        return user

    @staticmethod
    def public_user(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        return {"id": row["id"], "username": row["username"], "is_admin": bool(row["is_admin"]),
                "disabled": bool(row["disabled"]), "created_at": row["created_at"]}

    def authenticate_session(self, token: str) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT u.* FROM users u JOIN sessions s ON u.id=s.user_id "
                             "WHERE s.token_hash=? AND s.expires_at>? AND u.disabled=0", (token_hash(token), now())).fetchone()
        if row is None:
            raise AsrError(401, "unauthorized", "sign in to the workbench")
        return self.public_user(row)

    def user_for_login(self, username: str) -> dict[str, Any] | None:
        with self.connection() as db:
            row = db.execute("SELECT * FROM users WHERE username=?", (username.strip(),)).fetchone()
        return dict(row) if row is not None else None

    def session(self, token: str, user_id: str | None = None, expected_hash: str | None = None) -> None:
        with self.connection() as db:
            if user_id is None:
                db.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash(token),))
            else:
                # Recheck account state after password work and before session creation.
                row = db.execute("SELECT disabled,password_hash FROM users WHERE id=?", (user_id,)).fetchone()
                if row is None or row[0] or (expected_hash is not None and row[1] != expected_hash):
                    raise AsrError(401, "unauthorized", "incorrect username or password")
                db.execute("INSERT INTO sessions VALUES (?,?,?)", (token_hash(token), user_id, after_days(7)))

    def users(self) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [self.public_user(row) for row in rows]

    def update_user(self, user_id: str, hashed_password: str | None, disabled: bool | None) -> dict[str, Any]:
        with self.connection() as db:
            if db.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone() is None:
                raise AsrError(404, "user_not_found", "user not found")
            if hashed_password is not None:
                db.execute("UPDATE users SET password_hash=? WHERE id=?", (hashed_password, user_id))
            if disabled is not None:
                db.execute("UPDATE users SET disabled=? WHERE id=?", (int(disabled), user_id))
            if hashed_password is not None or disabled:
                db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
            row = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        assert row is not None
        return self.public_user(row)

    def update_glossary(self, entries: list[dict[str, Any]], base_version: int | None) -> dict[str, Any]:
        with self.connection() as db:
            current = json.loads(db.execute("SELECT payload FROM metadata WHERE key='glossary'").fetchone()[0])
            if base_version is not None and current["version"] != base_version:
                raise AsrError(409, "version_conflict", "glossary changed; refresh before saving")
            updated = {"version": current["version"] + 1, "entries": entries}
            db.execute("UPDATE metadata SET payload=? WHERE key='glossary'", (json.dumps(updated, ensure_ascii=False),))
        return updated

    def jobs(self, owner_id: str | None = None) -> list[dict[str, Any]]:
        with self.connection() as db:
            if owner_id is None:
                rows = db.execute("SELECT payload,revision FROM jobs ORDER BY sequence").fetchall()
            else:
                rows = db.execute("SELECT payload,revision FROM jobs WHERE owner_id=? ORDER BY sequence DESC", (owner_id,)).fetchall()
            return [self._hydrate(db, json.loads(row["payload"]), row["revision"]) for row in rows]

    def job_summaries(self, owner_id: str | None = None, *, limit: int = 50,
                      before: int | None = None, active_only: bool = False,
                      job_ids: list[str] | None = None) -> dict[str, Any]:
        if not 1 <= limit <= 100:
            raise AsrError(400, "bad_request", "summary page size must be between 1 and 100")
        conditions: list[str] = []
        args: list[Any] = []
        if owner_id is not None:
            conditions.append("owner_id=?")
            args.append(owner_id)
        if before is not None:
            conditions.append("sequence<?")
            args.append(before)
        if active_only:
            conditions.append("(status NOT IN ('completed','failed','cancelled') OR json_extract(summary,'$.alignment_status')='pending')")
        if job_ids is not None:
            if not job_ids or len(job_ids) > 100:
                raise AsrError(400, "bad_request", "supply between 1 and 100 task identifiers")
            conditions.append("id IN (" + ",".join("?" for _ in job_ids) + ")")
            args.extend(job_ids)
        clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        with self.connection() as db:
            rows = db.execute("SELECT sequence,summary,revision FROM jobs" + clause + " ORDER BY sequence DESC LIMIT ?", [*args, limit + 1]).fetchall()
        visible = rows[:limit]
        jobs = [{**json.loads(row["summary"]), "revision": row["revision"]} for row in visible]
        return {"jobs": jobs, "next_cursor": visible[-1]["sequence"] if len(rows) > limit else None}

    def job_revision(self, job_id: str, owner_id: str) -> int:
        with self.connection() as db:
            row = db.execute("SELECT revision FROM jobs WHERE id=? AND owner_id=?", (job_id, owner_id)).fetchone()
        if row is None:
            raise AsrError(404, "job_not_found", "workbench job not found")
        return int(row[0])

    def job_metadata(self, job_id: str, owner_id: str | None = None) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT payload,revision FROM jobs WHERE id=?" + (" AND owner_id=?" if owner_id is not None else ""),
                             (job_id, owner_id) if owner_id is not None else (job_id,)).fetchone()
        if row is None:
            raise AsrError(404, "job_not_found", "workbench job not found")
        return {**json.loads(row["payload"]), "revision": row["revision"]}

    def pending_jobs(self) -> list[str]:
        """Maintenance first selects identifiers; completed result payloads stay on disk."""
        with self.connection() as db:
            rows = db.execute("SELECT id FROM jobs WHERE status NOT IN ('completed','failed','cancelled') OR json_extract(summary,'$.alignment_status')='pending' ORDER BY sequence").fetchall()
        return [str(row[0]) for row in rows]

    def job(self, job_id: str, owner_id: str | None = None) -> dict[str, Any]:
        with self.connection() as db:
            if owner_id is None:
                row = db.execute("SELECT payload,revision FROM jobs WHERE id=?", (job_id,)).fetchone()
            else:
                row = db.execute("SELECT payload,revision FROM jobs WHERE id=? AND owner_id=?", (job_id, owner_id)).fetchone()
            if row is None:
                raise AsrError(404, "job_not_found", "workbench job not found")
            return self._hydrate(db, json.loads(row["payload"]), row["revision"])

    @staticmethod
    def _hydrate(db: sqlite3.Connection, job: dict[str, Any], revision: int) -> dict[str, Any]:
        result = job.get("result")
        if result and result.get("_version_ref"):
            row = db.execute("SELECT payload FROM versions WHERE job_id=? AND version=?", (job["id"], result["version"])).fetchone()
            if row is None:
                raise AsrError(409, "result_not_ready", "saved transcript version is missing")
            job["result"] = json.loads(row[0])
        progress = job.get("progress") or {}
        if progress.pop("_chunks_ref", False):
            chunks = [json.loads(row[0]) for row in db.execute("SELECT payload FROM job_chunks WHERE job_id=? ORDER BY chunk_index", (job["id"],))]
            progress["chunks"] = chunks
            progress["text"] = "\n".join(str(chunk.get("text", "")) for chunk in chunks)
        job["revision"] = revision
        return job

    @staticmethod
    def _compact(db: sqlite3.Connection, job: dict[str, Any]) -> dict[str, Any]:
        compact = dict(job)
        compact.pop("revision", None)
        result = job.get("result")
        if result and not result.get("_version_ref"):
            version = int(result.get("version", 1))
            db.execute("INSERT INTO versions VALUES(?,?,?) ON CONFLICT(job_id,version) DO UPDATE SET payload=excluded.payload WHERE versions.payload!=excluded.payload", (job["id"], version, json.dumps(result, ensure_ascii=False)))
            compact["result"] = {"_version_ref": True, "version": version, "alignment_status": result.get("alignment_status")}
        progress = dict(job.get("progress") or {})
        chunks = progress.pop("chunks", None)
        if chunks is not None:
            indices = []
            for chunk in chunks:
                index = int(chunk["index"])
                indices.append(index)
                db.execute("INSERT INTO job_chunks VALUES(?,?,?) ON CONFLICT(job_id,chunk_index) DO UPDATE SET payload=excluded.payload WHERE job_chunks.payload!=excluded.payload", (job["id"], index, json.dumps(chunk, ensure_ascii=False)))
            if indices:
                db.execute("DELETE FROM job_chunks WHERE job_id=? AND chunk_index NOT IN (" + ",".join("?" for _ in indices) + ")", [job["id"], *indices])
            else:
                db.execute("DELETE FROM job_chunks WHERE job_id=?", (job["id"],))
            progress.pop("text", None)
            progress["_chunks_ref"] = True
        compact["progress"] = progress
        return compact

    def save_job(self, job: dict[str, Any], reserved_bytes: int | None = None) -> None:
        with self.connection() as db:
            compact = self._compact(db, job)
            changed = db.execute("UPDATE jobs SET status=?,payload=? WHERE id=?", (job["status"], json.dumps(compact, ensure_ascii=False), job["id"]))
            if changed.rowcount == 0:
                raise AsrError(404, "job_not_found", "workbench job not found")
            if reserved_bytes is not None:
                db.execute("UPDATE jobs SET reserved_bytes=? WHERE id=?", (reserved_bytes, job["id"]))
            sizes = job.get("retained_file_bytes")
            if isinstance(sizes, dict):
                temporary, text = self.file_categories(sizes)
                self._set_physical_usage(db, job["id"], job["owner_id"], sum(int(value) for value in sizes.values()), temporary, text)

    def save_version(self, job: dict[str, Any]) -> None:
        self.save_job(job)

    def save_chunk(self, job_id: str, chunk: dict[str, Any], total: int) -> None:
        """Publish only after the durable filesystem checkpoint has completed."""
        with self.connection() as db:
            row = db.execute("SELECT status,payload FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise AsrError(404, "job_not_found", "workbench job not found")
            job = json.loads(row["payload"])
            if row["status"] in {"completed", "failed", "cancelled", "cancel_requested"} or job.get("delete_requested") or job.get("cancel_requested"):
                return
            db.execute("INSERT INTO job_chunks VALUES(?,?,?) ON CONFLICT(job_id,chunk_index) DO UPDATE SET payload=excluded.payload", (job_id, chunk["index"], json.dumps(chunk, ensure_ascii=False)))
            count = int(db.execute("SELECT count(*) FROM job_chunks WHERE job_id=?", (job_id,)).fetchone()[0])
            job["status"] = "transcribing"
            job["progress"] = {**job.get("progress", {}), "phase": "transcribing", "_chunks_ref": True,
                               "completed_chunks": count, "total_chunks": total, "percent": count / total * 95}
            db.execute("UPDATE jobs SET status=?,payload=? WHERE id=?", (job["status"], json.dumps(job, ensure_ascii=False), job_id))

    def version(self, job_id: str, version: int) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT payload FROM versions WHERE job_id=? AND version=?", (job_id, version)).fetchone()
        if row is None:
            raise AsrError(404, "version_not_found", "transcript version not found")
        return cast(dict[str, Any], json.loads(row["payload"]))

    def storage_usage(self, owner_id: str | None = None) -> dict[str, int]:
        """Read the ledger only; filesystem repair is a separate maintenance task."""
        clause = " WHERE owner_id=?" if owner_id is not None else ""
        args = (owner_id,) if owner_id is not None else ()
        with self.connection() as db:
            row = db.execute("SELECT coalesce(sum(logical_bytes-media_budget_bytes),0),"
                             "coalesce(sum(physical_bytes),0),coalesce(sum(temporary_bytes),0),"
                             "coalesce(sum(text_file_bytes),0) FROM capacity" + clause, args).fetchone()
            job_clause = " AND j.owner_id=?" if owner_id is not None else ""
            upload_clause = " AND owner_id=?" if owner_id is not None else ""
            pending_jobs = db.execute("SELECT coalesce(sum(max(j.reserved_bytes-coalesce(a.physical_bytes,0),0)),0) "
                                      "FROM jobs j LEFT JOIN assets a ON a.asset_id=j.id WHERE 1=1" + job_clause, args).fetchone()[0]
            pending_uploads = db.execute("SELECT coalesce(sum(max(size_bytes-offset,0)),0) FROM uploads WHERE job_id IS NULL" + upload_clause, args).fetchone()[0]
        sql_text, physical, temporary, text_files = (max(0, int(value)) for value in row)
        reserved = int(pending_jobs) + int(pending_uploads)
        text = sql_text + text_files
        saved = max(0, physical-temporary-text_files)
        used = saved + text + temporary
        return {"saved_bytes": saved, "text_bytes": text, "temporary_bytes": temporary,
                "reserved_bytes": reserved, "used_bytes": used, "allocated_bytes": used + reserved}

    def usage(self, owner_id: str | None = None) -> int:
        return self.storage_usage(owner_id)["allocated_bytes"]

    def physical_usage(self, owner_id: str | None = None) -> int:
        with self.connection() as db:
            result = db.execute("SELECT coalesce(sum(physical_bytes),0) FROM capacity" + (" WHERE owner_id=?" if owner_id else ""), (owner_id,) if owner_id else ()).fetchone()[0]
        return int(result)

    def future_disk_reservations(self) -> dict[str, int]:
        """Only unwritten media promises; SQLite payloads are already materialized."""
        with self.connection() as db:
            uploads = db.execute("SELECT id,size_bytes-offset FROM uploads WHERE job_id IS NULL AND size_bytes>offset").fetchall()
            jobs = db.execute("SELECT j.id,j.reserved_bytes-coalesce(a.physical_bytes,0) FROM jobs j LEFT JOIN assets a ON a.asset_id=j.id WHERE j.reserved_bytes>coalesce(a.physical_bytes,0)").fetchall()
        return {**{"upload:" + str(row[0]): int(row[1]) for row in uploads},
                **{"job:" + str(row[0]): int(row[1]) for row in jobs}}

    @staticmethod
    def file_categories(sizes: dict[str, Any]) -> tuple[int, int]:
        temporary = text = 0
        for name, raw_size in sizes.items():
            size = max(0, int(raw_size))
            path = Path(name)
            if (path.parts and path.parts[0] in {"work", "operation", "operations", "temporary", "pcm"}) or ".partial" in path.name or path.suffix == ".wav":
                temporary += size
            elif path.suffix == ".json" or (path.parts and path.parts[0] in {"checkpoint", "checkpoints"}):
                text += size
        return temporary, text

    @staticmethod
    def _set_physical_usage(db: sqlite3.Connection, asset_id: str, owner_id: str,
                            physical_bytes: int, temporary_bytes: int, text_file_bytes: int) -> None:
        physical = max(0, physical_bytes)
        temporary = min(physical, max(0, temporary_bytes))
        text = min(physical-temporary, max(0, text_file_bytes))
        old = db.execute("SELECT owner_id,physical_bytes,temporary_bytes,text_file_bytes FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        if old:
            db.execute("UPDATE capacity SET physical_bytes=physical_bytes-?,temporary_bytes=temporary_bytes-?,text_file_bytes=text_file_bytes-? WHERE owner_id=?",
                       (old["physical_bytes"], old["temporary_bytes"], old["text_file_bytes"], old["owner_id"]))
        db.execute("INSERT OR IGNORE INTO capacity(owner_id) VALUES(?)", (owner_id,))
        if physical:
            db.execute("INSERT INTO assets(asset_id,owner_id,physical_bytes,temporary_bytes,text_file_bytes) VALUES(?,?,?,?,?) ON CONFLICT(asset_id) DO UPDATE SET owner_id=excluded.owner_id,physical_bytes=excluded.physical_bytes,temporary_bytes=excluded.temporary_bytes,text_file_bytes=excluded.text_file_bytes",
                       (asset_id, owner_id, physical, temporary, text))
            db.execute("UPDATE capacity SET physical_bytes=physical_bytes+?,temporary_bytes=temporary_bytes+?,text_file_bytes=text_file_bytes+? WHERE owner_id=?", (physical, temporary, text, owner_id))
        else:
            db.execute("DELETE FROM assets WHERE asset_id=?", (asset_id,))

    def set_physical_usage(self, asset_id: str, owner_id: str, physical_bytes: int, *,
                           temporary_bytes: int = 0, text_file_bytes: int = 0) -> None:
        """Record materialized bytes once; categories partition the same bytes."""
        with self.connection() as db:
            self._set_physical_usage(db, asset_id, owner_id, physical_bytes, temporary_bytes, text_file_bytes)

    def glossary(self) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT payload FROM metadata WHERE key='glossary'").fetchone()
        assert row is not None
        return cast(dict[str, Any], json.loads(row[0]))
