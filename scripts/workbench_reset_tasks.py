"""Remove workbench tasks offline, preserving accounts and the public glossary.

Run the default dry run first. Use the existing release backup helper before
--apply; rollback needs the matching old source and database together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from asr_server.errors import AsrError
from asr_server.workbench.store import SCHEMA_VERSION, WorkbenchStore
from scripts.workbench_release import ReleaseError, offline_lock


TASK_TABLES = ("versions", "job_chunks", "legacy_job_payloads", "uploads", "jobs", "assets", "capacity")
OWNED_DIRECTORIES = ("jobs", "uploads", "work", "temporary", "operations")


def identities(db: sqlite3.Connection) -> dict[str, str]:
    """Compare complete rows without printing credentials or glossary content."""
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    result = {}
    for table in ("users", "sessions", "metadata"):
        rows = db.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall() if table in tables else []
        result[table] = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return result


def reset_tasks(data_dir: Path, *, apply: bool = False) -> dict[str, Any]:
    data_dir = data_dir.absolute()
    if data_dir.is_symlink() or not data_dir.is_dir():
        raise ReleaseError("workbench data directory must be an existing real directory")
    data_dir = data_dir.resolve()
    database = data_dir / "workbench.sqlite3"
    if database.is_symlink() or not database.is_file():
        raise ReleaseError("workbench database is missing or is a symlink")
    with offline_lock(data_dir), closing(sqlite3.connect(database)) as db:
        schema = int(db.execute("PRAGMA user_version").fetchone()[0])
        if schema > SCHEMA_VERSION:
            raise ReleaseError("database belongs to a newer workbench version")
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ReleaseError("database integrity check failed")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        before = identities(db)
        counts = {table: int(db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]) if table in tables else 0 for table in TASK_TABLES}
        files: list[Path] = []
        directories = []
        for name in OWNED_DIRECTORIES:
            root = data_dir / name
            if root.is_symlink():
                raise ReleaseError(f"owned data path is a symlink: {name}")
            if not root.exists():
                continue
            if not root.is_dir():
                raise ReleaseError(f"owned data path is not a directory: {name}")
            directories.append(root)
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise ReleaseError(f"owned data contains a symlink: {path.relative_to(data_dir)}")
                if path.is_file():
                    files.append(path)
        seen: set[tuple[int, int]] = set()
        size = 0
        for path in files:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                size += stat.st_size
                seen.add(identity)
        summary: dict[str, Any] = {"applied": False, "schema_before": schema,
                                   "task_rows": counts, "files": len(files), "file_bytes": size,
                                   "preserved_identity": before}
        if not apply:
            return summary
        # Files disappear before their accounting. A failed deletion keeps the
        # database references and permits an explicit rerun of the cleanup.
        for directory in directories:
            shutil.rmtree(directory)
        with db:
            db.execute("PRAGMA recursive_triggers=ON")
            for table in TASK_TABLES:
                if table in tables:
                    db.execute(f"DELETE FROM {table}")
            if "sqlite_sequence" in tables:
                db.execute("DELETE FROM sqlite_sequence WHERE name='jobs'")
            if identities(db) != before:
                raise ReleaseError("account, session or glossary identity changed; database cleanup rolled back")
        store = WorkbenchStore(data_dir)
        try:
            store.initialize()
            if store.storage_usage()["allocated_bytes"] != 0:
                raise ReleaseError("capacity ledger did not reset to zero")
        finally:
            store.close()
        # Make deleted private payloads unavailable in free SQLite pages/WAL.
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.execute("VACUUM")
        if identities(db) != before:
            raise ReleaseError("preserved identities differ after schema upgrade")
        after_counts = {table: int(db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]) for table in TASK_TABLES}
        if any(after_counts.values()) or any(directory.exists() for directory in directories):
            raise ReleaseError("task cleanup verification failed")
        return {**summary, "applied": True, "schema_after": SCHEMA_VERSION,
                "task_rows_after": after_counts, "allocated_bytes": 0,
                "preserved_identities_match": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline workbench task cleanup; accounts and glossary are preserved")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="delete all workbench tasks after the offline dry run")
    args = parser.parse_args()
    try:
        print(json.dumps(reset_tasks(args.data_dir, apply=args.apply), ensure_ascii=False, indent=2))
    except (ReleaseError, AsrError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
