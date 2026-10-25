"""Build local releases and verified, offline workbench data snapshots.

Backups contain account password hashes and retained media. Keep the complete
backup outside Git and restore it with the matching source release recorded in
manifest.json. This helper never contacts or restarts a running deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import tarfile
import tempfile
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator, cast


SOURCE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "manifest.json"
SOURCE_DIRECTORIES = (
    "asr_server", "asr_mcp", "scripts", "tests", "docs", "requirements", "deploy",
    "prompts", "assets/readme", "evaluation", "frontend/src", "frontend/dist", "skills/hcds-lan-asr",
)
SOURCE_FILES = (
    "AGENTS.md", "CLAUDE.md", "README.md", "pyproject.toml", "uv.lock", ".env.example", ".gitignore",
    "frontend/package.json", "frontend/package-lock.json", "frontend/pnpm-lock.yaml",
    "frontend/index.html", "frontend/tsconfig.json", "frontend/tsconfig.app.json",
    "frontend/tsconfig.node.json", "frontend/vite.config.ts",
    "test-fixtures/audio/test_short.wav", "test-fixtures/audio/README.md",
)
EXCLUDED_PARTS = {".git", "node_modules", "__pycache__", ".venv", ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache", "logs", "secrets"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pem", ".key"}


class ReleaseError(Exception):
    pass


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def permitted(path: Path) -> bool:
    return not (
        set(path.parts) & EXCLUDED_PARTS
        or path.as_posix() == "skills/hcds-lan-asr/server.json"
        or path.suffix in EXCLUDED_SUFFIXES
        or (path.name.startswith(".env") and path.name != ".env.example")
        or path.name in {".DS_Store", "service.lock", "token", "token.txt", "tokens.json", "secrets.json", "credentials.json", ".npmrc", ".netrc"}
    )


def source_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for relative in SOURCE_FILES:
        path = root / relative
        if path.is_file():
            files.add(path)
    for relative in SOURCE_DIRECTORIES:
        directory = root / relative
        if directory.is_dir():
            files.update(path for path in directory.rglob("*") if path.is_file() and permitted(path.relative_to(root)))
    for path in files:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ReleaseError(f"source path leaves project directory: {path.relative_to(root)}")
        if path.is_symlink() and path.name != "CLAUDE.md":
            raise ReleaseError(f"source symlink is not allowed: {path.relative_to(root)}")
    return sorted(files)


def source_identity(root: Path) -> dict[str, Any]:
    files = {str(path.relative_to(root)): digest(path) for path in source_files(root)}
    source_digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    commit: str | None = None
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            commit = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {"commit": commit, "source_digest": source_digest,
            "instructions": "Restore this data with the source release matching source_digest; commit alone may omit local changes."}


def file_record(path: Path) -> dict[str, Any]:
    return {"sha256": digest(path), "size_bytes": path.stat().st_size, "mode": stat.S_IMODE(path.stat().st_mode)}


def manifest(kind: str, root: Path, files: list[Path]) -> dict[str, Any]:
    return {"format_version": 1, "kind": kind, "created_at": datetime.now(timezone.utc).isoformat(),
            "files": {str(path.relative_to(root)): file_record(path) for path in sorted(files)}}


def ensure_new(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ReleaseError("output already exists; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)


def bundle(output: Path, source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    output = output.resolve()
    ensure_new(output)
    if not (source_root / "frontend/dist/index.html").is_file():
        raise ReleaseError("frontend/dist/index.html is missing; build the frontend before bundling")
    files = source_files(source_root)
    info = manifest("source_release", source_root, files)
    info["source"] = source_identity(source_root)
    with tempfile.NamedTemporaryFile(prefix=".asr-release-", dir=output.parent, delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        with tarfile.open(staging, "w:gz") as archive:
            for path in files:
                # Dereference only the known, internal CLAUDE.md compatibility link.
                data = path.read_bytes()
                entry = tarfile.TarInfo(str(path.relative_to(source_root)))
                entry.size = len(data)
                entry.mode = stat.S_IMODE(path.stat().st_mode)
                entry.mtime = int(path.stat().st_mtime)
                import io
                archive.addfile(entry, io.BytesIO(data))
            import io
            data = (json.dumps(info, ensure_ascii=False, indent=2) + "\n").encode()
            entry = tarfile.TarInfo(MANIFEST)
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))
        # Hard-link publication fails atomically if another process won the name.
        os.link(staging, output)
    finally:
        staging.unlink(missing_ok=True)
    return {"archive": str(output), "sha256": digest(output), "files": len(files), "source": info["source"]}


@contextmanager
def offline_lock(data_dir: Path) -> Iterator[None]:
    import fcntl
    lock: BinaryIO = (data_dir / "service.lock").open("a+b")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReleaseError("workbench service is active; stop it before taking an offline backup") from exc
        yield
    finally:
        lock.close()


def data_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory_name in ("jobs", "uploads"):
        directory = root / directory_name
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if path.is_symlink():
                raise ReleaseError(f"data symlink is not allowed: {relative}")
            if path.is_file() and permitted(relative) and "work" not in relative.parts:
                files.append(path)
    return files


def backup(data_dir: Path, output: Path, source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    data_dir, output = data_dir.resolve(), output.resolve()
    if not (data_dir / "workbench.sqlite3").is_file():
        raise ReleaseError("data directory has no workbench.sqlite3")
    if output.is_relative_to(data_dir):
        raise ReleaseError("backup output must be outside the service data directory")
    ensure_new(output)
    staging = Path(tempfile.mkdtemp(prefix=".asr-backup-", dir=output.parent))
    try:
        with offline_lock(data_dir):
            with closing(sqlite3.connect((data_dir / "workbench.sqlite3").as_uri() + "?mode=ro", uri=True)) as source:
                with closing(sqlite3.connect(staging / "workbench.sqlite3")) as target:
                    source.backup(target)
                    if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise ReleaseError("SQLite backup integrity check failed")
                    target.execute("PRAGMA journal_mode=DELETE").fetchone()
            shutil.copystat(data_dir / "workbench.sqlite3", staging / "workbench.sqlite3")
            for path in data_files(data_dir):
                target_path = staging / path.relative_to(data_dir)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target_path)
            files = [path for path in staging.rglob("*") if path.is_file()]
            info = manifest("workbench_backup", staging, files)
            info["source"] = source_identity(source_root)
            (staging / MANIFEST).write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            # rename does not replace an existing nonempty directory. A sibling
            # staging directory also guarantees a single filesystem commit.
            ensure_new(output)
            staging.rename(output)
        return {"backup": str(output), "files": len(files), "source": info["source"]}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def read_manifest(backup_dir: Path) -> dict[str, Any]:
    path = backup_dir / MANIFEST
    if not path.is_file() or path.is_symlink():
        raise ReleaseError("backup manifest is missing")
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ReleaseError("backup manifest is invalid") from exc
    if not isinstance(info, dict) or info.get("format_version") != 1 or info.get("kind") != "workbench_backup" or not isinstance(info.get("files"), dict):
        raise ReleaseError("unsupported backup manifest")
    return info


def verify_backup(backup_dir: Path) -> dict[str, Any]:
    backup_dir = backup_dir.resolve()
    info = read_manifest(backup_dir)
    if "workbench.sqlite3" not in info["files"]:
        raise ReleaseError("backup database is missing from manifest")
    for relative, expected in info["files"].items():
        relative_path = Path(relative)
        path = backup_dir / relative_path
        if relative_path.is_absolute() or ".." in relative_path.parts or not path.resolve().is_relative_to(backup_dir.resolve()):
            raise ReleaseError("backup manifest contains an unsafe path")
        if relative_path.parts[0] not in {"jobs", "uploads", "workbench.sqlite3"} or not permitted(relative_path):
            raise ReleaseError("backup manifest contains an unexpected path")
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != backup_dir.parent):
            raise ReleaseError("backup symlinks are not allowed")
        if not path.is_file() or not isinstance(expected, dict):
            raise ReleaseError(f"backup file is missing: {relative}")
        if not isinstance(expected.get("mode"), int) or not 0 <= expected["mode"] <= 0o7777:
            raise ReleaseError(f"backup permissions are invalid: {relative}")
        if digest(path) != expected.get("sha256") or path.stat().st_size != expected.get("size_bytes"):
            raise ReleaseError(f"backup file hash mismatch: {relative}")
    with closing(sqlite3.connect((backup_dir / "workbench.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ReleaseError("backup database integrity check failed")
    return info


def restore(backup_dir: Path, data_dir: Path, source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    backup_dir, data_dir = backup_dir.resolve(), data_dir.absolute()
    if data_dir.is_symlink():
        raise ReleaseError("restore destination must not be a symlink")
    if data_dir.exists() and (not data_dir.is_dir() or any(data_dir.iterdir())):
        raise ReleaseError("restore destination must be new or empty")
    if data_dir.resolve().is_relative_to(backup_dir) or backup_dir.is_relative_to(data_dir.resolve()):
        raise ReleaseError("restore destination and backup must not overlap")
    info = verify_backup(backup_dir)
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".asr-restore-", dir=data_dir.parent))
    try:
        for relative, expected in info["files"].items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_dir / relative, target)
            target.chmod(int(expected["mode"]) & 0o777)
        (staging / "restored-source.json").write_text(json.dumps(info.get("source"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if data_dir.exists():
            data_dir.rmdir()
        staging.rename(data_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    current_source = source_identity(source_root)
    return {"data_dir": str(data_dir), "files": len(info["files"]), "required_source": info.get("source"),
            "source_matches": info.get("source", {}).get("source_digest") == current_source["source_digest"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ASR source releases and offline, verified workbench backups")
    commands = parser.add_subparsers(dest="command", required=True)
    package_parser = commands.add_parser("bundle")
    package_parser.add_argument("--output", type=Path, required=True)
    backup_parser = commands.add_parser("backup")
    backup_parser.add_argument("--data-dir", type=Path, required=True)
    backup_parser.add_argument("--output", type=Path, required=True)
    restore_parser = commands.add_parser("restore")
    restore_parser.add_argument("--backup", type=Path, required=True)
    restore_parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "bundle":
            result = bundle(args.output)
        elif args.command == "backup":
            result = backup(args.data_dir, args.output)
        else:
            result = restore(args.backup, args.data_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ReleaseError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
