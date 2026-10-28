from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tarfile
from pathlib import Path

import pytest

from asr_server.workbench.store import WorkbenchStore
from scripts.workbench_release import ReleaseError, backup, bundle, digest, offline_lock, restore, source_identity, verify_backup


def source_fixture(root: Path) -> Path:
    (root / "asr_server").mkdir(parents=True)
    (root / "asr_server/main.py").write_text("print('release')\n")
    (root / "frontend/dist").mkdir(parents=True)
    (root / "frontend/dist/index.html").write_text("<title>ASR</title>")
    (root / "pyproject.toml").write_text('[project]\nname="asr-test"\nversion="0.1.0"\n')
    return root


def data_fixture(root: Path) -> WorkbenchStore:
    store = WorkbenchStore(root)
    store.create_user("owner", "private-password", True)
    (root / "jobs/wb_example/checkpoints").mkdir(parents=True)
    (root / "jobs/wb_example/audio.wav").write_bytes(b"retained-audio")
    (root / "jobs/wb_example/checkpoints/chunk-0.json").write_text('{"text":"原文"}', encoding="utf-8")
    (root / "uploads").mkdir()
    (root / "uploads/upload_example").write_bytes(b"partial-upload")
    return store


def test_bundle_uses_allowlist_and_hash_manifest(tmp_path: Path) -> None:
    root = source_fixture(tmp_path / "source")
    (root / ".env").write_text("SECRET=never-copy")
    (root / "token.txt").write_text("never-copy-token")
    (root / "data").mkdir()
    (root / "data/private.wav").write_bytes(b"never-copy-runtime")
    (root / "frontend/node_modules").mkdir()
    (root / "frontend/node_modules/secret.js").write_text("never-copy-module")
    (root / "asr_server/__pycache__").mkdir()
    (root / "asr_server/__pycache__/x.pyc").write_bytes(b"cache")
    (root / "asr_server/.env").write_text("nested-secret")
    (root / "skills/hcds-lan-asr/scripts").mkdir(parents=True)
    (root / "skills/hcds-lan-asr/SKILL.md").write_text("# Official Skill\n")
    (root / "skills/hcds-lan-asr/scripts/asr_batch.py").write_text("print('client')\n")
    (root / "skills/hcds-lan-asr/.env").write_text("skill-secret")
    (root / "skills/hcds-lan-asr/server.json").write_text('{"base_url":"http://private-host:18080"}')
    (root / "skills/lan-asr").mkdir()
    (root / "skills/lan-asr/SKILL.md").write_text("# Retired Skill\n")
    public_resources = (
        "LICENSE", "THIRD_PARTY_NOTICES.md", ".python-version", ".github/workflows/ci.yml", "frontend/design.md", "frontend/.gitignore",
        "test-fixtures/audio/test_short.wav", "test-fixtures/audio/README.md", "scripts/generate_test_audio.py",
        "test-fixtures/audio/test_short.reference.txt", "test-fixtures/audio/test_short.source.json",
        "examples/batch-manifest.sample.json", "examples/output-sample/index.md",
        "examples/output-sample/manifest.json",
        "examples/output-sample/items/sample/transcript.md",
        "examples/output-sample/items/sample/transcript.json",
    )
    for name in public_resources:
        resource = root / name
        resource.parent.mkdir(parents=True, exist_ok=True)
        resource.write_text("synthetic sample\n")
    license_text = "MIT License\n\nCopyright (c) 2026 Synthetic test contributors\n"
    (root / "LICENSE").write_text(license_text)
    private_materials = (
        "docs/validation-artifacts/run/summary.json", "docs/artifacts/private.md", "docs/scratch/private.md",
        "scripts/asr-transcripts/transcript.md", "assets/readme/.DS_Store",
    )
    for name in private_materials:
        resource = root / name
        resource.parent.mkdir(parents=True, exist_ok=True)
        resource.write_text("never-copy-private-material\n")
    output = tmp_path / "release.tar.gz"
    result = bundle(output, root)
    assert result["sha256"] == digest(output)
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert "frontend/dist/index.html" in names
        assert "asr_server/main.py" in names
        assert "skills/hcds-lan-asr/SKILL.md" in names
        assert "skills/hcds-lan-asr/scripts/asr_batch.py" in names
        assert "skills/hcds-lan-asr/server.json" not in names
        assert "skills/lan-asr/SKILL.md" not in names
        assert all("node_modules" not in name and "__pycache__" not in name and ".env" not in name for name in names)
        assert "token.txt" not in names
        assert "data/private.wav" not in names
        assert all(name in names for name in public_resources)
        assert not any(name in names for name in private_materials)
        license_file = archive.extractfile("LICENSE")
        assert license_file is not None
        assert license_file.read() == license_text.encode()
        manifest_file = archive.extractfile("manifest.json")
        assert manifest_file is not None
        manifest = json.loads(manifest_file.read())
        assert manifest["files"]["asr_server/main.py"]["sha256"] == digest(root / "asr_server/main.py")
        assert not any(name.startswith("/") or ".." in Path(name).parts for name in names)
    with pytest.raises(ReleaseError):
        bundle(output, root)


def test_bundle_requires_built_frontend(tmp_path: Path) -> None:
    root = source_fixture(tmp_path / "source")
    (root / "frontend/dist/index.html").unlink()
    with pytest.raises(ReleaseError, match="build the frontend"):
        bundle(tmp_path / "missing.tar.gz", root)


def test_bundle_blocks_references_to_excluded_review_pending_material(tmp_path: Path) -> None:
    root = source_fixture(tmp_path / "source")
    (root / "README.md").write_text("# Package\n[Evidence](docs/validation-artifacts/run/raw.json)\n")
    raw = root / "docs/validation-artifacts/run/raw.json"
    raw.parent.mkdir(parents=True)
    raw.write_text('{"text":"synthetic-private-marker"}')
    output = tmp_path / "incomplete.tar.gz"
    with pytest.raises(ReleaseError, match="unshipped_link_target"):
        bundle(output, root)
    assert not output.exists()


def test_bundle_requires_explicit_source_members_but_accepts_generated_frontend(tmp_path: Path) -> None:
    root = source_fixture(tmp_path / "source")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "asr_server", "pyproject.toml"], check=True)
    (root / "docs").mkdir()
    (root / "docs/unreviewed.md").write_text("synthetic unreviewed note\n")
    output = tmp_path / "release.tar.gz"
    with pytest.raises(ReleaseError, match="not in the Git index"):
        bundle(output, root)
    assert not output.exists()
    subprocess.run(["git", "-C", str(root), "add", "docs/unreviewed.md"], check=True)
    bundle(output, root)
    with tarfile.open(output) as archive:
        assert "docs/unreviewed.md" in archive.getnames()
        assert "frontend/dist/index.html" in archive.getnames()


def test_source_export_rebundles_with_manifest_and_keeps_source_identity(tmp_path: Path) -> None:
    root = source_fixture(tmp_path / "source")
    original = source_identity(root)
    output = tmp_path / "release.tar.gz"
    bundle(output, root)
    exported = tmp_path / "exported"
    with tarfile.open(output) as archive:
        archive.extractall(exported, filter="data")
    assert source_identity(exported)["source_digest"] == original["source_digest"]
    bundle(tmp_path / "exported-release.tar.gz", exported)
    (exported / "docs").mkdir()
    (exported / "docs/unreviewed.md").write_text("synthetic unreviewed note\n")
    with pytest.raises(ReleaseError, match="not in the Git index or release manifest"):
        bundle(tmp_path / "unexpected.tar.gz", exported)


@pytest.mark.parametrize("relative", ["docs/private.json", "docs/previous-release.tar.gz", "frontend/dist/index.js.map"])
def test_bundle_rejects_undeclared_resource_types(tmp_path: Path, relative: str) -> None:
    root = source_fixture(tmp_path / "source")
    resource = root / relative
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_text("synthetic private payload\n")
    with pytest.raises(ReleaseError, match="unsupported source resource"):
        bundle(tmp_path / "release.tar.gz", root)


def test_bundle_output_cannot_become_future_source_input(tmp_path: Path) -> None:
    root = source_fixture(tmp_path / "source")
    for relative in ("docs/release.tar.gz", "frontend/dist/release.tar.gz", "release.tar.gz"):
        with pytest.raises(ReleaseError, match="generated output"):
            bundle(root / relative, root)
    output = root / "artifacts/release.tar.gz"
    bundle(output, root)
    with tarfile.open(output) as archive:
        assert not any(name.startswith("artifacts/") for name in archive.getnames())


def test_backup_restore_preserves_sqlite_media_checkpoints_and_permissions(tmp_path: Path) -> None:
    root = tmp_path / "data"
    store = data_fixture(root)
    source = source_fixture(tmp_path / "source")
    audio = root / "jobs/wb_example/audio.wav"
    audio.chmod(0o600)
    (root / ".env").write_text("do-not-back-up")
    (root / "token.txt").write_text("do-not-back-up")
    output = tmp_path / "backup"
    result = backup(root, output, source)
    assert result["files"] == 4
    assert not (output / ".env").exists()
    assert not (output / "token.txt").exists()
    assert not (output / "service.lock").exists()
    assert not (output / "workbench.sqlite3-wal").exists()
    restored = tmp_path / "restored"
    status = restore(output, restored, source)
    assert status["source_matches"] is True
    assert (restored / "jobs/wb_example/audio.wav").read_bytes() == b"retained-audio"
    assert (restored / "jobs/wb_example/checkpoints/chunk-0.json").read_text() == '{"text":"原文"}'
    assert os.stat(restored / "jobs/wb_example/audio.wav").st_mode & 0o777 == 0o600
    with sqlite3.connect(restored / "workbench.sqlite3") as db:
        assert db.execute("SELECT username FROM users").fetchone()[0] == "owner"
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert (restored / "restored-source.json").exists()
    with pytest.raises(ReleaseError):
        restore(output, restored, source)
    with pytest.raises(ReleaseError):
        backup(root, output, source)
    assert store.usage() == 0


def test_active_service_backup_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "data"
    data_fixture(root)
    source = source_fixture(tmp_path / "source")
    with offline_lock(root):
        with pytest.raises(ReleaseError, match="service is active"):
            backup(root, tmp_path / "backup", source)
    assert not (tmp_path / "backup").exists()
    assert not list(tmp_path.glob(".asr-backup-*"))


def test_restore_rejects_hash_tampering_and_path_traversal(tmp_path: Path) -> None:
    root = tmp_path / "data"
    data_fixture(root)
    source = source_fixture(tmp_path / "source")
    snapshot = tmp_path / "backup"
    backup(root, snapshot, source)
    audio = snapshot / "jobs/wb_example/audio.wav"
    audio.write_bytes(b"changed")
    with pytest.raises(ReleaseError, match="hash mismatch"):
        restore(snapshot, tmp_path / "new", source)
    assert not (tmp_path / "new").exists()
    audio.write_bytes(b"retained-audio")
    path = snapshot / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../escaped.txt"] = manifest["files"]["jobs/wb_example/audio.wav"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ReleaseError, match="unsafe path"):
        restore(snapshot, tmp_path / "new", source)
    assert not (tmp_path / "escaped.txt").exists()


def test_backup_rejects_external_data_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "data"
    data_fixture(root)
    source = source_fixture(tmp_path / "source")
    secret = tmp_path / "secret"
    secret.write_text("not-owned-by-service")
    (root / "jobs/wb_example/symlink").symlink_to(secret)
    with pytest.raises(ReleaseError, match="symlink"):
        backup(root, tmp_path / "backup", source)


def test_restore_accepts_empty_directory_and_reports_code_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "data"
    data_fixture(root)
    source = source_fixture(tmp_path / "source")
    snapshot = tmp_path / "backup"
    backup(root, snapshot, source)
    (source / "asr_server/main.py").write_text("print('changed version')")
    destination = tmp_path / "empty"
    destination.mkdir()
    result = restore(snapshot, destination, source)
    assert result["source_matches"] is False
    assert result["required_source"]["source_digest"]
    assert verify_backup(snapshot)["kind"] == "workbench_backup"
