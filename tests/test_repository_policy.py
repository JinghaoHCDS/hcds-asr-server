from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import check_repository as policy


def write(root: Path, name: str, data: str | bytes = "content\n") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return path


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def package(root: Path) -> Path:
    for name in policy.REQUIRED_PACKAGE_FILES:
        write(root, name, "# Package\n" if name.endswith(".md") else "content\n")
    refresh_manifest(root)
    return root


def refresh_manifest(root: Path) -> None:
    members = {path.relative_to(root).as_posix(): {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size,
    } for path in root.rglob("*") if path.is_file() and path != root / "manifest.json"}
    write(root, "manifest.json", json.dumps({"format_version": 1, "kind": "source_release", "files": members}))


def test_git_inventory_includes_new_sources_and_rejects_tracked_ignored_files(tmp_path: Path) -> None:
    git(tmp_path, "-c", "init.templateDir=", "init", "--quiet")
    write(tmp_path, ".gitignore", "data/\n")
    write(tmp_path, "data/private.json", '{"text":"private-body-marker"}')
    write(tmp_path, "scripts/new.py", "pass\n")
    git(tmp_path, "add", "-f", "data/private.json")
    names, _tracked, changed, _ignored = policy.inventory(tmp_path)
    assert "scripts/new.py" in names and "scripts/new.py" in changed
    findings = policy.check_tree(tmp_path)
    assert policy.Finding("data/private.json", 1, "tracked_ignored_conflict") in findings
    assert policy.Finding("data/private.json", 1, "generated_file_admission") in findings


def test_unavailable_git_inventory_cannot_be_reported_as_an_empty_clean_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(policy, "_git", lambda *args: subprocess.CompletedProcess([], 1, b"", b""))
    with pytest.raises(OSError, match="inventory is unavailable"):
        policy.inventory(tmp_path)


@pytest.mark.parametrize("name", ["artifacts/raw.json", "scratch/run.txt", "asr-transcripts/result.md",
                                  "frontend/dist/index.html", "skills/hcds-lan-asr/server.json"])
def test_exported_source_rejects_generated_paths(tmp_path: Path, name: str) -> None:
    write(tmp_path, name)
    assert policy.Finding(name, 1, "generated_file_admission") in policy.check_tree(tmp_path)


def test_synthetic_fixture_has_no_legacy_large_file_exemption(tmp_path: Path) -> None:
    name = "test-fixtures/audio/test_short.wav"
    path = write(tmp_path, name, b"a" * policy.MIB)
    assert policy.check_tree(tmp_path) == []
    path.write_bytes(b"b" * (policy.MIB + 1))
    assert policy.check_tree(tmp_path) == [policy.Finding(name, 1, "large_file_requires_review")]
    path.write_bytes(b"b" * (5 * policy.MIB + 1))
    assert policy.Finding(name, 1, "file_exceeds_five_mib") in policy.check_tree(tmp_path)


def test_links_follow_moved_files_unicode_duplicate_anchors_and_html_images(tmp_path: Path) -> None:
    write(tmp_path, "docs/New Guide.md", "# 使用 API\n\n## 重复\n\n## 重复\n\n## quality_v1\n\n<a id=\"explicit-anchor\"></a>\n")
    write(tmp_path, "assets/pixel.png", b"synthetic-image")
    write(tmp_path, "README.md", """# Start
[Guide](docs/New%20Guide.md#使用-api)
[Duplicate](docs/New%20Guide.md#重复-1)
[Code name](docs/New%20Guide.md#quality_v1)
[Explicit](docs/New%20Guide.md#explicit-anchor)
<img
  src="assets/pixel.png"
>
[API](/v1/models) [Skill](/skill/hcds-lan-asr.zip) [Docs](/docs/mcp)
[External](https://example.invalid/missing#anything)
[Reference][guide]
[guide]: docs/New%20Guide.md#使用-api "Title"
```markdown
[Only an example](not-a-real-file.md#not-an-anchor)
```
""")
    assert policy.check_tree(tmp_path) == []


def test_broken_links_and_anchors_are_reported_without_target_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "private-link-target-marker"
    write(tmp_path, "README.md", f"# Start\n[Missing]({marker}.json)\n[Anchor](guide.md#missing)\n")
    write(tmp_path, "guide.md", "# Present\n")
    monkeypatch.setattr(sys, "argv", ["check_repository", "--root", str(tmp_path)])
    assert policy.main() == 1
    output = capsys.readouterr().out
    records = [json.loads(line) for line in output.splitlines()]
    assert marker not in output
    assert {record["category"] for record in records if "category" in record} == {"missing_link_target", "missing_anchor"}
    assert {record["line"] for record in records if "category" in record} == {2, 3}
    assert records[-1]["secret_scan"] == "not_performed"


def test_package_rejects_missing_runtime_resources_and_unresolved_raw_evidence(tmp_path: Path) -> None:
    root = package(tmp_path)
    assert policy.check_tree(root, package=True) == []
    write(root, "README.md", "# Package\n[Historical evidence](docs/validation-artifacts/run/raw.json)\n")
    (root / "skills/hcds-lan-asr/SKILL.md").unlink()
    refresh_manifest(root)
    findings = policy.check_tree(root, package=True)
    assert policy.Finding("README.md", 2, "missing_link_target") in findings
    assert policy.Finding("skills/hcds-lan-asr/SKILL.md", 1, "missing_package_resource") in findings


@pytest.mark.parametrize("complete", [True, False])
def test_package_cli_checks_only_the_exported_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], complete: bool,
) -> None:
    extracted = package(tmp_path / "exported")
    script = write(extracted, "scripts/check_repository.py", "# synthetic package entry\n")
    if not complete:
        (extracted / "LICENSE").unlink()
    refresh_manifest(extracted)
    write(tmp_path, "README.md", "[Source tree must not be checked](missing-source-only-file.md)\n")
    original_check = policy.check_tree
    calls: list[tuple[Path, bool]] = []

    def scoped_check(root: Path, *, package: bool = False) -> list[policy.Finding]:
        calls.append((root, package))
        return original_check(root, package=package)

    monkeypatch.setattr(policy, "check_tree", scoped_check)
    monkeypatch.setattr(policy, "__file__", str(script))
    monkeypatch.setattr(sys, "argv", ["check_repository", "--package-root", str(extracted)])
    assert not (extracted / ".git").exists()
    assert policy.main() == (0 if complete else 1)
    assert calls == [(extracted, True)]
    output = capsys.readouterr()
    records = [json.loads(line) for line in output.out.splitlines()]
    assert records[-1]["errors"] == (0 if complete else 1)
    assert not output.err
    if not complete:
        assert records[0]["path"] == "package/LICENSE"


def test_cli_source_and_package_modes_are_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["check_repository", "--root", str(tmp_path),
                                     "--package-root", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        policy.main()
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["LICENSE", "test-fixtures/audio/test_short.wav",
                                     "test-fixtures/audio/README.md", "scripts/generate_test_audio.py"])
def test_package_requires_license_and_synthetic_audio_source_entries(tmp_path: Path, missing: str) -> None:
    root = package(tmp_path)
    (root / missing).unlink()
    refresh_manifest(root)
    assert policy.Finding(missing, 1, "missing_package_resource") in policy.check_tree(root, package=True)


def test_package_checks_manifest_and_frontend_asset_closure(tmp_path: Path) -> None:
    root = package(tmp_path)
    write(root, "frontend/dist/index.html", '<script src="/assets/app.js"></script>\n')
    refresh_manifest(root)
    assert policy.Finding("frontend/dist/index.html", 1, "missing_frontend_asset") in policy.check_tree(root, package=True)
    write(root, "frontend/dist/assets/app.js", "// synthetic\n")
    refresh_manifest(root)
    assert policy.check_tree(root, package=True) == []
    write(root, "frontend/dist/assets/app.js", "// changed\n")
    write(root, "extra.txt", "not-in-manifest\n")
    findings = policy.check_tree(root, package=True)
    assert policy.Finding("frontend/dist/assets/app.js", 1, "package_hash_mismatch") in findings
    assert policy.Finding("extra.txt", 1, "unlisted_package_member") in findings


def test_package_manifest_cannot_read_outside_root(tmp_path: Path) -> None:
    root = package(tmp_path / "package")
    write(tmp_path, "private-marker.txt", "private-body-marker")
    write(root, "manifest.json", json.dumps({"format_version": 1, "kind": "source_release",
                                           "files": {"../private-marker.txt": {"sha256": "", "size_bytes": 0}}}))
    assert policy.Finding("manifest.json", 1, "unsafe_package_member") in policy.check_tree(root, package=True)


@pytest.mark.parametrize("metadata", [{"format_version": 2, "kind": "source_release"},
                                      {"format_version": 1, "kind": "workbench_backup"}])
def test_package_manifest_requires_the_source_release_contract(tmp_path: Path, metadata: dict[str, object]) -> None:
    root = package(tmp_path)
    write(root, "manifest.json", json.dumps({**metadata, "files": {}}))
    assert policy.Finding("manifest.json", 1, "invalid_package_manifest") in policy.check_tree(root, package=True)
