"""Check repository admission and local documentation/resource closure.

This is not a secret scanner or a declaration of permission to publish. Findings
contain only the referring repository path, line and category, never source text
or the rejected link target. No files are written and no network is contacted.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


MIB = 1024 * 1024
GENERATED_PARTS = frozenset({
    ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".cache", "logs", "tmp", "uploads", "models", "model_cache",
    "hf_cache", "transcripts", "asr-transcripts", "data", "artifacts", "scratch",
})
GENERATED_FILES = frozenset({".DS_Store", "skills-lock.json"})
REQUIRED_PACKAGE_FILES = (
    "README.md", "AGENTS.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "uv.lock", ".python-version",
    "asr_server/main.py", "frontend/dist/index.html",
    "skills/hcds-lan-asr/SKILL.md", "skills/hcds-lan-asr/scripts/asr_batch.py",
    "docs/reference/docs-endpoint-capabilities.md", "docs/guides/mcp-bridge.md",
    "test-fixtures/audio/test_short.wav", "test-fixtures/audio/README.md", "scripts/generate_test_audio.py",
    "test-fixtures/audio/test_short.reference.txt", "test-fixtures/audio/test_short.source.json",
    "examples/batch-manifest.sample.json", "examples/output-sample/index.md",
    "examples/output-sample/manifest.json",
    "examples/output-sample/items/sample-oral-001-deadbeef0001/transcript.md",
    "examples/output-sample/items/sample-oral-001-deadbeef0001/transcript.json",
)
SERVICE_PREFIXES = ("/api", "/v1", "/skill", "/docs", "/assets")
SERVICE_PATHS = frozenset({"/", "/health", "/readyz", "/openapi.json"})


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    category: str
    status: str = "error"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, timeout=30, check=False)


def _paths(output: bytes) -> set[str]:
    return {item.decode("utf-8", errors="surrogateescape") for item in output.split(b"\0") if item}


def inventory(root: Path) -> tuple[list[str], set[str], set[str], set[str]]:
    """Use Git's visibility, or all exported files when no checkout is present."""
    try:
        probe = _git(root, "rev-parse", "--show-toplevel")
    except FileNotFoundError:
        probe = subprocess.CompletedProcess([], 1, b"", b"")
    if probe.returncode == 0 and Path(probe.stdout.decode().strip()).resolve() == root:
        tracked_result = _git(root, "ls-files", "--cached", "-z")
        visible_result = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
        ignored_result = _git(root, "ls-files", "-ci", "--exclude-standard", "-z")
        changed_result = _git(root, "diff", "HEAD", "--name-only", "-z")
        if any(result.returncode for result in (tracked_result, visible_result, ignored_result)):
            raise OSError("repository inventory failed")
        tracked, visible = _paths(tracked_result.stdout), _paths(visible_result.stdout)
        changed = _paths(changed_result.stdout) if changed_result.returncode == 0 else tracked
        files = [name for name in tracked | visible if (root / name).is_file() or (root / name).is_symlink()]
        return sorted(files), tracked, changed | visible, _paths(ignored_result.stdout)
    if (root / ".git").exists():
        raise OSError("Git checkout inventory is unavailable")
    files = []
    def unreadable(error: OSError) -> None:
        raise error
    for directory, directories, leaves in os.walk(root, followlinks=False, onerror=unreadable):
        directories[:] = [name for name in directories if name != ".git"]
        for name in [*leaves, *(name for name in directories if (Path(directory) / name).is_symlink())]:
            files.append((Path(directory) / name).relative_to(root).as_posix())
    files.sort()
    return files, set(), set(files), set()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(MIB):
            digest.update(block)
    return digest.hexdigest()


def _generated(name: str, *, package: bool) -> bool:
    path = PurePosixPath(name)
    return bool(
        set(path.parts) & GENERATED_PARTS
        or path.name in GENERATED_FILES
        or path.suffix in {".pyc", ".pyo", ".sqlite", ".sqlite3", ".db"}
        or (path.name.startswith(".env") and path.name != ".env.example")
        or name == "skills/hcds-lan-asr/server.json"
        or name.startswith(".agents/")
        or (not package and name.startswith("frontend/dist/"))
    )


def _outside(path: Path, root: Path) -> bool:
    return not path.resolve().is_relative_to(root)


def _without_fences(text: str) -> list[str]:
    result: list[str] = []
    fence_character = ""
    fence_length = 0
    for line in text.splitlines():
        fence = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if not fence_character:
                fence_character, fence_length = marker[0], len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = ""
            result.append("")
        else:
            result.append("" if fence_character else line)
    return result


def _inline_text(text: str) -> str:
    text = re.sub(r"!?\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"(?<!\w)__([^_]+)__(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    text = re.sub(r"[*`~]", "", text)
    return html.unescape(text)


def markdown_anchors(text: str) -> set[str]:
    lines = _without_fences(text)
    anchors = set(re.findall(r"\b(?:id|name)=[\"']([^\"']+)[\"']", "\n".join(lines)))
    used: set[str] = set()
    for index, line in enumerate(lines):
        heading = re.match(r"^ {0,3}#{1,6}[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$", line)
        value = heading.group(1) if heading else None
        if value is None and index + 1 < len(lines) and line.strip() and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", lines[index + 1]):
            value = line.strip()
        if value is None:
            continue
        slug = re.sub(r"[^\w\- ]", "", _inline_text(value).strip().lower()).replace(" ", "-")
        candidate, suffix = slug, 0
        while candidate in used:
            suffix += 1
            candidate = f"{slug}-{suffix}"
        used.add(candidate)
        anchors.add(candidate)
    return anchors


def _target(value: str) -> str:
    value = value.strip()
    if value.startswith("<") and ">" in value:
        value = value[1:value.index(">")]
    else:
        value = re.split(r"\s+[\"']", value, maxsplit=1)[0]
    return html.unescape(re.sub(r"\\([() ])", r"\1", value))


def markdown_links(text: str) -> list[tuple[int, str]]:
    """Read inline, reference-definition and HTML links outside code fences."""
    links: list[tuple[int, str]] = []
    lines = _without_fences(text)
    for number, raw in enumerate(lines, 1):
        line = re.sub(r"(`+).*?\1", "", raw)
        definition = re.match(r"^ {0,3}\[[^]]+\]:\s*(.+)$", line)
        if definition:
            links.append((number, _target(definition.group(1))))
            continue
        for match in re.finditer(r"!?\[[^\n]*?\]\(", line):
            start, cursor, depth = match.end(), match.end(), 1
            while cursor < len(line) and depth:
                if line[cursor] == "\\":
                    cursor += 2
                    continue
                if line[cursor] == "(":
                    depth += 1
                elif line[cursor] == ")":
                    depth -= 1
                cursor += 1
            if not depth:
                links.append((number, _target(line[start:cursor - 1])))
    joined = "\n".join(re.sub(r"(`+).*?\1", "", line) for line in lines)
    for tag in re.finditer(r"<(?:a|img)\b[^>]*>", joined, flags=re.IGNORECASE | re.DOTALL):
        for attribute in re.finditer(r"\b(?:href|src)\s*=\s*([\"'])(.*?)\1", tag.group(), re.IGNORECASE | re.DOTALL):
            links.append((joined.count("\n", 0, tag.start()) + 1, html.unescape(attribute.group(2))))
    return links


def check_markdown(root: Path, names: list[str], *, members: set[str] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    cache: dict[Path, set[str]] = {}
    for name in names:
        source = root / name
        if source.suffix.lower() != ".md" or _outside(source, root):
            continue
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            findings.append(Finding(name, 1, "markdown_unreadable"))
            continue
        for line, value in markdown_links(text):
            try:
                parsed = urlsplit(value)
            except ValueError:
                findings.append(Finding(name, line, "invalid_link"))
                continue
            if parsed.scheme or parsed.netloc:
                if parsed.scheme == "file":
                    findings.append(Finding(name, line, "absolute_file_link"))
                continue
            path = unquote(parsed.path)
            if path.startswith("/"):
                if path in SERVICE_PATHS or any(path == prefix or path.startswith(prefix + "/") for prefix in SERVICE_PREFIXES):
                    continue
                findings.append(Finding(name, line, "absolute_file_link"))
                continue
            target = source.parent / path if path else source
            if _outside(target, root):
                findings.append(Finding(name, line, "link_leaves_repository"))
            elif not target.exists():
                findings.append(Finding(name, line, "missing_link_target"))
            elif members is not None and not (
                target.resolve().relative_to(root).as_posix() in members
                or target.is_dir() and any(member.startswith(target.resolve().relative_to(root).as_posix().rstrip("/") + "/")
                                           for member in members)
            ):
                findings.append(Finding(name, line, "unshipped_link_target"))
            elif parsed.fragment:
                if target.is_dir():
                    target = target / "README.md"
                if target.suffix.lower() != ".md" or not target.is_file():
                    findings.append(Finding(name, line, "unsupported_local_anchor"))
                    continue
                target = target.resolve()
                if target not in cache:
                    try:
                        cache[target] = markdown_anchors(target.read_text(encoding="utf-8"))
                    except (OSError, UnicodeError):
                        findings.append(Finding(name, line, "anchor_target_unreadable"))
                        continue
                fragment = unquote(parsed.fragment)
                if fragment not in cache[target]:
                    findings.append(Finding(name, line, "missing_anchor"))
    return findings


def check_package(root: Path, names: list[str]) -> list[Finding]:
    findings = [Finding(name, 1, "missing_package_resource") for name in REQUIRED_PACKAGE_FILES
                if not (root / name).is_file() or _outside(root / name, root)]
    manifest_path = root / "manifest.json"
    if _outside(manifest_path, root):
        findings.append(Finding("manifest.json", 1, "unsafe_package_manifest"))
        return findings
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("format_version") != 1 or manifest.get("kind") != "source_release":
            raise ValueError("unsupported package manifest")
        members = manifest["files"]
        if not isinstance(members, dict):
            raise ValueError("invalid file map")
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        findings.append(Finding("manifest.json", 1, "invalid_package_manifest"))
        return findings
    for name, record in members.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            findings.append(Finding("manifest.json", 1, "invalid_package_member"))
            continue
        path = root / name
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or _outside(path, root):
            findings.append(Finding("manifest.json", 1, "unsafe_package_member"))
        elif not path.is_file():
            findings.append(Finding("manifest.json", 1, "missing_manifest_member"))
        elif path.stat().st_size != record.get("size_bytes") or _digest(path) != record.get("sha256"):
            findings.append(Finding(name, 1, "package_hash_mismatch"))
    for name in names:
        if name != "manifest.json" and name not in members:
            findings.append(Finding(name, 1, "unlisted_package_member"))
    index = root / "frontend/dist/index.html"
    if index.is_file() and not _outside(index, root):
        try:
            text = index.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            findings.append(Finding("frontend/dist/index.html", 1, "frontend_index_unreadable"))
        else:
            for attribute in re.finditer(r"\b(?:src|href)\s*=\s*([\"'])(.*?)\1", text):
                parsed = urlsplit(attribute.group(2))
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                asset = index.parent / unquote(parsed.path).lstrip("/")
                if _outside(asset, index.parent) or not asset.is_file():
                    findings.append(Finding("frontend/dist/index.html", text.count("\n", 0, attribute.start()) + 1,
                                            "missing_frontend_asset"))
    return findings


def check_tree(root: Path, *, package: bool = False) -> list[Finding]:
    root = root.resolve()
    if not root.is_dir():
        raise OSError("repository directory is unavailable")
    names, _tracked, changed, ignored = inventory(root)
    findings = [Finding(name, 1, "tracked_ignored_conflict") for name in sorted(ignored)]
    for name in names:
        path = root / name
        if _outside(path, root):
            findings.append(Finding(name, 1, "source_symlink_escape"))
            continue
        if not path.is_file():
            findings.append(Finding(name, 1, "missing_source_file"))
            continue
        if _generated(name, package=package):
            findings.append(Finding(name, 1, "generated_file_admission"))
        size = path.stat().st_size
        if size > 5 * MIB:
            findings.append(Finding(name, 1, "file_exceeds_five_mib"))
        elif size > MIB and name in changed:
            findings.append(Finding(name, 1, "large_file_requires_review"))
    findings.extend(check_markdown(root, names))
    if package:
        findings.extend(check_package(root, names))
    return sorted(set(findings))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                       help="inspect source only (default: the script's project root)")
    scope.add_argument("--package-root", type=Path, help="inspect only this extracted runtime package")
    args = parser.parse_args()
    try:
        if args.package_root is not None:
            findings = [Finding("package/" + item.path, item.line, item.category, item.status)
                        for item in check_tree(args.package_root, package=True)]
        else:
            findings = check_tree(args.root)
    except (OSError, subprocess.SubprocessError):
        findings = [Finding(".", 1, "repository_check_unavailable")]
    findings = sorted(set(findings))
    for finding in findings:
        print(json.dumps(asdict(finding), ensure_ascii=True))
    errors = sum(finding.status == "error" for finding in findings)
    print(json.dumps({"errors": errors, "pending": sum(finding.status == "pending" for finding in findings),
                      "secret_scan": "not_performed", "publication_authorization": "not_evaluated"}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
