"""Run pinned Gitleaks locally; expose locations and rule IDs, never matches."""
from __future__ import annotations

import argparse
import bisect
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


GITLEAKS_VERSION = "8.30.1"
_REPORT_TEMPLATE = ('[{{range $i, $finding := .}}{{if $i}},{{end}}'
                    '{"RuleID":{{printf "%q" $finding.RuleID}},"StartLine":{{$finding.StartLine}},'
                    '"File":{{printf "%q" $finding.File}}}{{end}}]')


def current_input(root: Path) -> tuple[str, list[tuple[int, str]]]:
    """Scan visible source and deletion lines, without opening ignored data."""
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=root, capture_output=True, check=True, timeout=15)
    parts: list[str] = []
    boundaries: list[tuple[int, str]] = []
    next_line = 1
    for name in sorted(set(result.stdout.split(b"\0")) - {b""}):
        relative = name.decode("utf-8")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeError:
            continue
        boundaries.append((next_line, relative))
        chunk = text if text.endswith("\n") else text + "\n"
        parts.append(chunk)
        next_line += chunk.count("\n")
    diff = subprocess.run(["git", "diff", "--no-ext-diff", "--no-textconv", "HEAD", "--"],
                          cwd=root, capture_output=True, check=True, timeout=15).stdout.decode("utf-8", errors="replace")
    if diff:
        boundaries.append((next_line, "<working-tree-diff>"))
        # Scan both additions and deletions as source content, not patch syntax.
        # Preserve line numbers, but remove the one-character diff marker: it can
        # otherwise make a source digest match the generic credential rule.
        parts.append("\n".join(line[1:] if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
                               else line for line in diff.splitlines()) + "\n")
    return "".join(parts), boundaries


def scan(root: Path, executable: Path, refs: list[str]) -> dict[str, Any]:
    version = subprocess.run([str(executable), "version"], capture_output=True, text=True, check=True, timeout=10)
    if version.stdout.strip().lstrip("v") != GITLEAKS_VERSION:
        raise ValueError("unexpected scanner version")
    command = [str(executable), "--no-banner", "--redact=100", "--report-format=template"]
    if (root / ".gitleaks.toml").is_file():
        command += ["--config=" + str(root / ".gitleaks.toml")]
    boundaries: list[tuple[int, str]] = []
    source: str | None = None
    resolved: list[str] = []
    if refs:
        for ref in refs:
            revision = subprocess.run(["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
                                      cwd=root, capture_output=True, text=True, check=True, timeout=10)
            resolved.append(revision.stdout.strip())
        command += ["git", "--log-opts=" + " ".join(resolved), str(root)]
    else:
        source, boundaries = current_input(root)
        command += ["stdin"]
    # Some platforms cannot reopen a piped /dev/stdout. The scanner's template
    # contains only rule/location fields: matches, secrets and identities never
    # enter the temporary report. Its private directory is removed on every exit.
    with tempfile.TemporaryDirectory(prefix="asr-secret-check-") as directory:
        template = Path(directory) / "locations.tmpl"
        report = Path(directory) / "locations.json"
        template.write_text(_REPORT_TEMPLATE, encoding="utf-8")
        template.chmod(0o600)
        report.touch(mode=0o600)
        result = subprocess.run([*command, "--report-template=" + str(template), "--report-path=" + str(report)],
                                cwd=root, input=source, capture_output=True, text=True, timeout=120)
        if result.returncode not in {0, 1}:
            raise RuntimeError("scanner failed")
        findings = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(findings, list) or any(not isinstance(item, dict) for item in findings):
        raise ValueError("unexpected scanner report")
    if result.returncode == 1 and not findings:
        raise RuntimeError("scanner did not provide its failed findings")
    locations = []
    starts = [start for start, _ in boundaries]
    for finding in findings:
        line = int(finding.get("StartLine", 0))
        if boundaries:
            index = bisect.bisect_right(starts, line) - 1
            start, path = boundaries[max(0, index)]
            line = max(1, line - start + 1)
        else:
            path = str(finding.get("File", "<history>"))
            # Do not expose scanner-provided absolute host paths.
            if Path(path).is_absolute() or ".." in Path(path).parts:
                path = "<history>"
        rule = str(finding.get("RuleID", "unknown"))
        locations.append({"path": path, "line": line,
                          "rule": rule if rule.isascii() and rule.replace("-", "").isalnum() else "unknown"})
    return {"status": "failed" if locations else "passed", "scanner": "gitleaks", "version": GITLEAKS_VERSION,
            "scope": "explicit_git_refs" if refs else "visible_source_and_working_diff",
            "refs": resolved, "findings": locations,
            "limitations": "Pattern scan only; not a privacy or publication authorization."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gitleaks", type=Path, required=True)
    parser.add_argument("--ref", action="append", default=[], help="explicit Git ref; omit for source and working diff")
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        git_entry = root / ".git"
        if not (git_entry.is_dir() or git_entry.is_file()):
            parser.exit(2, "secret check unavailable: --root must have its own .git file or directory\n")
        result = scan(root, args.gitleaks.resolve(), args.ref)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        parser.exit(2, "secret check unavailable: verify the pinned scanner, Git scope and local input\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
