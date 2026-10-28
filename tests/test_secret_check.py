from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import check_secrets


def test_cli_rejects_export_inside_an_ancestor_repository_without_scanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / ".git").mkdir()
    exported = tmp_path / "source-export"
    exported.mkdir()
    monkeypatch.setattr(sys, "argv", ["check_secrets", "--root", str(exported),
                                     "--gitleaks", str(tmp_path / "scanner")])

    def unexpected_scan(*args: object) -> dict[str, Any]:
        pytest.fail("a source export must not invoke the scanner")

    monkeypatch.setattr(check_secrets, "scan", unexpected_scan)
    with pytest.raises(SystemExit) as error:
        check_secrets.main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "own .git file or directory" in output.err
    assert str(tmp_path) not in output.out + output.err
    assert not output.out


def test_current_input_scans_added_and_deleted_values_without_patch_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if "ls-files" in command:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        diff = b'--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-api_key = "synthetic-old"\n+api_key = "synthetic-new"\n'
        return subprocess.CompletedProcess(command, 0, diff, b"")
    monkeypatch.setattr("scripts.check_secrets.subprocess.run", run)
    text, boundaries = check_secrets.current_input(tmp_path)
    assert '-api_key' not in text and '+api_key' not in text
    assert 'api_key = "synthetic-old"' in text and 'api_key = "synthetic-new"' in text
    assert boundaries == [(1, "<working-tree-diff>")]


def test_scanner_result_contains_only_locations_not_matches_or_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "SYNTHETIC_PRIVATE_VALUE_DO_NOT_PRINT"
    monkeypatch.setattr(check_secrets, "current_input", lambda root: ("synthetic input", [(1, "example.py")]))

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[-1] == "version":
            return subprocess.CompletedProcess(command, 0, check_secrets.GITLEAKS_VERSION, "")
        assert "--redact=100" in command
        template = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--report-template=")))
        assert all(field not in template.read_text() for field in ["Secret", "Match", "Author", "Email", "Message"])
        report = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--report-path=")))
        report.write_text(json.dumps([{"StartLine": 2, "RuleID": "generic-api-key", "Secret": marker,
                                      "Match": marker, "Author": marker, "Email": marker, "Message": marker}]))
        return subprocess.CompletedProcess(command, 1, "", marker)

    monkeypatch.setattr("scripts.check_secrets.subprocess.run", run)
    result = check_secrets.scan(tmp_path, Path("scanner"), [])
    assert marker not in json.dumps(result)
    assert result["status"] == "failed"
    assert result["findings"] == [{"path": "example.py", "line": 2, "rule": "generic-api-key"}]


def test_scanner_failed_without_report_cannot_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_secrets, "current_input", lambda root: ("", []))

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0 if command[-1] == "version" else 1,
                                           check_secrets.GITLEAKS_VERSION if command[-1] == "version" else "", "")

    monkeypatch.setattr("scripts.check_secrets.subprocess.run", run)
    with pytest.raises(ValueError):
        check_secrets.scan(tmp_path, Path("scanner"), [])


@pytest.mark.parametrize("report", ["", "not-json", "{}", "[null]"])
def test_success_exit_requires_valid_findings_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: str,
) -> None:
    monkeypatch.setattr(check_secrets, "current_input", lambda root: ("", []))

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[-1] != "version":
            output = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--report-path=")))
            output.write_text(report)
        return subprocess.CompletedProcess(command, 0,
                                           check_secrets.GITLEAKS_VERSION if command[-1] == "version" else report, "")

    monkeypatch.setattr("scripts.check_secrets.subprocess.run", run)
    with pytest.raises(ValueError):
        check_secrets.scan(tmp_path, Path("scanner"), [])
