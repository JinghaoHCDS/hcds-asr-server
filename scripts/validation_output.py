"""Content-free smoke summaries and explicit, private content reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from asr_server.observability import _language_label


def text_metrics(text: str, prefix: str = "text") -> dict[str, object]:
    return {f"{prefix}_characters": len(text),
            f"{prefix}_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def safe_language(value: object) -> str:
    return _language_label(value) or "unknown"


def numeric_metrics(value: object) -> dict[str, Any]:
    """Keep numeric measurements without copying free text, terms or bodies."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.isascii() or not key.replace("_", "").isalnum():
            continue
        if isinstance(item, (bool, int)) or isinstance(item, float) and math.isfinite(item):
            result[key] = item
        elif isinstance(item, dict):
            nested = numeric_metrics(item)
            if nested:
                result[key] = nested
    return result


def add_report_arguments(parser: argparse.ArgumentParser, *, output: bool = True) -> None:
    if output:
        parser.add_argument("--output", type=Path, help="new ignored or external JSON report (mode 0600)")
    parser.add_argument("--include-content", action="store_true",
                        help="include transcript content only in --output; stdout remains content-free")


def validate_report_output(output: Path | None, *, include_content: bool = False) -> None:
    if include_content and output is None:
        raise ValueError("--include-content requires an explicit --output in a private location")
    if output is None:
        return
    if output.exists() or output.is_symlink():
        raise ValueError("report output must be a new file")
    root = Path(__file__).resolve().parents[1]
    resolved = output.resolve()
    if resolved.is_relative_to(root):
        # An exported source tree also must not collect its own reports.
        relative = resolved.relative_to(root)
        if not relative.parts or relative.parts[0] not in {"artifacts", "scratch", "data", "tmp"}:
            raise ValueError("report inside the source tree must use an ignored generated directory")
        if (root / ".git").exists():
            ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "--quiet", "--", str(resolved)],
                                     capture_output=True, check=False)
            if ignored.returncode:
                raise ValueError("report output must be ignored by Git")


def emit_report(summary: dict[str, Any], *, output: Path | None = None,
                content: dict[str, Any] | None = None, include_content: bool = False) -> None:
    validate_report_output(output, include_content=include_content)
    if output is not None:
        output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        report = {**summary, "content": content} if include_content else summary
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
