"""Serve the official, versioned client Skill from the checked-out release."""
from __future__ import annotations

import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi.responses import PlainTextResponse

from asr_server.errors import AsrError


SKILL_ROOT = Path(__file__).resolve().parent.parent / "skills" / "lan-asr"
SKILL_FILES = ("SKILL.md", "scripts/asr_batch.py")


class MarkdownResponse(PlainTextResponse):
    media_type = "text/markdown"


def skill_file(relative: str) -> bytes:
    # Only fixed, shipped files are exposed; there is no user-supplied path route.
    if relative not in SKILL_FILES:
        raise ValueError("not a distributed Skill file")
    try:
        return (SKILL_ROOT / relative).read_bytes()
    except OSError as exc:
        raise AsrError(
            503, "skill_unavailable", "official LAN ASR Skill is missing from this release"
        ) from exc


def skill_archive() -> bytes:
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for relative in SKILL_FILES:
            archive.writestr(f"lan-asr/{relative}", skill_file(relative))
    return output.getvalue()
