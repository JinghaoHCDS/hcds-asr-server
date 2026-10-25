"""Serve the official, versioned client Skill from the checked-out release."""
from __future__ import annotations

import io
import ipaddress
import json
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi.responses import PlainTextResponse

from asr_server.errors import AsrError


SKILL_ROOT = Path(__file__).resolve().parent.parent / "skills" / "hcds-lan-asr"
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
            503, "skill_unavailable", "official hcds-lan-asr Skill is missing from this release"
        ) from exc


def skill_base_url(configured: str, requested: str) -> str:
    """Prefer a configured LAN origin; otherwise use the download's own origin.

    Historical loopback/wildcard defaults are not client-facing addresses.
    No forwarded header or unrelated environment value is copied into a Skill.
    """
    def normalized(value: str) -> str:
        try:
            parts = urlsplit(value)
            if (parts.scheme not in {"http", "https"} or not parts.hostname
                    or parts.username is not None or parts.password is not None
                    or parts.query or parts.fragment or parts.port == 0
                    or any(char.isspace() or char in '`<>\\"' for char in value)):
                raise ValueError("invalid address")
        except ValueError as exc:
            raise AsrError(503, "skill_address_invalid", "Skill requires a valid HTTP server address without credentials") from exc
        path = quote(parts.path.rstrip("/"), safe="/%:@-._~!$&'()*+,;=")
        return urlunsplit((parts.scheme, parts.netloc, path, "", ""))

    if configured:
        value = normalized(configured)
        hostname = urlsplit(value).hostname or ""
        try:
            address = ipaddress.ip_address(hostname)
            usable = not (address.is_loopback or address.is_unspecified)
        except ValueError:
            usable = hostname.lower().rstrip(".") != "localhost"
        if usable:
            return value
    return normalized(requested)


def skill_document(base_url: str) -> bytes:
    text = skill_file("SKILL.md").decode("utf-8")
    deployment = (
        "## This deployment\n\n"
        f"Server address: `{base_url}`\n\n"
        "The downloaded ZIP includes this address in `server.json`. "
        "The bundled client uses it automatically unless an explicit argument "
        "or ASR_BASE_URL overrides it.\n"
    )
    return text.replace("<!-- deployment-address -->", deployment, 1).encode("utf-8")


def skill_archive(base_url: str) -> bytes:
    output = io.BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for relative in SKILL_FILES:
            content = skill_document(base_url) if relative == "SKILL.md" else skill_file(relative)
            archive.writestr(f"hcds-lan-asr/{relative}", content)
        archive.writestr("hcds-lan-asr/server.json", json.dumps({"base_url": base_url}, indent=2) + "\n")
    return output.getvalue()
