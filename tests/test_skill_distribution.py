from __future__ import annotations

import io
from pathlib import Path
from zipfile import ZipFile

import pytest
from httpx import ASGITransport, AsyncClient

from asr_server import skill as skill_module
from asr_server.config import Settings
from asr_server.main import create_app


async def test_skill_is_discoverable_public_and_downloads_the_shipped_files() -> None:
    app = create_app(Settings(api_token="test-secret"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        docs = await client.get("/docs")
        oauth_redirect = await client.get("/docs/oauth2-redirect")
        schema = (await client.get("/openapi.json")).json()
        guide = await client.get("/skill")
        script = await client.get("/skill/asr_batch.py")
        download = await client.get("/skill/lan-asr.zip")
        assert (await client.get("/v1/models")).status_code == 401

    assert "](/skill)" in schema["info"]["description"]
    assert docs.status_code == oauth_redirect.status_code == 200
    assert 'href="/skill"' in docs.text
    assert 'href="/openapi.json"' in docs.text
    assert "text/markdown" in schema["paths"]["/skill"]["get"]["responses"]["200"]["content"]
    assert "security" not in schema["paths"]["/skill"]["get"]
    assert guide.status_code == script.status_code == download.status_code == 200
    assert guide.headers["content-type"].startswith("text/markdown")
    assert script.headers["content-type"].startswith("text/plain")
    assert download.headers["content-type"] == "application/zip"
    assert guide.content == (skill_module.SKILL_ROOT / "SKILL.md").read_bytes()
    assert script.content == (skill_module.SKILL_ROOT / "scripts/asr_batch.py").read_bytes()
    with ZipFile(io.BytesIO(download.content)) as archive:
        assert set(archive.namelist()) == {"lan-asr/SKILL.md", "lan-asr/scripts/asr_batch.py"}
        assert archive.read("lan-asr/SKILL.md") == guide.content
        assert archive.read("lan-asr/scripts/asr_batch.py") == script.content


async def test_incomplete_skill_release_returns_a_controlled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(skill_module, "SKILL_ROOT", tmp_path)
    app = create_app(Settings())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for path in ("/skill", "/skill/asr_batch.py", "/skill/lan-asr.zip"):
            response = await client.get(path)
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "skill_unavailable"
            assert str(tmp_path) not in response.text
