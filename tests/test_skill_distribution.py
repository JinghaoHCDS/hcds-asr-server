from __future__ import annotations

import io
import json
import os
import subprocess
import sys
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
        download = await client.get("/skill/hcds-lan-asr.zip")
        old_download = await client.get("/skill/lan-asr.zip")
        assert (await client.get("/v1/models")).status_code == 401

    assert "](/skill)" in schema["info"]["description"]
    assert old_download.status_code == 307
    assert old_download.headers["location"] == "/skill/hcds-lan-asr.zip"
    assert "/skill/hcds-lan-asr.zip" in schema["paths"]
    assert "/skill/lan-asr.zip" not in schema["paths"]
    assert docs.status_code == oauth_redirect.status_code == 200
    assert 'href="/skill"' in docs.text
    assert 'href="/openapi.json"' in docs.text
    assert "text/markdown" in schema["paths"]["/skill"]["get"]["responses"]["200"]["content"]
    assert "security" not in schema["paths"]["/skill"]["get"]
    assert guide.status_code == script.status_code == download.status_code == 200
    assert guide.headers["content-type"].startswith("text/markdown")
    assert script.headers["content-type"].startswith("text/plain")
    assert download.headers["content-type"] == "application/zip"
    assert "Server address: `http://test`" in guide.text
    assert "name: hcds-lan-asr\n" in guide.text
    template = (skill_module.SKILL_ROOT / "SKILL.md").read_text()
    prefix, suffix = template.split("<!-- deployment-address -->")
    assert guide.text.startswith(prefix) and guide.text.endswith(suffix)
    assert guide.headers["cache-control"] == download.headers["cache-control"] == "no-store"
    assert script.content == (skill_module.SKILL_ROOT / "scripts/asr_batch.py").read_bytes()
    with ZipFile(io.BytesIO(download.content)) as archive:
        assert set(archive.namelist()) == {"hcds-lan-asr/SKILL.md", "hcds-lan-asr/scripts/asr_batch.py", "hcds-lan-asr/server.json"}
        assert archive.read("hcds-lan-asr/SKILL.md") == guide.content
        assert archive.read("hcds-lan-asr/scripts/asr_batch.py") == script.content
        assert json.loads(archive.read("hcds-lan-asr/server.json")) == {"base_url": "http://test"}
        assert b"test-secret" not in b"".join(archive.read(name) for name in archive.namelist())


@pytest.mark.parametrize("configured", ["", "http://127.0.0.1:18080", "http://0.0.0.0:18080"])
async def test_downloaded_client_uses_download_origin_without_manual_ip_configuration(
    configured: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(Settings(public_base_url=configured))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://asr-host.lan:18080") as client:
        download = await client.get("/skill/hcds-lan-asr.zip")
    assert download.status_code == 200
    with ZipFile(io.BytesIO(download.content)) as archive:
        archive.extractall(tmp_path)
    monkeypatch.delenv("ASR_BASE_URL", raising=False)
    environment = {**os.environ, "HOME": str(tmp_path / "home")}
    # This is the actual extracted client; no network, source patch or --base-url.
    result = subprocess.run(
        [sys.executable, "-c", "import runpy,sys; print(runpy.run_path(sys.argv[1])['resolve_base_url'](None))",
         str(tmp_path / "hcds-lan-asr/scripts/asr_batch.py")],
        env=environment, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "http://asr-host.lan:18080"


async def test_configured_deployment_url_wins_over_download_host() -> None:
    app = create_app(Settings(public_base_url="https://asr.example.test:18080/gateway/"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1:18080") as client:
        guide = await client.get("/skill")
        download = await client.get("/skill/hcds-lan-asr.zip")
    assert "Server address: `https://asr.example.test:18080/gateway`" in guide.text
    with ZipFile(io.BytesIO(download.content)) as archive:
        assert json.loads(archive.read("hcds-lan-asr/server.json"))["base_url"] == "https://asr.example.test:18080/gateway"
        assert archive.read("hcds-lan-asr/SKILL.md") == guide.content


async def test_request_origin_is_not_cached_between_clients() -> None:
    app = create_app(Settings())
    async with AsyncClient(transport=ASGITransport(app=app)) as client:
        for hostname in ("first.lan", "second.lan"):
            download = await client.get(f"http://{hostname}:18080/skill/hcds-lan-asr.zip")
            with ZipFile(io.BytesIO(download.content)) as archive:
                assert json.loads(archive.read("hcds-lan-asr/server.json"))["base_url"] == f"http://{hostname}:18080"


async def test_invalid_advertised_url_does_not_distribute_credentials() -> None:
    app = create_app(Settings(public_base_url="http://user:private-secret@asr.lan:18080"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for path in ("/skill", "/skill/hcds-lan-asr.zip"):
            response = await client.get(path)
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "skill_address_invalid"
            assert "private-secret" not in response.text


async def test_incomplete_skill_release_returns_a_controlled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(skill_module, "SKILL_ROOT", tmp_path)
    app = create_app(Settings())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for path in ("/skill", "/skill/asr_batch.py", "/skill/hcds-lan-asr.zip"):
            response = await client.get(path)
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "skill_unavailable"
            assert str(tmp_path) not in response.text
