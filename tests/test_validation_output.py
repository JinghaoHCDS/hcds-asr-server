from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.qwen_asr_backend_smoke import print_result_summary
from scripts.validation_output import emit_report, validate_report_output
from scripts.workbench_smoke import report_summary


PRIVATE_MARKER = "SYNTHETIC_PRIVATE_TRANSCRIPT_MARKER"


def test_workbench_summary_excludes_body_glossary_and_free_text() -> None:
    result = report_summary({"model": "qwen3-asr-1.7b", "text": PRIVATE_MARKER, "raw_text": PRIVATE_MARKER,
                             "quality": {"validated": False, "terms": [PRIVATE_MARKER], "note": PRIVATE_MARKER},
                             "timings": {"inference_seconds": 1.2, "path": PRIVATE_MARKER},
                             "deleted": True})
    assert PRIVATE_MARKER not in json.dumps(result)
    assert result["text_characters"] == len(PRIVATE_MARKER)
    assert len(result["text_sha256"]) == 64
    assert result["quality"] == {"validated": False}
    assert result["timings"] == {"inference_seconds": 1.2}


@pytest.mark.parametrize("include_content", [False, True])
def test_explicit_content_report_is_private_and_never_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], include_content: bool,
) -> None:
    output = tmp_path / "evidence/report.json"
    emit_report({"status": "passed"}, output=output, content={"text": PRIVATE_MARKER},
                include_content=include_content)
    captured = capsys.readouterr()
    assert PRIVATE_MARKER not in captured.out + captured.err
    assert (PRIVATE_MARKER in output.read_text()) is include_content
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="new file"):
        emit_report({}, output=output)


def test_content_requires_explicit_output_and_rejects_source_tree() -> None:
    with pytest.raises(ValueError, match="requires an explicit"):
        emit_report({}, content={"text": PRIVATE_MARKER}, include_content=True)
    source_output = Path(__file__).resolve().parents[1] / "docs/private-smoke-result.json"
    with pytest.raises(ValueError, match="ignored generated directory"):
        validate_report_output(source_output, include_content=True)


def test_backend_default_summary_does_not_print_transcription(capsys: pytest.CaptureFixture[str]) -> None:
    print_result_summary(argparse.Namespace(model="fixture-model", backend="transformers"),
                         language=PRIVATE_MARKER, text=PRIVATE_MARKER)
    captured = capsys.readouterr()
    assert PRIVATE_MARKER not in captured.out + captured.err
    assert json.loads(captured.out)["text_characters"] == len(PRIVATE_MARKER)
    assert json.loads(captured.out)["language"] == "unknown"


@pytest.mark.asyncio
async def test_pipeline_summary_excludes_timestamp_words_context_and_warning_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import qwen_pipeline_smoke as smoke

    body = {"model": "fixture-model", "text": PRIVATE_MARKER,
            "timestamps": [{"text": PRIVATE_MARKER, "start": 0.0, "end": 1.0}],
            "context": {"hotwords": [PRIVATE_MARKER]}, "timing": {"duration": 1.0},
            "generation": {"generated_tokens": 4}, "timings": {}, "warnings": [PRIVATE_MARKER]}
    audio = tmp_path / "synthetic.wav"
    audio.write_bytes(b"synthetic")

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> Client:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        async def post(self, *args: object, **kwargs: object) -> object:
            return SimpleNamespace(status_code=200, json=lambda: body)

    async def shutdown() -> None:
        pass

    monkeypatch.setattr(smoke, "create_app", lambda settings: SimpleNamespace(state=SimpleNamespace(manager=SimpleNamespace(shutdown=shutdown))))
    monkeypatch.setattr(smoke, "AsyncClient", Client)
    await smoke.run(argparse.Namespace(audio=audio, model="qwen3-asr-0.6b", timeout=1, language="auto",
                                       context="", keyword=[], reject_hallucinated_term=None))
    captured = capsys.readouterr()
    assert PRIVATE_MARKER not in captured.out + captured.err
    assert json.loads(captured.out)["timestamps_count"] == 1
