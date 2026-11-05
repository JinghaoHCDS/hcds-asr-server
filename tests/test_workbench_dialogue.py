from __future__ import annotations

import copy
from typing import Any

import pytest

from asr_server.workbench.speakers import ensure_speakers, dialogue_markdown


def transcript() -> dict[str, Any]:
    return {"text": "甲乙丙", "version": 1, "segments": [
        {"id": "s0", "start": 0, "end": 2, "speaker": "S01", "speaker_scope": "global", "text": "甲"},
        {"id": "s1", "start": 3, "end": 4, "speaker": "S01", "speaker_scope": "global", "text": "乙"},
        {"id": "s2", "start": 5, "end": 7, "speaker": "S02", "speaker_scope": "global", "text": "丙"},
    ]}


def test_speaker_mapping_scopes_and_original_assignment() -> None:
    result = transcript()
    result["segments"][0].update(speaker_scope="chunk", chunk_index=0)
    result["segments"][1].update(speaker_scope="chunk", chunk_index=1)
    ensure_speakers(result)
    assert result["segments"][0]["speaker_id"] != result["segments"][1]["speaker_id"]
    result["segments"][0]["speaker_id"] = "P3"
    ensure_speakers(result)
    assert result["segments"][0]["model_speaker_id"] == "P1"
    assert result["segments"][0]["speaker_id"] == "P3"


def test_markdown_merges_adjacent_speaker_without_mutating_version() -> None:
    result = transcript()
    original = copy.deepcopy(result)
    text = dialogue_markdown(result, "interview.mp4")
    assert "[00:00:00.000–00:00:04.000] P1" in text
    assert "甲 乙" in text  # joining doesn't summarize or remove either segment
    assert result == original


@pytest.mark.parametrize("case", ["overlap", "gap", "candidate", "unknown", "long", "large", "empty_turn"])
def test_markdown_preserves_review_boundaries(case: str) -> None:
    result = transcript()
    if case == "overlap":
        result["segments"][1]["start"] = 1
    elif case == "gap":
        result["gaps"] = [{"start": 2, "end": 3}]
    elif case == "candidate":
        result["segments"][1]["speaker_resolution"] = "new_candidate"
    elif case == "unknown":
        result["segments"][1]["speaker"] = None
    elif case == "long":
        result["segments"][1]["end"] = 61
    elif case == "large":
        result["segments"][1]["text"] = "乙" * 800
    else:
        result["segments"].insert(1, {"start": 2, "end": 2.5, "speaker": None, "text": ""})
    text = dialogue_markdown(result, "interview.mp4")
    assert "[00:00:00.000–00:00:02.000]" in text


def test_markdown_preserves_unresolved_local_label() -> None:
    result = transcript()
    result["segments"][1].update(speaker=None, source_speaker="chunk-0001:Speaker 2", speaker_resolution="unresolved")
    assert "待确认（分块内：chunk-0001:Speaker 2）" in dialogue_markdown(result, "interview.mp4")
