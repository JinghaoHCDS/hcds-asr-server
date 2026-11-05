"""Version-local speaker identities and compact, non-summarizing dialogue export."""
from __future__ import annotations

import copy
import re
from typing import Any


def ensure_speakers(result: dict[str, Any]) -> None:
    if "speakers" in result:
        return
    speakers: dict[str, Any] = {}
    identities: dict[tuple[str, str], str] = {}
    for segment in result.get("segments", []):
        source = segment.get("speaker")
        scope = segment.get("speaker_scope", result.get("execution", {}).get("speaker_scope", "chunk"))
        key = (str(source), "global" if scope == "global" else str(segment.get("chunk_index", 0)))
        speaker_id = None
        if source is not None:
            if key not in identities:
                speaker_id = f"P{len(speakers) + 1}"
                identities[key] = speaker_id
                speakers[speaker_id] = {"name": f"说话人 {len(speakers) + 1}", "scope": scope,
                                        "source_speaker": source, "chunk_index": segment.get("chunk_index")}
            speaker_id = identities[key]
        segment["speaker_id"] = speaker_id
        segment["model_speaker_id"] = speaker_id
    result["speakers"] = speakers


def speaker_view(result: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(result)
    ensure_speakers(result)
    return result


def pending_speaker(segment: dict[str, Any], speakers: dict[str, Any]) -> bool:
    if not segment.get("speaker_id"):
        return True
    if segment.get("speaker_assignment") == "user":
        return False
    return (speakers.get(segment["speaker_id"], {}).get("scope") != "global"
            or segment.get("speaker_resolution") in {"new_candidate", "unresolved"})


def _plain(value: str) -> str:
    # Keep transcript characters; escape only Markdown syntax/control newlines in labels.
    return re.sub(r"([\\`*_{}\[\]<>#|])", r"\\\1", value.replace("\r", " ").replace("\n", " "))


def _clock(value: float) -> str:
    milliseconds = round(max(0, value) * 1000)
    seconds, ms = divmod(milliseconds, 1000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def dialogue_markdown(original: dict[str, Any], filename: str) -> str:
    result = speaker_view(original)
    speakers = result["speakers"]
    segments = sorted(result.get("segments", []), key=lambda s: (s["start"], s["end"]))
    gaps = result.get("gaps") or []
    incomplete = result.get("completeness", {}).get("complete") is False or bool(gaps)
    pending = any(pending_speaker(s, speakers) for s in segments)
    lines = [f"# {_plain(filename)}", "", f"版本：{result.get('version', 1)}｜转录：{'不完整' if incomplete else '完整'}｜人物归属：{'部分待确认' if pending else '已标注（不代表全部人工确认）'}", ""]
    for key, person in speakers.items():
        scope = f"（分块内：{person.get('source_speaker', key)}）" if person.get("scope") == "chunk" else ""
        lines.append(f"- {key}＝{_plain(person['name'])}{_plain(scope)}")
    lines.append("")
    for gap in gaps:
        lines.append(f"缺口 [{_clock(gap['start'])}–{_clock(gap['end'])}]：待回听确认")
    # Interval sweep avoids quadratic work on hour-long transcripts.
    overlap_flags: list[bool] = []
    furthest_end = float('-inf')
    for index, segment in enumerate(segments):
        start, end = float(segment["start"]), float(segment["end"])
        overlap_flags.append(start < furthest_end or (index + 1 < len(segments) and float(segments[index + 1]["start"]) < end))
        furthest_end = max(furthest_end, end)
    turns: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not segment.get("text", "").strip():
            if turns:
                turns[-1]["blocked"] = True
            continue
        start, end = float(segment["start"]), float(segment["end"])
        uncertain = pending_speaker(segment, speakers)
        overlapping = overlap_flags[index]
        crossing_gap = any(float(g["start"]) <= end and float(g["end"]) >= (turns[-1]["end"] if turns else start) for g in gaps)
        previous = turns[-1] if turns else None
        if (previous and not uncertain and not overlapping and not previous["blocked"] and not crossing_gap
                and previous["speaker_id"] == segment.get("speaker_id")
                and 0 <= start - previous["end"] <= 2 and end - previous["start"] <= 60
                and len(previous["text"]) + len(segment["text"]) + 1 <= 800):
            previous["text"] += "\n" + segment["text"]
            previous["end"] = end
        else:
            turns.append({"start": start, "end": end, "text": segment["text"],
                          "speaker_id": segment.get("speaker_id"), "pending": uncertain,
                          "source_speaker": segment.get("source_speaker"),
                          "blocked": uncertain or overlapping or crossing_gap})
    for turn in turns:
        key = turn["speaker_id"]
        name = f"{key} {_plain(speakers[key]['name'])}" if key in speakers else "待确认"
        if key and turn["pending"]:
            name += "（归属待确认）"
        elif not key and turn["source_speaker"]:
            name += f"（分块内：{_plain(str(turn['source_speaker']))}）"
        lines.extend(["", f"[{_clock(turn['start'])}–{_clock(turn['end'])}] {name}", _plain(turn["text"])])
    if not segments and result.get("text"):
        lines.extend(["", "未分段正文（说话人待确认）", _plain(result["text"])])
    return "\n".join(lines).strip() + "\n"
