from __future__ import annotations

import hashlib
import json
import math
import os
from uuid import uuid4
from dataclasses import asdict
from pathlib import Path
from typing import Any

from asr_server.quality import RECOVERY_POLICY_VERSION

from asr_server.adapters.base import ContextApplication, GenerationMetrics, TimestampItem, TranscriptionResult, TranscriptionSegment, TranscriptionTimings
from asr_server.errors import AsrError
from asr_server.observability import emit


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False)
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Checkpoints:
    def __init__(self, directory: Path | None, identity: str) -> None:
        self.directory = directory
        self.identity = identity
        self.extra_seconds = 0.0
        self.review_attempts: dict[str, int] = {}
        self.recovery: dict[str, Any] = {"policy_version": RECOVERY_POLICY_VERSION, "base_fingerprint": identity, "actions": {}, "calls": {}, "intervals": {}}
        if directory is None:
            return
        metadata = directory / "identity.json"
        if metadata.exists():
            if self._read(metadata).get("fingerprint") != identity:
                raise AsrError(409, "checkpoint_version_mismatch", "stored model, workflow, glossary or input differs; start a new task")
        else:
            atomic_json(metadata, {"fingerprint": identity})
        ledger = directory / "review-budget.json"
        if ledger.exists():
            stored = self._read(ledger)
            self.extra_seconds = float(stored["extra_seconds"])
            if not math.isfinite(self.extra_seconds) or self.extra_seconds < 0:
                raise AsrError(409, "checkpoint_corrupt", "persisted review budget is invalid")
            self.review_attempts = {str(key): int(value) for key, value in stored.get("attempts", {}).items()}
        state = directory / "recovery-v1" / "state.json"
        if state.exists():
            self.recovery = self._read(state)
            if self.recovery.get("base_fingerprint") != identity or self.recovery.get("policy_version") != RECOVERY_POLICY_VERSION:
                raise AsrError(409, "checkpoint_version_mismatch", "recovery policy or base identity differs")
            self.extra_seconds = max(self.extra_seconds, float(self.recovery.get("review_audio_seconds", 0)))
            for key, value in self.recovery.get("review_attempts", {}).items():
                self.review_attempts[key] = max(self.review_attempts.get(key, 0), int(value))

    def reserve_review(self, seconds: float, budget: float, key: str = "") -> None:
        if self.extra_seconds + seconds > budget + 0.001:
            raise AsrError(422, "review_budget_exhausted", "automatic acoustic review budget is exhausted")
        self.extra_seconds += seconds
        if key:
            if self.review_attempts.get(key, 0) >= 2:
                self.extra_seconds -= seconds
                raise AsrError(422, "review_budget_exhausted", "automatic review attempt limit is exhausted")
            self.review_attempts[key] = self.review_attempts.get(key, 0) + 1
        # Reserve before invoking the worker so crashes cannot reset cost.
        if self.directory is not None:
            atomic_json(self.directory / "review-budget.json", {"extra_seconds": self.extra_seconds, "attempts": self.review_attempts})

    def load(self, index: int) -> tuple[TranscriptionResult, list[dict[str, object]], float] | None:
        if self.directory is None:
            return None
        path = self.directory / f"chunk-{index:05d}.json"
        if not path.exists():
            path = self.directory / "recovery-v1" / f"chunk-{index:05d}.json"
        if not path.exists():
            return None
        data = self._read(path)
        if path.parent.name == "recovery-v1" and (data.get("base_fingerprint") != self.identity or data.get("recovery_policy_version") != RECOVERY_POLICY_VERSION):
            raise AsrError(409, "checkpoint_version_mismatch", "recovered chunk belongs to another execution")
        if path.parent.name == "recovery-v1" and data.get("chunk_index") != index:
            raise AsrError(409, "checkpoint_version_mismatch", "recovered chunk index differs")
        raw = data["result"]
        result = self.result_from_payload(raw)
        return result, data.get("reviews", []), float(data.get("extra_seconds", 0))

    @staticmethod
    def result_from_payload(raw: dict[str, Any]) -> TranscriptionResult:
        try:
            return TranscriptionResult(
                text=raw["text"], duration=raw["duration"], language=raw["language"], warnings=raw["warnings"],
                segments=[TranscriptionSegment(**item) for item in raw["segments"]],
                timestamps=[TimestampItem(**item) for item in raw["timestamps"]],
                timings=TranscriptionTimings(**raw["timings"]), generation=GenerationMetrics(**raw["generation"]),
                context=ContextApplication(**raw["context"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AsrError(409, "checkpoint_corrupt", "saved transcription payload is invalid") from exc

    def save(self, index: int, result: TranscriptionResult, reviews: list[dict[str, object]], extra_seconds: float) -> None:
        if self.directory is not None:
            atomic_json(self.directory / "recovery-v1" / f"chunk-{index:05d}.json", {"base_fingerprint": self.identity, "recovery_policy_version": RECOVERY_POLICY_VERSION, "chunk_index": index, "result": asdict(result), "reviews": reviews, "extra_seconds": extra_seconds})


    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("expected object")
            return value
        except (OSError, ValueError, TypeError) as exc:
            raise AsrError(409, "checkpoint_corrupt", "a persisted checkpoint cannot be read", {"artifact": path.name}) from exc

    def _persist(self) -> None:
        if self.directory is not None:
            atomic_json(self.directory / "recovery-v1" / "state.json", {**self.recovery, "calls": {}})

    def initialize_recovery(self, manifest: dict[str, object]) -> None:
        """Bind the independent recovery namespace to the immutable base plan."""
        previous = self.recovery.get("manifest")
        if previous is not None and previous != manifest:
            raise AsrError(409, "checkpoint_version_mismatch", "recovery execution manifest differs")
        if previous is None:
            legacy = {}
            if self.directory is not None:
                for path in sorted(self.directory.glob("chunk-*.json")):
                    self._read(path)
                    legacy[path.name] = file_digest(path)
            self.recovery["manifest"] = manifest
            self.recovery["legacy_checkpoint_digests"] = legacy
            self.recovery["legacy_review_attempts"] = dict(self.review_attempts)
            self.recovery["legacy_review_audio_seconds"] = self.extra_seconds
            self._persist()
        elif self.directory is not None:
            for name, expected in self.recovery.get("legacy_checkpoint_digests", {}).items():
                path = self.directory / name
                if not path.exists() or file_digest(path) != expected:
                    raise AsrError(409, "checkpoint_corrupt", "an imported base checkpoint changed", {"artifact": name})
        if self.directory is not None:
            imported = self.directory / "recovery-import.json"
            if imported.exists():
                data = self._read(imported)
                if data.get("base_fingerprint") != self.identity:
                    raise AsrError(409, "checkpoint_version_mismatch", "legacy failure evidence belongs to another base execution")
                if not data.get("source_error_digest"):
                    raise AsrError(409, "checkpoint_version_mismatch", "legacy failure import lacks source error digest")
                chunks = manifest.get("chunks", [])
                for failed in data.get("failed_chunks", []):
                    index = failed.get("index")
                    if not isinstance(index, int) or not isinstance(chunks, list) or index < 0 or index >= len(chunks):
                        raise AsrError(409, "checkpoint_version_mismatch", "legacy failed interval is invalid")
                    window = chunks[index]
                    if not isinstance(window, dict) or failed.get("start") != window["start"] or failed.get("end") != window["end"] or failed.get("cause") not in {"empty_transcription", "generation_truncated"}:
                        raise AsrError(409, "checkpoint_version_mismatch", "legacy failed interval does not match the base plan")
                    key = f"chunk:{index}:initial"
                    if self.call(key) is None:
                        self.recovery["calls"][key] = {"status": "error", "code": failed["cause"], "imported": True, "source_error_digest": data["source_error_digest"], "requested_language": manifest.get("requested_language")}
                        self._persist_call(key)
                self.recovery["legacy_failure_import_digest"] = file_digest(imported)
                self._persist()

    def action_used(self, index: int, action: str) -> bool:
        if action in {"split_2", "split_4"}:
            legacy_count = int(self.recovery.get("legacy_review_attempts", {}).get(f"chunk:{index}", 0))
            if legacy_count >= (1 if action == "split_2" else 2):
                return True
        return f"chunk:{index}:{action}" in self.recovery["actions"]

    def action_count(self, index: int) -> int:
        legacy = int(self.recovery.get("legacy_review_attempts", {}).get(f"chunk:{index}", 0))
        return legacy + sum(1 for key in self.recovery["actions"] if str(key).startswith(f"chunk:{index}:"))

    def can_reserve_action(self, index: int, action: str, seconds: float, budget: float) -> bool:
        return not self.action_used(index, action) and self.action_count(index) < 3 and self.extra_seconds + seconds <= budget + 0.001

    def reserve_action(self, index: int, action: str, seconds: float, budget: float, details: dict[str, object]) -> None:
        if self.action_used(index, action) or self.action_count(index) >= 3:
            raise AsrError(422, "review_budget_exhausted", "this quality recovery action has already been used")
        if self.extra_seconds + seconds > budget + 0.001:
            raise AsrError(422, "review_budget_exhausted", "automatic acoustic review budget is exhausted")
        # The independent policy state commits the action and debit together.
        # The legacy ledger is only a conservative compatibility mirror.
        before_seconds = self.extra_seconds
        self.extra_seconds += seconds
        if action.startswith("split_"):
            key = f"chunk:{index}"
            self.review_attempts[key] = self.review_attempts.get(key, 0) + 1
        self.recovery["actions"][f"chunk:{index}:{action}"] = {"seconds": seconds, "attempt_id": uuid4().hex, **details}
        self.recovery["review_audio_seconds"] = self.extra_seconds
        self.recovery["review_attempts"] = dict(self.review_attempts)
        self._persist()
        if self.directory is not None:
            atomic_json(self.directory / "review-budget.json", {"extra_seconds": self.extra_seconds, "attempts": self.review_attempts})
        record = self.recovery["actions"][f"chunk:{index}:{action}"]
        emit("quality.action_reserved", chunk_index=index, recovery_action=action,
            recovery_policy_version=RECOVERY_POLICY_VERSION, attempt_id=record["attempt_id"],
            requested_language=details.get("requested_language"), effective_language=details.get("effective_language"),
            source_chunk_indices=details.get("source_chunk_indices"), model_language_labels=details.get("model_language_labels"),
            review_audio_seconds=self.extra_seconds, review_budget_seconds=budget,
            budget_before_seconds=max(0, budget - before_seconds), budget_after_seconds=max(0, budget - self.extra_seconds))

    def call(self, key: str) -> dict[str, Any] | None:
        if key not in self.recovery["calls"] and self.directory is not None:
            path = self.directory / "recovery-v1" / "calls" / (fingerprint(key) + ".json")
            if path.exists():
                value = self._read(path)
                if value.get("base_fingerprint") != self.identity or value.get("call_key") != key:
                    raise AsrError(409, "checkpoint_version_mismatch", "saved recognition call belongs to another execution")
                self.recovery["calls"][key] = value
        value = self.recovery["calls"].get(key)
        return value if isinstance(value, dict) else None

    def _persist_call(self, key: str) -> None:
        if self.directory is not None:
            path = self.directory / "recovery-v1" / "calls" / (fingerprint(key) + ".json")
            self.recovery["calls"][key].update({"base_fingerprint": self.identity, "call_key": key})
            atomic_json(path, self.recovery["calls"][key])

    def begin_call(self, key: str, details: dict[str, object], *, budget: float, review_seconds: float = 0) -> None:
        prior = self.call(key)
        if prior is not None and prior.get("status") == "started":
            if prior.get("infrastructure_replays", 0) >= 1:
                raise AsrError(503, "recovery_attempt_exhausted", "interrupted model call exhausted its infrastructure retry")
            if review_seconds:
                self.reserve_review(review_seconds, budget)
                self.recovery["review_audio_seconds"] = self.extra_seconds
                self.recovery["review_attempts"] = dict(self.review_attempts)
                self._persist()
            prior["infrastructure_replays"] = prior.get("infrastructure_replays", 0) + 1
        else:
            self.recovery["calls"][key] = {"status": "started", "attempt_id": uuid4().hex, "infrastructure_replays": 0, **details}
        self._persist_call(key)
        record = self.recovery["calls"][key]
        emit("quality.call_started", call_key=key, attempt_id=record["attempt_id"],
            recovery_policy_version=RECOVERY_POLICY_VERSION, requested_language=record.get("requested_language"),
            effective_language=record.get("effective_language"), start_s=record.get("start"), end_s=record.get("end"),
            max_new_tokens=record.get("max_new_tokens"), review_audio_seconds=self.extra_seconds,
            review_budget_seconds=budget, reason="infrastructure_replay" if record.get("infrastructure_replays") else "initial_call")

    def finish_call(self, key: str, result: TranscriptionResult | None = None, *, error: str | None = None, evidence: dict[str, object] | None = None) -> None:
        record = self.recovery["calls"].setdefault(key, {})
        record.update({"status": "error" if error else "transcribed", "code": error})
        if result is not None:
            record["result"] = asdict(result)
        if evidence is not None:
            record["empty_evidence"] = evidence
        self._persist_call(key)
        emit("quality.call_persisted", call_key=key, attempt_id=record.get("attempt_id"),
            recovery_policy_version=RECOVERY_POLICY_VERSION, requested_language=record.get("requested_language"),
            effective_language=record.get("effective_language"), status=record["status"], error_code=error,
            review_audio_seconds=self.extra_seconds)

    def save_aligned_call(self, key: str, result: TranscriptionResult) -> None:
        self.recovery["calls"][key].update({"status": "aligned", "result": asdict(result)})
        self._persist_call(key)
        record = self.recovery["calls"][key]
        emit("quality.alignment_persisted", call_key=key, attempt_id=record.get("attempt_id"),
            recovery_policy_version=RECOVERY_POLICY_VERSION, requested_language=record.get("requested_language"),
            effective_language=record.get("effective_language"), status="aligned")

    def interval(self, index: int, record: dict[str, object]) -> None:
        self.recovery["intervals"][str(index)] = record
        self._persist()
