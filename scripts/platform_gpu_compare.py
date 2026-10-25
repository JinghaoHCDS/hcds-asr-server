"""Compare isolated cold-process workbench runs with one existing Python runtime.

No installer, server socket, production database, or .env loader is used. The
raw reports contain transcripts and must stay outside tracked source files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any


MODEL = "qwen3-asr-1.7b"
SMOKE_TIMEOUT = 600

# Import after inserting the selected source, in a brand-new isolated Python.
# An ordinary import keeps __package__='scripts'; workbench_smoke's CLI fallback
# therefore cannot add the invoking/current checkout ahead of the baseline.
CHILD_BOOTSTRAP = r'''
import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

configuration = json.loads(sys.argv[1])
source = Path(configuration["source"]).resolve()
sys.path.insert(0, str(source))
smoke = importlib.import_module("scripts.workbench_smoke")
import asr_server
assert Path(smoke.__file__).resolve() == source / "scripts" / "workbench_smoke.py"
assert Path(asr_server.__file__).resolve() == source / "asr_server" / "__init__.py"
pipeline = importlib.import_module("asr_server.quality.pipeline")
original_glossary_context = pipeline.glossary_context
prompt_digests, glossary_digests = set(), set()
def observe_glossary(context, hotwords, snapshot):
    prompt, report = original_glossary_context(context, hotwords, snapshot)
    prompt_digests.add(hashlib.sha256(prompt.encode()).hexdigest())
    glossary_digests.add(hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
    return prompt, report
pipeline.glossary_context = observe_glossary
if configuration["log_level"] is not None:
    original_settings = smoke.Settings
    def selected_settings(**kwargs):
        return original_settings(**kwargs, log_level=configuration["log_level"])
    smoke.Settings = selected_settings

arguments = argparse.Namespace(audio=Path(configuration["audio"]), model=configuration["model"],
                               mock=configuration["mock"], timeout=configuration["timeout"])
started = time.perf_counter()
report = asyncio.run(smoke.run(arguments))
smoke_wall = time.perf_counter() - started
Path(configuration["report"]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
versions = {}
for name in ("torch", "transformers", "fastapi", "starlette", "silero-vad", "httpx"):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
runtime = {"python": platform.python_version(), "versions": versions,
           "smoke_wall_seconds": smoke_wall, "source_import_verified": True,
           "prompt_sha256": sorted(prompt_digests), "glossary_sha256": sorted(glossary_digests),
           "helper_sha256": hashlib.sha256(Path(smoke.__file__).read_bytes()).hexdigest()}
Path(configuration["runtime"]).write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
'''


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_identity(source: Path) -> dict[str, Any]:
    files = sorted((source / "asr_server").rglob("*.py")) + [source / "scripts" / "workbench_smoke.py"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(digest_file(path).encode())
        digest.update(b"\n")
    return {"label": source.name, "runtime_source_sha256": digest.hexdigest(),
            "helper_sha256": digest_file(source / "scripts" / "workbench_smoke.py"), "files_hashed": len(files)}


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as target:
        os.chmod(path, 0o600)
        json.dump(value, target, ensure_ascii=False, indent=2)
        target.write("\n")


def require_untracked_output(path: Path, sources: list[Path]) -> None:
    for source in sources:
        if not path.is_relative_to(source):
            continue
        root = subprocess.run(["git", "-C", str(source), "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, check=False)
        if root.returncode != 0:
            continue  # A source export has no repository that can track output.
        ignored = subprocess.run(["git", "-C", str(source), "check-ignore", "--quiet", "--", str(path)], check=False)
        if ignored.returncode != 0:
            raise ValueError("output inside a Git source tree must be ignored; choose data/, tmp/, or an external directory")


def child_environment(scratch: Path, quality_vad: Path | None) -> dict[str, str]:
    # Keep the existing interpreter's model cache/runtime configuration, while
    # preventing ASR service state and credentials from entering the smoke app.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("ASR_") and key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update({"TMPDIR": str(scratch), "TEMP": str(scratch), "TMP": str(scratch),
                        "ASR_TEMP_DIR": str(scratch / "temporary"), "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                        "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONUNBUFFERED": "1"})
    if quality_vad is not None:
        environment["ASR_QUALITY_VAD_DIR"] = str(quality_vad)
    return environment


def stop_process(process: subprocess.Popen[bytes]) -> None:
    def terminate(sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except ProcessLookupError:
            pass
    terminate(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        terminate(signal.SIGKILL)
        process.wait(timeout=10)


def run_once(*, source: Path, label: str, iteration: int, audio: Path, duration: float,
             output: Path, scratch: Path, mock: bool, log_level: str | None,
             quality_vad: Path | None) -> dict[str, Any]:
    name = f"{label}-{iteration:02d}"
    work = scratch / name
    work.mkdir(mode=0o700)
    report_path = output / f"{name}.report.json"
    runtime_path = output / f"{name}.runtime.json"
    configuration = {"source": str(source), "audio": str(audio), "report": str(report_path),
                     "runtime": str(runtime_path), "model": MODEL, "mock": mock,
                     "timeout": SMOKE_TIMEOUT, "log_level": log_level}
    started = time.perf_counter()
    timed_out = False
    with (output / f"{name}.stdout.log").open("xb") as stdout, (output / f"{name}.stderr.log").open("xb") as stderr:
        process = subprocess.Popen([sys.executable, "-I", "-c", CHILD_BOOTSTRAP, json.dumps(configuration)],
            cwd=work, env=child_environment(work, quality_vad), stdout=stdout, stderr=stderr,
            start_new_session=os.name == "posix")
        try:
            process.wait(timeout=SMOKE_TIMEOUT + 180)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_process(process)
        except BaseException:
            stop_process(process)
            raise
    wall = time.perf_counter() - started
    metrics: dict[str, Any] = {"run": name, "source": label, "iteration": iteration, "exit_code": process.returncode,
        "timed_out": timed_out, "process_wall_seconds": wall, "workflow_rtf_end_to_end": wall / duration,
        "model_state": "mock adapter in a fresh gateway; no GPU model" if mock else "cold: new gateway/model worker processes",
        "disk_model_cache": "unused by mock" if mock else "existing cache; offline-only",
        "logging_level": log_level, "raw_report": report_path.name,
        "stdout": name + ".stdout.log", "stderr": name + ".stderr.log"}
    if process.returncode == 0:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        text = str(report["text"])
        quality = report.get("quality") or {}
        metrics.update({"text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text_characters": len(text),
            "raw_text_characters": len(str(report.get("raw_text") or "")), "timestamp_count": report.get("timestamps"),
            "timestamp_sources": report.get("timestamp_sources"), "segment_count": report.get("segments"),
            "model_revision": quality.get("model_revision"), "aligner_revision": quality.get("aligner_revision"),
            "speech_detector": quality.get("speech_detector"), "profile_version": quality.get("profile_version"),
            "edit_version": report.get("edit_version"), "edit_alignment": report.get("edit_alignment"),
            "exports": report.get("exports"), "deleted": report.get("deleted"),
            "smoke_wall_seconds": runtime["smoke_wall_seconds"], "versions": runtime["versions"],
            "prompt_sha256": runtime["prompt_sha256"], "glossary_sha256": runtime["glossary_sha256"],
            "python": runtime["python"], "source_import_verified": runtime["source_import_verified"]})
    write_json(output / f"{name}.metrics.json", metrics)
    if process.returncode != 0 or timed_out:
        raise RuntimeError(f"{name} failed (exit {process.returncode}, timeout={timed_out}); inspect its saved stderr log")
    return metrics


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [float(run["process_wall_seconds"]) for run in runs]
    rtfs = [float(run["workflow_rtf_end_to_end"]) for run in runs]
    return {"runs": len(runs), "median_process_wall_seconds": statistics.median(walls),
            "min_process_wall_seconds": min(walls), "max_process_wall_seconds": max(walls),
            "median_workflow_rtf_end_to_end": statistics.median(rtfs),
            "text_digests": sorted({str(run["text_sha256"]) for run in runs})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--current-source", type=Path, required=True)
    parser.add_argument("--audio180s", type=Path, required=True, help="same local 180-second WAV; --mock also accepts a short harness fixture")
    parser.add_argument("--output-dir", type=Path, required=True, help="new ignored or external directory; raw transcripts stay here")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mock", action="store_true", help="exercise the same workflow without a GPU or model installation")
    parser.add_argument("--compare-logging", action="store_true", help="also run current CRITICAL, for an exploratory INFO/CRITICAL comparison")
    parser.add_argument("--quality-vad-dir", type=Path, help="explicit shared read-only Silero model bundle for both source trees")
    args = parser.parse_args()
    baseline, current, audio, output = (path.resolve() for path in (args.baseline_source, args.current_source, args.audio180s, args.output_dir))
    if args.repeats < 3:
        parser.error("at least three repetitions per source are required")
    for source in (baseline, current):
        if not (source / "asr_server" / "__init__.py").is_file() or not (source / "scripts" / "workbench_smoke.py").is_file():
            parser.error("both source trees must contain asr_server and scripts/workbench_smoke.py")
    if output.exists():
        parser.error("output-dir must be a new directory")
    quality_vad = args.quality_vad_dir.resolve() if args.quality_vad_dir else None
    if quality_vad is not None and not quality_vad.is_dir():
        parser.error("quality-vad-dir must be an existing model directory")
    try:
        require_untracked_output(output, [baseline, current])
        with wave.open(str(audio), "rb") as track:
            duration = track.getnframes() / track.getframerate()
        if duration <= 0:
            parser.error("audio must contain at least one frame")
        if not args.mock and abs(duration - 180.0) > 0.1:
            parser.error("audio180s must be a 180-second WAV (tolerance 0.1 second)")
        identities = {"baseline": source_identity(baseline), "current_info": source_identity(current)}
        if identities["baseline"]["helper_sha256"] != identities["current_info"]["helper_sha256"]:
            parser.error("smoke helpers differ; align their workload before comparing source trees")
    except (OSError, wave.Error, ValueError) as exc:
        parser.error(str(exc))
    output.mkdir(parents=True, mode=0o700)
    scratch = output / "scratch"
    scratch.mkdir(mode=0o700)
    inputs = scratch / "input"
    frozen_audio = inputs / audio.name
    summary: dict[str, Any] = {
        "schema_version": 1, "mock": args.mock, "status": "running", "model": MODEL,
        "audio": {"sha256": digest_file(audio), "duration_seconds": duration, "size_bytes": audio.stat().st_size},
        "sources": identities, "repeats": args.repeats, "python": platform.python_version(),
        "workload": "workbench smoke: login/resumable upload, quality_v1, alignment, edits, exports and deletion",
        "validation_scope": "mock harness exercise only; no GPU throughput or accuracy claim" if args.mock else "real GPU workflow comparison",
        "parameters": {"language": "auto", "domain": "ai", "hotwords": ["Qwen3-ASR", "ABSENTTERMZXQ"],
                       "quality_profile": "quality_v1", "timeout_seconds": SMOKE_TIMEOUT},
        "cache": "every run starts a fresh gateway/model process; existing disk cache, downloads disabled",
        "order": "all baseline repetitions, then all current INFO repetitions; optional current CRITICAL last",
        "interpretation": "workflow wall time includes loading, upload, recognition, edit alignment, export and cleanup; baseline text is not accuracy ground truth",
        "logging_interpretation": "small-sample exploratory comparison; it cannot establish a stable logging overhead below 3 percent",
        "runs": [], "aggregates": {},
    }
    groups: list[tuple[str, Path, str | None]] = [("baseline", baseline, None), ("current_info", current, "INFO")]
    if args.compare_logging:
        groups.append(("current_critical", current, "CRITICAL"))
    try:
        inputs.mkdir(mode=0o700)
        shutil.copyfile(audio, frozen_audio)
        os.chmod(frozen_audio, 0o400)
        if digest_file(frozen_audio) != summary["audio"]["sha256"]:
            raise RuntimeError("audio changed while freezing the comparison input")
        for label, source, level in groups:
            group_runs = []
            for iteration in range(1, args.repeats + 1):
                run = run_once(source=source, label=label, iteration=iteration, audio=frozen_audio, duration=duration,
                    output=output, scratch=scratch, mock=args.mock, log_level=level, quality_vad=quality_vad)
                group_runs.append(run)
                summary["runs"].append(run)
                print(f"{run['run']}: {run['process_wall_seconds']:.3f}s, workflow RTF {run['workflow_rtf_end_to_end']:.4f}", flush=True)
            summary["aggregates"][label] = aggregate(group_runs)
        aggregates = summary["aggregates"]
        fields = ("model_revision", "aligner_revision", "profile_version", "speech_detector", "prompt_sha256", "glossary_sha256", "versions")
        identities_seen = {json.dumps({field: run.get(field) for field in fields}, sort_keys=True) for run in summary["runs"]}
        summary["frozen_workload_identity_matches"] = len(identities_seen) == 1
        if not summary["frozen_workload_identity_matches"]:
            raise RuntimeError("model/profile/prompt/glossary/runtime identity differs; saved runs cannot establish the requested controlled comparison")
        summary["current_vs_baseline_wall_change_percent"] = (
            aggregates["current_info"]["median_process_wall_seconds"] / aggregates["baseline"]["median_process_wall_seconds"] - 1) * 100
        if args.compare_logging:
            summary["observed_info_vs_critical_wall_change_percent"] = (
                aggregates["current_info"]["median_process_wall_seconds"] / aggregates["current_critical"]["median_process_wall_seconds"] - 1) * 100
        summary["status"] = "completed"
    except BaseException as exc:
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        summary["error"] = str(exc) if isinstance(exc, (OSError, RuntimeError, ValueError)) else type(exc).__name__
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
