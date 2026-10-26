from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

AdapterMode = Literal["mock", "qwen", "registered"]


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 18080
    public_base_url: str = "http://127.0.0.1:18080"
    api_token: str | None = field(default=None, repr=False)
    default_model: str = "qwen3-asr-1.7b"
    adapter: AdapterMode = "mock"
    enable_moss: bool = False
    enable_qwen_alignment: bool = False
    qwen_batch_size: int = 1
    job_result_ttl_seconds: int = 3600
    idle_unload_seconds: float = 180.0
    max_queued_jobs: int = 20
    max_upload_mb: int = 512
    ffprobe_timeout_seconds: float = 30.0
    ffmpeg_timeout_seconds: float = 1800.0
    worker_startup_timeout_seconds: float = 30.0
    worker_load_timeout_seconds: float = 900.0
    worker_inference_timeout_seconds: float = 3600.0
    # Phase limits keep the previous envelope until target-machine measurements
    # justify tighter values. Observation emits diagnostics without termination.
    worker_chunk_timeout_seconds: float | None = None
    worker_alignment_timeout_seconds: float | None = None
    worker_native_timeout_seconds: float | None = None
    worker_deadline_observation_seconds: float = 300.0
    worker_circuit_failure_limit: int = 3
    worker_circuit_cooldown_seconds: float = 60.0
    gateway_watchdog_seconds: float = 0.0
    worker_shutdown_timeout_seconds: float = 10.0
    job_shutdown_grace_seconds: float = 30.0
    min_free_disk_mb: int = 5120
    max_spool_mb: int = 12288
    max_workspace_mb: int = 1280
    max_workspace_files: int = 8
    receive_slots: int = 4
    receive_timeout_seconds: float = 30.0
    multipart_metadata_bytes: int = 256 * 1024
    preprocessing_prefetch: bool = False
    log_level: str = "INFO"
    log_format: str = "json"
    log_dir: Path | None = None
    workbench_enabled: bool = True
    workbench_data_dir: Path = Path("data/workbench")
    workbench_retention_days: int = 7
    workbench_max_upload_bytes: int = 8 * 1024**3
    workbench_total_quota_bytes: int = 64 * 1024**3
    workbench_user_quota_bytes: int = 16 * 1024**3


def load_settings() -> Settings:
    adapter = os.getenv("ASR_ADAPTER", "mock")
    if adapter not in {"mock", "qwen", "registered"}:
        raise ValueError("ASR_ADAPTER must be 'mock', 'qwen' or 'registered'")
    adapter_mode = cast(AdapterMode, adapter)
    enable_moss = _env_bool("ASR_ENABLE_MOSS", default=False)
    enable_qwen_alignment = _env_bool("ASR_ENABLE_QWEN_ALIGNMENT", default=False)
    qwen_batch_size = int(os.getenv("ASR_QWEN_BATCH_SIZE", "1"))
    if qwen_batch_size < 1:
        raise ValueError("ASR_QWEN_BATCH_SIZE must be greater than or equal to 1")
    job_result_ttl_seconds = int(os.getenv("ASR_JOB_RESULT_TTL_SECONDS", "3600"))
    if job_result_ttl_seconds < 1:
        raise ValueError("ASR_JOB_RESULT_TTL_SECONDS must be greater than or equal to 1")
    idle_unload_seconds = float(os.getenv("ASR_IDLE_UNLOAD_SECONDS", "180"))
    if idle_unload_seconds < 0:
        raise ValueError("ASR_IDLE_UNLOAD_SECONDS must be greater than or equal to 0")
    max_queued_jobs = int(os.getenv("ASR_MAX_QUEUED_JOBS", "20"))
    if max_queued_jobs < 1:
        raise ValueError("ASR_MAX_QUEUED_JOBS must be greater than or equal to 1")
    max_upload_mb = int(os.getenv("ASR_MAX_UPLOAD_MB", "512"))
    if max_upload_mb < 1:
        raise ValueError("ASR_MAX_UPLOAD_MB must be greater than or equal to 1")
    ffprobe_timeout_seconds = _positive_float("ASR_FFPROBE_TIMEOUT_SECONDS", 30.0)
    ffmpeg_timeout_seconds = _positive_float("ASR_FFMPEG_TIMEOUT_SECONDS", 1800.0)
    worker_startup_timeout_seconds = _positive_float("ASR_WORKER_STARTUP_TIMEOUT_SECONDS", 30.0)
    worker_load_timeout_seconds = _positive_float("ASR_WORKER_LOAD_TIMEOUT_SECONDS", 900.0)
    worker_inference_timeout_seconds = _positive_float("ASR_WORKER_INFERENCE_TIMEOUT_SECONDS", 3600.0)
    worker_shutdown_timeout_seconds = _positive_float("ASR_WORKER_SHUTDOWN_TIMEOUT_SECONDS", 10.0)
    job_shutdown_grace_seconds = _positive_float("ASR_JOB_SHUTDOWN_GRACE_SECONDS", 30.0)
    min_free_disk_mb = int(os.getenv("ASR_MIN_FREE_DISK_MB", "5120"))
    max_spool_mb = int(os.getenv("ASR_MAX_SPOOL_MB", "12288"))
    max_workspace_mb = int(os.getenv("ASR_MAX_WORKSPACE_MB", "1280"))
    max_workspace_files = int(os.getenv("ASR_MAX_WORKSPACE_FILES", "8"))
    if min_free_disk_mb < 0 or max_spool_mb < 1 or max_workspace_mb < 1 or max_workspace_files < 1:
        raise ValueError("ASR disk and workspace limits must be non-negative/positive")
    workbench_values = {
        "workbench_retention_days": int(os.getenv("ASR_WORKBENCH_RETENTION_DAYS", "7")),
        "workbench_max_upload_bytes": int(os.getenv("ASR_WORKBENCH_MAX_UPLOAD_BYTES", str(8 * 1024**3))),
        "workbench_total_quota_bytes": int(os.getenv("ASR_WORKBENCH_TOTAL_QUOTA_BYTES", str(64 * 1024**3))),
        "workbench_user_quota_bytes": int(os.getenv("ASR_WORKBENCH_USER_QUOTA_BYTES", str(16 * 1024**3))),
    }
    if any(value <= 0 for value in workbench_values.values()):
        raise ValueError("ASR_WORKBENCH limits must be positive")
    return Settings(
        receive_slots=_positive_int("ASR_RECEIVE_SLOTS", 4),
        receive_timeout_seconds=_positive_float("ASR_RECEIVE_TIMEOUT_SECONDS", 30.0),
        multipart_metadata_bytes=_positive_int("ASR_MULTIPART_METADATA_BYTES", 256 * 1024),
        preprocessing_prefetch=_env_bool("ASR_PREPROCESSING_PREFETCH", default=False),
        log_level=os.getenv("ASR_LOG_LEVEL", "INFO"),
        log_format=os.getenv("ASR_LOG_FORMAT", "json"),
        log_dir=Path(os.environ["ASR_LOG_DIR"]) if os.getenv("ASR_LOG_DIR") else None,
        host=os.getenv("ASR_HOST", "0.0.0.0"),
        port=int(os.getenv("ASR_PORT", "18080")),
        public_base_url=os.getenv("ASR_PUBLIC_BASE_URL", "http://127.0.0.1:18080"),
        api_token=os.getenv("ASR_API_TOKEN") or None,
        default_model=os.getenv("ASR_DEFAULT_MODEL", "qwen3-asr-1.7b"),
        adapter=adapter_mode,
        enable_moss=enable_moss,
        enable_qwen_alignment=enable_qwen_alignment,
        qwen_batch_size=qwen_batch_size,
        job_result_ttl_seconds=job_result_ttl_seconds,
        idle_unload_seconds=idle_unload_seconds,
        max_queued_jobs=max_queued_jobs,
        max_upload_mb=max_upload_mb,
        ffprobe_timeout_seconds=ffprobe_timeout_seconds,
        ffmpeg_timeout_seconds=ffmpeg_timeout_seconds,
        worker_startup_timeout_seconds=worker_startup_timeout_seconds,
        worker_load_timeout_seconds=worker_load_timeout_seconds,
        worker_inference_timeout_seconds=worker_inference_timeout_seconds,
        worker_shutdown_timeout_seconds=worker_shutdown_timeout_seconds,
        worker_chunk_timeout_seconds=_positive_float("ASR_WORKER_CHUNK_TIMEOUT_SECONDS", worker_inference_timeout_seconds),
        worker_alignment_timeout_seconds=_positive_float("ASR_WORKER_ALIGNMENT_TIMEOUT_SECONDS", worker_inference_timeout_seconds),
        worker_native_timeout_seconds=_positive_float("ASR_WORKER_NATIVE_TIMEOUT_SECONDS", worker_inference_timeout_seconds),
        worker_deadline_observation_seconds=_positive_float("ASR_WORKER_DEADLINE_OBSERVATION_SECONDS", 300.0),
        worker_circuit_failure_limit=_positive_int("ASR_WORKER_CIRCUIT_FAILURE_LIMIT", 3),
        worker_circuit_cooldown_seconds=_positive_float("ASR_WORKER_CIRCUIT_COOLDOWN_SECONDS", 60.0),
        gateway_watchdog_seconds=_nonnegative_float("ASR_GATEWAY_WATCHDOG_SECONDS", 0.0),
        job_shutdown_grace_seconds=job_shutdown_grace_seconds,
        min_free_disk_mb=min_free_disk_mb,
        max_spool_mb=max_spool_mb,
        max_workspace_mb=max_workspace_mb,
        max_workspace_files=max_workspace_files,
        workbench_enabled=_env_bool("ASR_WORKBENCH_ENABLED", default=True),
        workbench_data_dir=Path(os.getenv("ASR_WORKBENCH_DATA_DIR", "data/workbench")),
        workbench_retention_days=workbench_values["workbench_retention_days"],
        workbench_max_upload_bytes=workbench_values["workbench_max_upload_bytes"],
        workbench_total_quota_bytes=workbench_values["workbench_total_quota_bytes"],
        workbench_user_quota_bytes=workbench_values["workbench_user_quota_bytes"],
    )


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than 0")
    return value


def _nonnegative_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value
