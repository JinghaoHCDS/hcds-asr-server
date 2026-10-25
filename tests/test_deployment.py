from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def deployment_sandbox(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    """Exercise deployment control flow without installing packages or contacting WSL."""
    checkout = tmp_path / 'checkout space % $name'
    (checkout / "deploy").mkdir(parents=True)
    for name in ("wsl-deploy.sh", "asr-server.service"):
        shutil.copyfile(REPO / "deploy" / name, checkout / "deploy" / name)
    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir()
    calls = tmp_path / "calls.jsonl"
    stub = f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path
name = Path(sys.argv[0]).name
if name == 'uname':
    print(os.environ['MOCK_OS'] if sys.argv[1] == '-s' else os.environ['MOCK_KERNEL'])
    raise SystemExit(0)
with open(os.environ['MOCK_CALLS'], 'a') as stream:
    stream.write(json.dumps([name, *sys.argv[1:]]) + '\\n')
if name in ('systemctl', 'curl') and os.environ.get('MOCK_SERVICE_STATE'):
    state = Path(os.environ['MOCK_SERVICE_STATE'])
    if name == 'curl':
        if state.exists():
            print('{{"status": "ok"}}')
            raise SystemExit(0)
        raise SystemExit(7)
    if 'is-active' in sys.argv:
        raise SystemExit(0 if state.exists() else 3)
    if 'start' in sys.argv:
        state.touch()
    if 'stop' in sys.argv:
        state.unlink(missing_ok=True)
if name == 'uv' and sys.argv[1] == 'sync':
    directory = Path('.venv/bin')
    directory.mkdir(parents=True, exist_ok=True)
    executable = directory / 'python'
    if not executable.exists():
        executable.symlink_to(sys.executable)
"""
    for name in ("uv", "systemctl", "nvidia-smi", "curl"):
        executable = mock_bin / name
        executable.write_text(stub, encoding="utf-8")
        executable.chmod(0o755)
    uname = mock_bin / "uname"
    uname.write_text(stub, encoding="utf-8")
    uname.chmod(0o755)
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("ASR_")
    }
    environment.update({
        "HOME": str(tmp_path / "home"),
        "PATH": f"{mock_bin}{os.pathsep}{os.environ['PATH']}",
        "MOCK_CALLS": str(calls),
        "MOCK_OS": "Darwin",
        "MOCK_KERNEL": "local-test-kernel",
    })
    return checkout, environment, calls


def run_deploy(checkout: Path, environment: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(checkout / "deploy/wsl-deploy.sh"),
         "--skip-system-packages", "--skip-tests", "--skip-mypy", "--skip-http-smoke", *args],
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


def recorded_calls(path: Path) -> list[list[str]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_deployment_uses_current_checkout_and_preserves_env(tmp_path: Path) -> None:
    checkout, environment, calls = deployment_sandbox(tmp_path)
    env_file = checkout / ".env"
    env_file.write_text("ASR_ENABLE_MOSS=0\n", encoding="utf-8")

    result = run_deploy(checkout, environment, "--mock")

    assert result.returncode == 0, result.stderr
    unit = (Path(environment["HOME"]) / ".config/systemd/user/asr-server.service").read_text()
    escaped_checkout = str(checkout).replace("%", "%%")
    assert f'WorkingDirectory="{escaped_checkout}"' in unit
    assert f'EnvironmentFile=-"{escaped_checkout}/.env"' in unit
    assert f'ExecStart="{escaped_checkout.replace("$", "$$")}/.venv/bin/uvicorn"' in unit
    assert "@PROJECT_DIR@" not in unit
    assert "Environment=HF_HUB_OFFLINE=0" in unit
    assert "Environment=ASR_ENABLE_MOSS=0" in unit
    assert env_file.read_text() == "ASR_ENABLE_MOSS=0\n"
    commands = recorded_calls(calls)
    assert ["uv", "sync", "--frozen", "--inexact"] in commands
    assert ["systemctl", "--user", "restart", "asr-server.service"] in commands
    assert not any(command[0] == "nvidia-smi" or command[:2] == ["uv", "pip"] for command in commands)


def test_offline_service_is_explicit(tmp_path: Path) -> None:
    checkout, environment, _ = deployment_sandbox(tmp_path)
    result = run_deploy(checkout, environment, "--mock", "--offline-service", "--no-start")

    assert result.returncode == 0, result.stderr
    unit = (Path(environment["HOME"]) / ".config/systemd/user/asr-server.service").read_text()
    assert "Environment=HF_HUB_OFFLINE=1" in unit
    assert "Environment=TRANSFORMERS_OFFLINE=1" in unit


def test_deployment_persists_advertised_skill_address(tmp_path: Path) -> None:
    checkout, environment, _ = deployment_sandbox(tmp_path)
    result = run_deploy(checkout, environment, "--mock", "--public-base-url", "http://my-asr.lan:18080", "--no-start")
    assert result.returncode == 0, result.stderr
    unit = (Path(environment["HOME"]) / ".config/systemd/user/asr-server.service").read_text()
    assert 'Environment="ASR_PUBLIC_BASE_URL=http://my-asr.lan:18080"' in unit


@pytest.mark.parametrize(("os_name", "kernel"), [("Darwin", "local-test"), ("Linux", "6.12-generic")])
@pytest.mark.parametrize("adapter", ["qwen", "registered"])
def test_real_gpu_deploy_fails_before_mutation_outside_wsl(tmp_path: Path, os_name: str, kernel: str, adapter: str) -> None:
    checkout, environment, calls = deployment_sandbox(tmp_path)
    environment.update(MOCK_OS=os_name, MOCK_KERNEL=kernel)

    result = run_deploy(checkout, environment, "--adapter", adapter)

    assert result.returncode != 0
    assert "WSL2" in result.stderr
    assert recorded_calls(calls) == []
    assert not (checkout / ".venv").exists()


@pytest.mark.parametrize("adapter", ["qwen", "registered"])
def test_wsl_gpu_install_uses_explicit_cuda_source(tmp_path: Path, adapter: str) -> None:
    checkout, environment, calls = deployment_sandbox(tmp_path)
    environment.update(MOCK_OS="Linux", MOCK_KERNEL="6.6.87-microsoft-standard-WSL2")

    result = run_deploy(checkout, environment, "--adapter", adapter, "--skip-cuda-check", "--skip-qwen-backend-smoke", "--no-enable-service")

    assert result.returncode == 0, result.stderr
    commands = recorded_calls(calls)
    assert ["nvidia-smi"] in commands
    assert ["uv", "pip", "install", "--torch-backend", "cu128", "-r", "requirements/wsl-gpu-cu128.txt"] in commands


@pytest.mark.parametrize("port", ["8001", "8765", "9999"])
def test_deployment_rejects_other_lan_ports_before_mutation(tmp_path: Path, port: str) -> None:
    checkout, environment, calls = deployment_sandbox(tmp_path)

    result = run_deploy(checkout, environment, "--mock", "--port", port)

    assert result.returncode != 0
    assert "18080" in result.stderr
    assert recorded_calls(calls) == []


def test_deployment_rejects_windows_mount_before_mutation(tmp_path: Path) -> None:
    checkout, environment, calls = deployment_sandbox(tmp_path)

    result = run_deploy(checkout, environment, "--mock", "--deploy-dir", "/mnt/c/asr-server")

    assert result.returncode != 0
    assert "WSL Linux filesystem" in result.stderr
    assert recorded_calls(calls) == []


def test_deployment_rejects_recursive_copy(tmp_path: Path) -> None:
    checkout, environment, calls = deployment_sandbox(tmp_path)

    result = run_deploy(checkout, environment, "--mock", "--deploy-dir", str(checkout / "nested"))

    assert result.returncode != 0
    assert "nested inside" in result.stderr
    assert recorded_calls(calls) == []


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is required for source-copy integration")
def test_copy_preserves_target_settings_and_excludes_runtime_data(tmp_path: Path) -> None:
    checkout, environment, _ = deployment_sandbox(tmp_path)
    target = tmp_path / "production"
    target.mkdir()
    (target / ".env").write_text("target setting\n")
    (target / "operator-notes.txt").write_text("preserve\n")
    (checkout / ".env").write_text("source secret\n")
    (checkout / ".env.production").write_text("source secret\n")
    (checkout / ".env.example").write_text("ASR_ENABLE_MOSS=0\n")
    for name in (".venv", "model_cache", "hf_cache", "logs", "tmp", "uploads", "transcripts"):
        (checkout / name).mkdir()
        (checkout / name / "runtime-file").write_text("runtime data\n")

    result = run_deploy(checkout, environment, "--mock", "--deploy-dir", str(target), "--no-enable-service")

    assert result.returncode == 0, result.stderr
    assert (target / ".env").read_text() == "target setting\n"
    assert not (target / ".env.production").exists()
    assert (target / ".env.example").exists()
    assert (target / "operator-notes.txt").read_text() == "preserve\n"
    assert not any((target / name / "runtime-file").exists() for name in (".venv", "model_cache", "hf_cache", "logs", "tmp", "uploads", "transcripts"))


def test_start_and_shutdown_use_bounded_local_health_checks(tmp_path: Path) -> None:
    checkout, environment, calls = deployment_sandbox(tmp_path)
    environment["MOCK_SERVICE_STATE"] = str(tmp_path / "service-running")
    environment["ASR_START_WAIT_SECONDS"] = "3"
    environment["ASR_SHUTDOWN_WAIT_SECONDS"] = "3"
    for name in ("start_asr_server.sh", "shutdown_asr_server.sh"):
        result = subprocess.run(
            ["bash", str(REPO / "scripts" / name)], cwd=checkout, env=environment,
            capture_output=True, text=True, check=False, timeout=10,
        )
        assert result.returncode == 0, result.stderr

    commands = recorded_calls(calls)
    assert ["systemctl", "--user", "start", "asr-server.service"] in commands
    assert ["systemctl", "--user", "stop", "asr-server.service"] in commands
    health_checks = [command for command in commands if command[0] == "curl"]
    assert health_checks
    assert all(command[-1] == "http://127.0.0.1:18080/health" for command in health_checks)
    assert all(command[1:7] == ["--noproxy", "*", "--connect-timeout", "2", "--max-time", "5"] for command in health_checks)
