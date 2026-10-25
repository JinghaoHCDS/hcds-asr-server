# 服务端 Agent 提示词

> 2026-10-25 平台扩展遵循 [平台运行契约](../docs/asr-platform-runtime.md)：所有 HTTP 推理共享 FIFO，短同步也会排队；工作台可显式选择支持的模型/profile，并提供摘要分页、revision/ETag；日志可按 job/trace 关联。历史内存-only限制只适用于旧 `/v1` owner。

> 2026-10 扩展：工作台 `/api/**` 使用 SQLite、个人 Cookie 和七天保留；旧 `/v1/**` 继续内存任务与原鉴权，新增 `quality_profile` 默认 `legacy`。历史“不做网页/数据库”仅适用旧阶段，以 [工作台契约](../docs/product-workbench.md) 和项目 AGENTS 为准。

部署路径和 LAN IP 由用户环境决定；示例 `YOUR_WINDOWS_LAN_IP` 与 `/path/to/` 必须替换。历史 Mac / Arch 验收不代表其他平台已实测。

你是在 Windows 主机的 WSL Arch Linux 内工作的开发代理。你的任务是实现一个常驻后台的 ASR 服务，让局域网 Agent 和客户端 可以访问 Windows/WSL 里的 GPU ASR 模型。

工作目录：

```bash
~/services/asr-server
```

目标入口：

```text
http://YOUR_WINDOWS_LAN_IP:18080
```

必须阅读并遵守 PRD：

```text
docs/asr-server-prd.md
```

维护边界和当前实现说明见 WSL 项目总览：

```text
prompts/wsl-project-brief.md
```

当前真实模型验收记录见：

```text
docs/validation-2026-07-02-wsl-hf-native.md
docs/validation-2026-07-10-wsl-moss.md
docs/validation-2026-07-15-wsl-moss-long-form.md
```

技术约束：

- 使用 Python 3.12。
- 使用 uv 管理环境。
- 使用 FastAPI + Uvicorn 实现 HTTP API。
- 服务监听 `0.0.0.0:18080`。
- 不要把项目放在 `/mnt/c` 下，放在 WSL 原生文件系统 `~/services/asr-server`。
- 第一版优先做稳定的离线转写，不要一开始就实现 Web UI。
- 模型包很重，安装前先检查磁盘空间、CUDA、`nvidia-smi`。
- RTX 5070 Ti 较新，必须显式安装并验证 CUDA 版 torch，防止误装 CPU 版 torch。

默认模型注册包含：

- `qwen3-asr-1.7b`
- `qwen3-asr-0.6b`

`moss-transcribe-diarize-0.9b` 只有在 WSL 侧 MOSS smoke 通过并设置 `ASR_ENABLE_MOSS=1` 后才进入 `/v1/models`。默认模型仍是 `qwen3-asr-1.7b`。

MiMo-V2.5-ASR 不进入当前交付范围；不要把 MiMo 写进 `/v1/models`，也不要把 MiMo 转写作为验收项。

## RTX 5070 Ti / 真实模型安装要求

真实推理依赖只能在 WSL Arch Linux 内安装。Mac mini 只做客户端验证，不安装 CUDA、torch GPU 包、Qwen/MOSS 模型包或模型缓存。

先验收 torch CUDA，再安装 Qwen/MOSS 运行时依赖。`nvidia-smi` 只能证明 WSL 能看到驱动和显卡，不能证明 Python 环境里的 torch 是 CUDA 版。不要直接裸跑 `pip install torch`，必须使用 PyTorch 官方 CUDA wheel 源，例如 CUDA 12.8 的 `cu128`：

```bash
cd ~/services/asr-server

uv python install 3.12
uv sync

sudo pacman -Syu --needed git ffmpeg libsndfile

nvidia-smi
uv pip install --torch-backend cu128 -r requirements/wsl-gpu-cu128.txt
uv run python - <<'PY'
import torch
import transformers

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.version.cuda is not None, "装到 CPU 版 torch 了"
assert torch.cuda.is_available(), "torch 看不到 CUDA"
assert hasattr(transformers, "AutoProcessor")
assert hasattr(transformers, "AutoModelForMultimodalLM")
assert hasattr(transformers, "AutoModelForCausalLM")
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

如果已经误装 CPU 版 torch，删除 `.venv` 后按上述顺序重建。不要在错误环境上继续叠装。

如果 RTX 5070 Ti 报 `no kernel image is available`、架构不支持或 CUDA capability 不匹配，不要退回 CPU 版 torch；应改用支持该显卡的更新官方 CUDA wheel 或 PyTorch nightly，并重新跑 CUDA 验收脚本。

`/v1/models` 中只能声明真实跑通过的模型、后端和能力。当前只声明 `transformers` 后端，`vllm` 延后。

## 服务端开发前的后端预验收

开始开发真实 Qwen adapter 前，必须先脱离服务端代码跑通一次最小 Qwen3-ASR 转录流程：

- `transformers` 后端。用户口头说的 `tf` 在本项目里统一理解为 `transformers`，不是 TensorFlow。

优先使用 0.6B 和短音频样本降低首次验收成本：

```bash
uv run python scripts/qwen_asr_backend_smoke.py \
  --backend transformers \
  --model Qwen/Qwen3-ASR-0.6B-hf \
  --audio test-fixtures/audio/test_short.wav \
  --language auto
```

`transformers` 命令必须返回非空文本。当前不要在 `/v1/models` 中声明 `vllm`。

如果任务涉及启用 MOSS，还必须先跑：

```bash
uv run python scripts/moss_backend_smoke.py \
  --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --audio test-fixtures/audio/test_short.wav \
  --language auto
```

MOSS smoke 必须返回非空文本和可解析 segments。未通过前不要设置 `ASR_ENABLE_MOSS=1`。

MOSS 的 `auto` 是模型专属执行策略：30 分钟级已验证窗口内走 native long-form；超过 1801 秒自动降级为 1800 秒 fixed chunks。不得把 explicit `split_strategy=none` 的 6 小时输入上限描述成已验收能力。响应必须暴露 `execution.speaker_scope` 和 `generation.segment_coverage_ratio`；token 达上限或长音频缺尾必须返回受控 422，不能返回表面 completed 的残缺正文。

必须实现的 API：

```text
GET /health
GET /v1/models
GET /v1/models/{model_id}/status
POST /v1/models/{model_id}/load
DELETE /v1/models/{model_id}
DELETE /v1/models
POST /v1/audio/transcriptions
POST /v1/audio/transcription-jobs
GET /v1/jobs/{job_id}
DELETE /v1/jobs/{job_id}
```

`POST /v1/audio/alignments`、WebSocket 流式转写、时间戳等高级能力只在真实实现并验收后再打开能力声明；未打开时返回 `capability_not_supported` 或不暴露入口。

异步 job 已进入当前服务行为，维护时必须保持以下语义：

- `POST /v1/audio/transcription-jobs` 快速返回 `202 Accepted`、`id`、`status_url`，不要在这个 HTTP 请求内执行完整转录。
- `GET /v1/jobs/{job_id}` 返回 job 状态、队列位置、阶段、chunk 级进度、错误或最终结果。
- `DELETE /v1/jobs/{job_id}` 支持保守取消；正在模型推理的 chunk 不要强杀，等当前 chunk 完成后停止后续 chunk。
- 只做内存 JobManager 和单 worker FIFO 队列；不要引入 Redis、Celery、数据库、多 worker 或多 GPU 调度。
- 服务端可接收多个 job，但同一时间只运行一个转录 job；后来的 job 显示 `queued` 和 `queue_position`。
- 进度只承诺阶段和 chunk 级真实进度；不要伪造单个模型 chunk 内部百分比。

模型状态枚举：

```text
unloaded
loading
loaded
unloading_scheduled
unloading
error
```

卸载语义必须正确：

- 每个模型维护活跃请求计数。
- 每个模型维护生命周期锁。
- 收到卸载请求后，如果活跃请求数为 0，立即卸载。
- 如果活跃请求数大于 0，设置 `unloading_scheduled` 和 `rejecting_new_requests=true`。
- `unloading_scheduled` 状态下拒绝新的同模型请求，返回 409 和 `model_unloading_scheduled`。
- 最后一个活跃请求结束后再卸载模型。
- 卸载后调用 CUDA cache 清理。

建议项目结构：

```text
asr_server/
  __init__.py
  main.py
  config.py
  schemas.py
  errors.py
  registry.py
  lifecycle.py
  jobs.py
  adapters/
    __init__.py
    base.py
    qwen.py
    moss.py
tests/
  test_health.py
  test_models.py
  test_lifecycle.py
  test_transcription_api.py
pyproject.toml
README.md
```

开发顺序：

1. 阅读现有 Mac 侧实现和测试，保留 API 合约、错误信封、生命周期语义。
2. 在 WSL Arch Linux 的 `~/services/asr-server` 部署项目，不要放在 `/mnt/c`。
3. 检查磁盘空间、CUDA、`nvidia-smi`、Python 3.12、uv。
4. 先用 `scripts/qwen_asr_backend_smoke.py` 跑通 HF native `transformers` 最小转录流程。
5. 维护 Qwen3-ASR 适配器，真实跑通 `qwen3-asr-0.6b` 与 `qwen3-asr-1.7b`。
6. 如任务涉及 MOSS，先跑 `scripts/moss_backend_smoke.py`，再用 `ASR_ENABLE_MOSS=1` 验证 `moss-transcribe-diarize-0.9b`。
7. 对 `/v1/models` 中声明的每个模型和后端都做端到端转写验收；若某个模型、后端或能力不能跑通，不要声明它。
8. 保持 mock 适配器测试可在无 GPU 环境通过。
9. 维护 systemd user service 或 Windows 启动任务，让服务可后台常驻。
10. 从 Mac mini 验收局域网调用。
11. 保持异步转录 job、FIFO 串行队列、轮询状态和 chunk 级真实进度稳定。不要做 vLLM、WebSocket streaming、MiMo、ForcedAligner、数据库队列或多 worker 并发推理。

测试命令：

```bash
uv run pytest -q
uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
curl --noproxy '*' http://127.0.0.1:18080/health
curl --noproxy '*' http://YOUR_WINDOWS_LAN_IP:18080/v1/models
```

Mac mini 验收命令：

```bash
curl --noproxy '*' -v http://YOUR_WINDOWS_LAN_IP:18080/health
curl --noproxy '*' -v http://YOUR_WINDOWS_LAN_IP:18080/v1/models
```

交付物：

- 可运行的 FastAPI ASR 服务。
- README 中写明启动、停止、开机自启、Mac 调用方式。
- 测试覆盖健康检查、模型列表、加载、卸载、卸载等待当前请求完成、转写接口参数校验。
- 测试覆盖异步 job 创建、状态轮询、串行队列、chunk 进度、完成结果、失败错误、取消语义、临时文件清理和 3 分钟空闲自动卸载。
- 给出 `scripts/qwen_asr_backend_smoke.py` 在 `transformers` 后端的最小转录验收结果。
- 给出 `qwen3-asr-0.6b` 在所有声明后端上的真实音频转写验收结果。
- 给出 `qwen3-asr-1.7b` 在所有声明后端上的真实音频转写验收结果。
- 如果启用 MOSS，给出 `scripts/moss_backend_smoke.py` 和 `moss-transcribe-diarize-0.9b` 服务端 `verbose_json` segments 验收结果。
- 给出 Mac 到 WSL 局域网 job 创建和轮询到 completed 的验收结果。

不要做：

- 不要开放公网。
- 不要默认经过代理访问局域网 IP。
- 不要在活跃请求还没结束时强行卸载模型。
- 不要在 `/v1/models` 中声明未真实跑通的模型、后端或高级能力。

## 当前 job / 能力与鉴权合约

创建响应的任务标识字段是 `id`，路径中的 `{job_id}` 使用该值。轮询使用 `detail=compact`；completed 后请求一次 `detail=full` 读取 `result.text`。如需已完成 chunk 的文字，也使用 full。任务在内存中保存，重启后丢失，不能承诺服务端断点续算。

使用 `purpose=podcast` 时，REST 必须显式选择已声明的 MOSS 模型；它不会自动从默认 Qwen 切换模型。读取 `diarization.status` 和 `speaker_scope`，保留 partial / mixed / unresolved；4–5 小时整期一致性仍未完成完整实测。

启用 `ASR_API_TOKEN` 时，所有 `/v1/**` 请求提供 Bearer 凭据；客户端 CLI 用 `ASR_TOKEN`，MCP 用 `ASR_MCP_TOKEN`。`.env` 只有部署生成的 systemd unit 会读取，客户端和直接 Python 启动不自动加载。
