# WSL 侧代理项目总览提示词

> 2026-10 扩展：工作台 `/api/**` 使用 SQLite、个人 Cookie 和七天保留；旧 `/v1/**` 继续内存任务与原鉴权，新增 `quality_profile` 默认 `legacy`。历史“不做网页/数据库”仅适用旧阶段，以 [工作台契约](../docs/product-workbench.md) 和项目 AGENTS 为准。

部署路径和 LAN IP 由用户环境决定；示例 `YOUR_WINDOWS_LAN_IP` 与 `/path/to/` 必须替换。历史 Mac / Arch 验收不代表其他平台已实测。

你是在 Windows PC 的 WSL Arch Linux 内接手这个项目的开发代理。请先用这份总览理解项目，再阅读细节文档。

## 你要完成什么

这个项目是一个局域网 ASR 网关。局域网 Agent 客户端项目通过 HTTP 上传音频，WSL Arch Linux 内的服务使用 Windows PC 上的 RTX 5070 Ti GPU 跑 Qwen3-ASR；在 MOSS gate 开启后，也可以调用 MOSS-Transcribe-Diarize 返回段级说话人结果。

最终服务入口：

```text
http://YOUR_WINDOWS_LAN_IP:18080
```

WSL 内项目目录：

```text
~/services/hcds-asr-server
```

不要把项目部署在 `/mnt/c`。不要使用历史测试端口 `8765`。

## 当前仓库已经有什么

Mac 侧已经提前做好了这些不依赖 GPU 的工作：

- FastAPI 服务入口：`asr_server/main.py`
- 模型注册表：默认包含 `qwen3-asr-0.6b` 和 `qwen3-asr-1.7b`；`ASR_ENABLE_MOSS=1` 时额外包含 `moss-transcribe-diarize-0.9b`
- 生命周期管理：加载、卸载、活跃请求计数、`unloading_scheduled`
- 空闲卸载：转录结束后默认 180 秒无新请求自动卸载模型并释放 CUDA cache
- 异步转录 job：内存 JobManager、单 worker FIFO 队列、轮询状态和 chunk 级进度
- 临时文件清理：同步请求、异步 job、取消和失败路径均应清理上传与中间音频文件
- mock ASR 适配器：用于无 GPU 环境测试
- Qwen 真实适配器：`asr_server/adapters/qwen.py`
- MOSS 真实适配器：`asr_server/adapters/moss.py`
- HTTP smoke test：`tests/test_http_smoke.py`
- WSL 一键部署脚本：`deploy/wsl-deploy.sh`
- WSL 一键 smoke 脚本：`scripts/wsl_smoke.sh`
- Qwen 后端最小验收脚本：`scripts/qwen_asr_backend_smoke.py`
- MOSS 后端最小验收脚本：`scripts/moss_backend_smoke.py`
- WSL 部署文档：`docs/wsl-deployment.md`
- 验收记录模板：`docs/validation-template.md`
- 测试音频：`test-fixtures/audio/test_short.wav` 和 `/path/to/your-long-audio.mp3`

Mac 侧代码默认使用 `ASR_ADAPTER=mock`，不会加载真实模型。WSL 真实部署时使用：

```bash
ASR_ADAPTER=qwen uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

MOSS 只有在独立 smoke 通过后才设置：

```bash
ASR_ADAPTER=qwen ASR_ENABLE_MOSS=1 uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

## 你必须先读

按顺序阅读：

```text
AGENTS.md
docs/asr-server-prd.md
docs/wsl-deployment.md
prompts/server-agent.md
docs/validation-template.md
```

当前真实模型验收记录：

```text
docs/validation-2026-07-02-wsl-hf-native.md
docs/validation-2026-07-10-wsl-moss.md
docs/validation-2026-07-15-wsl-moss-long-form.md
```

## 当前模型范围

默认模型是 Qwen3-ASR 两个尺寸：

```text
qwen3-asr-0.6b
qwen3-asr-1.7b
```

MOSS 模型 `moss-transcribe-diarize-0.9b` 是可选 gate 模型，只在 `ASR_ENABLE_MOSS=1` 且 WSL smoke 通过后声明。不要把它设为默认模型。

MiMo-V2.5-ASR 不在当前范围内。不要把 MiMo 写进 `/v1/models`，也不要把 MiMo 转写当作验收项。

高级能力如时间戳、强制对齐、WebSocket 流式转写，只有真实实现并验收后才能声明。没验收就不要在 `/v1/models` 的 capabilities 里打开。

## 最容易出错的地方

RTX 5070 Ti 是较新的显卡。不要直接裸跑：

```bash
pip install torch
```

这很容易装成 CPU 版 torch。正确流程是通过项目的 WSL GPU requirements 安装并验收 CUDA 版 torch，例如 CUDA 12.8 的 `cu128`：

```bash
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

`nvidia-smi` 只能证明 WSL 能看到显卡，不代表 Python 里的 torch 是 CUDA 版。必须跑上面的 Python 验收。

## 服务端开发前置门槛

在写或修改真实 Qwen adapter 之前，先脱离服务端代码跑通最小转录流程：

```bash
uv run python scripts/qwen_asr_backend_smoke.py \
  --backend transformers \
  --model Qwen/Qwen3-ASR-0.6B-hf \
  --audio test-fixtures/audio/test_short.wav \
  --language auto
```

用户口头说的 `tf` 在本项目里表示 `transformers`，不是 TensorFlow。

`transformers` 后端返回非空文本后，才能维护或启用服务端真实 Qwen adapter。当前不要在 `/v1/models` 中声明 `vllm`。

如果启用 MOSS，还必须先跑：

```bash
uv run python scripts/moss_backend_smoke.py \
  --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --audio test-fixtures/audio/test_short.wav \
  --language auto
```

MOSS smoke 必须返回非空文本和可解析 segments；未通过前不要设置 `ASR_ENABLE_MOSS=1`。

MOSS `auto` 在已验证的 30 分钟级窗口内使用 native long-form，超过 1801 秒时明确降级为 1800 秒 fixed chunks。只有响应 `execution.speaker_scope=global` 时 speaker ID 才能视作整段一致；chunk 模式不得伪装成全局 diarization。生成达到 token 上限或 segment 未覆盖音频尾部时返回受控 422。

## 验收路径

建议执行顺序：

1. 在 `~/services/hcds-asr-server` 准备项目。
2. `uv sync`
3. `uv run pytest -q`
4. `uv run mypy asr_server tests scripts`
5. 验证 CUDA 版 torch。
6. 安装并验收 HF native Transformers 依赖。
7. 跑通 `scripts/qwen_asr_backend_smoke.py --backend transformers`。
8. 如启用 MOSS，跑通 `scripts/moss_backend_smoke.py`。
9. 启动真实服务：`ASR_ADAPTER=qwen uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080`
10. 如启用 MOSS，启动命令增加 `ASR_ENABLE_MOSS=1`。
11. 运行：`ASR_BASE_URL=http://127.0.0.1:18080 uv run pytest tests/test_http_smoke.py -q`
12. 从 Mac mini 用 `curl --noproxy '*'` 验收局域网入口。
13. 把结果填写到 `docs/validation-template.md` 对应格式里。

## 异步转录 Job 维护边界

异步任务和进度查询已经是当前服务行为。维护时继续只围绕已声明的 `transformers` 后端保持稳定；不要做 vLLM、WebSocket streaming、MiMo、Web UI、公网访问、数据库队列或多机调度，除非 PRD 明确进入对应阶段。

### 从第一性原理理解这个任务

当前 `POST /v1/audio/transcriptions` 是同步 HTTP 请求。Mac 前端发出请求后，只能看到“请求还没结束”；服务端完成所有解码、切分、模型加载、chunk 转录、合并后才一次性返回结果。

这带来两个问题：

- 用户不知道服务还在正常工作，还是卡在解码、模型加载、某个 chunk 推理或合并阶段。
- 长音频请求会长时间占住一个 HTTP 连接；前端刷新或网络中断后，服务端和用户都缺少可恢复的任务状态。

真实进度的边界也要讲清楚：

- 我们可以做“服务端真实阶段进度”：`queued`、`preprocessing`、`splitting`、`loading_model`、`transcribing`、`merging`、`completed`、`failed`。
- 切分完成后，可以做“真实 chunk 级进度”：总 chunk 数、已完成 chunk 数、当前 chunk index、百分比。
- 当前真实 adapter 拿不到单个 chunk 内部的 token/帧级进度，所以不要伪造“模型内部 37%”这种精度。单个 chunk 正在推理时，只能说当前 chunk 正在处理。

### 并发与算力边界

用户当前没有同时转录多个音频的需求。为了保护 RTX 5070 Ti 显存和模型状态，当前服务采用最简单、可解释、可验收的串行模型：

- Mac mini 仍然只作为轻量客户端，不安装 CUDA、torch GPU 包、Qwen/MOSS 模型包或模型缓存。
- 真实模型推理只在 WSL Arch Linux 内执行。
- 服务端可以接收多个 job，但同一时间只运行一个转录 job；后来的 job 进入 FIFO 队列。
- 不追求并发吞吐，不要为了并发引入 Redis、Celery、数据库、进程池或多 worker 推理。
- 如果多个 HTTP 请求同时到达，服务端应按队列顺序依次转录，并在 `GET /v1/jobs/{job_id}` 中暴露 `queue_position`。
- 不要在活跃推理时强制卸载模型；卸载请求仍按现有生命周期规则排队或拒绝新同模型请求。

### API 目标

当前已经实现三个异步 job 接口，沿用 PRD 路径：

```text
POST /v1/audio/transcription-jobs
GET /v1/jobs/{job_id}
DELETE /v1/jobs/{job_id}
```

保留现有同步接口：

```text
POST /v1/audio/transcriptions
```

同步接口用于短音频和简单脚本；异步 job 用于中长音频、前端需要进度条、或不希望长 HTTP 连接阻塞的场景。

### POST /v1/audio/transcription-jobs

请求格式使用 `multipart/form-data`，字段尽量复用同步转录接口：

- `file`：音频文件，必填；先只支持单文件，不做 `files[]` 数组。
- `model`：默认 `qwen3-asr-1.7b`。
- `language`：默认 `auto`。
- `response_format`：默认 `json`；job 最终结果先只保证 JSON。
- `timestamps`：默认 `none`；如果请求 `word` 或 `char`，按现有能力返回 `capability_not_supported`。
- `backend`：默认 `auto`；当前只能解析到 `transformers`。
- `max_new_tokens`、`context`、`hotwords`、`split_strategy`、`max_chunk_seconds`、`overlap_seconds`、`preserve_segments`：语义与同步接口一致。

接口必须快速返回，不在这个 HTTP 请求里执行真实转录。

建议返回 `202 Accepted`：

```json
{
  "id": "job_01JZ0000000000000000000000",
  "status": "queued",
  "model": "qwen3-asr-1.7b",
  "backend": "transformers",
  "queue_position": 1,
  "created_at": "2026-06-29T12:00:00Z",
  "status_url": "/v1/jobs/job_01JZ0000000000000000000000"
}
```

实现要求：

- 上传音频必须写入服务端临时 job 工作目录，不要把完整音频保存在 Git 目录。
- job 完成或失败后要清理上传原始音频和中间 chunk 临时文件。
- 普通日志不要记录完整 context、hotwords 或音频内容；最多记录长度、hash、文件大小、时长和 job id。
- job id 使用不可预测 ID，例如 `uuid4` 或 ULID 风格字符串，不要用递增整数暴露请求量。
- 内存中保存 job 状态和最终结果即可；进程重启后 job 丢失可以接受，但要在文档和验收记录中说明。
- 给 job 结果设置内存保留时间，建议默认 1 小时，可通过环境变量配置，例如 `ASR_JOB_RESULT_TTL_SECONDS=3600`。

### GET /v1/jobs/{job_id}

返回 job 当前状态、阶段进度、错误或最终结果。

状态枚举建议：

```text
queued
preprocessing
splitting
loading_model
transcribing
merging
completed
failed
cancel_requested
cancelled
expired
```

队列中示例：

```json
{
  "id": "job_01JZ0000000000000000000000",
  "status": "queued",
  "model": "qwen3-asr-1.7b",
  "backend": "transformers",
  "queue_position": 2,
  "progress": {
    "phase": "queued",
    "percent": 0.0,
    "message": "waiting for previous transcription jobs"
  },
  "created_at": "2026-06-29T12:00:00Z",
  "started_at": null,
  "completed_at": null,
  "elapsed_seconds": 0.0
}
```

转录中示例：

```json
{
  "id": "job_01JZ0000000000000000000000",
  "status": "transcribing",
  "model": "qwen3-asr-1.7b",
  "backend": "transformers",
  "queue_position": 0,
  "progress": {
    "phase": "transcribing",
    "percent": 37.5,
    "total_chunks": 32,
    "completed_chunks": 12,
    "current_chunk": 13,
    "current_chunk_start": 1412.4,
    "current_chunk_end": 1530.1,
    "message": "transcribing chunk 13 of 32"
  },
  "split": {
    "strategy": "silero",
    "requested_strategy": "auto",
    "vad_backend": "silero",
    "chunk_count": 32,
    "soft_chunk_seconds": 120,
    "hard_chunk_seconds": 300,
    "overlap_seconds": 2
  },
  "created_at": "2026-06-29T12:00:00Z",
  "started_at": "2026-06-29T12:00:03Z",
  "completed_at": null,
  "elapsed_seconds": 74.3
}
```

完成示例：

```json
{
  "id": "job_01JZ0000000000000000000000",
  "status": "completed",
  "progress": {
    "phase": "completed",
    "percent": 100.0,
    "total_chunks": 32,
    "completed_chunks": 32
  },
  "result": {
    "id": "tr_01JZ0000000000000000000000",
    "model": "qwen3-asr-1.7b",
    "backend": "transformers",
    "language": "zh",
    "text": "完整合并后的转写文本。",
    "duration": 3630.05,
    "split": {},
    "chunks": [],
    "usage": {
      "audio_seconds": 3630.05
    },
    "timings": {},
    "warnings": []
  },
  "error": null,
  "created_at": "2026-06-29T12:00:00Z",
  "started_at": "2026-06-29T12:00:03Z",
  "completed_at": "2026-06-29T12:04:12Z",
  "elapsed_seconds": 249.0
}
```

失败示例必须使用统一 error envelope 的内部形状：

```json
{
  "id": "job_01JZ0000000000000000000000",
  "status": "failed",
  "progress": {
    "phase": "failed",
    "percent": 37.5,
    "total_chunks": 32,
    "completed_chunks": 12
  },
  "result": null,
  "error": {
    "code": "gpu_unavailable",
    "message": "CUDA out of memory during Qwen ASR",
    "details": {
      "phase": "transcribing",
      "chunk_index": 12
    }
  }
}
```

未知 job 返回：

```json
{
  "error": {
    "code": "job_not_found",
    "message": "unknown job: job_xxx",
    "details": {}
  }
}
```

### DELETE /v1/jobs/{job_id}

取消语义要保守：

- 如果 job 还在 `queued`，可以直接变成 `cancelled`，不进入模型推理。
- 如果 job 正在 `preprocessing`、`splitting` 或 `merging`，尽量在阶段边界取消。
- 如果 job 正在模型推理某个 chunk，不要强杀推理线程或卸载模型；设置 `cancel_requested`，等当前 chunk 结束后停止后续 chunk，并把状态改为 `cancelled`。
- 如果 job 已经 `completed`、`failed`、`cancelled` 或 `expired`，DELETE 返回当前状态，不要报 500。

建议返回：

```json
{
  "id": "job_01JZ0000000000000000000000",
  "status": "cancel_requested",
  "message": "cancellation will take effect after the current chunk finishes"
}
```

### 同步接口与异步接口的关系

不要删除现有同步接口。当前推荐行为：

- 小于等于 10 分钟的音频：同步接口继续允许 `200 OK` 返回完整结果。
- 超过 10 分钟的音频：同步接口不要长时间阻塞；建议创建 job 并返回 `202 Accepted`，响应里给出 `id` 和 `status_url`。
- 如果暂时不想让同步接口自动创建 job，也可以返回 `422 duration_limit_exceeded` 并在 details 中提示使用 `/v1/audio/transcription-jobs`。但优先实现自动创建 job，更符合 PRD。
- job 接口不受 10 分钟同步阈值限制，但仍受单文件 6 小时和服务端上传大小限制。

### 内部实现建议

新增一个轻量 `JobManager`，不要引入外部基础设施：

- 位置建议：`asr_server/jobs.py` 或 `asr_server/jobs/manager.py`。
- `app.state.job_manager` 持有内存 job 表、FIFO 队列、单 worker task。
- FastAPI startup 时启动 worker；shutdown 时尽量停止 worker 并清理临时文件。
- 用 `asyncio.Queue` 或受锁保护的 `deque` 实现队列。
- job 状态更新必须集中在 JobManager，不要散落在多个 endpoint 里。
- job worker 可以复用现有同步转录的内部函数，但要能在关键阶段更新 progress。
- 如果当前同步转录逻辑太难复用，先抽出一个内部 service 函数，例如 `run_transcription_request(...)`，让同步接口和 job worker 共用同一条解码、切分、转录、合并路径。
- 不要复制两套真实 adapter 调用逻辑，避免同步接口和 job 接口行为分叉。

关键进度更新点：

1. `queued`：job 创建后进入队列。
2. `preprocessing`：开始读取上传文件、检查大小、FFmpeg 解码/规范化。
3. `splitting`：开始获取音频元数据、VAD/fixed 切分。
4. `loading_model`：切分完成后，进入模型加载或确认模型已加载。
5. `transcribing`：每个 chunk 开始前更新 `current_chunk`；每个 chunk 完成后更新 `completed_chunks` 和 `percent`。
6. `merging`：所有 chunk 完成后开始合并文本、去重、构造响应。
7. `completed`：保存最终 result。
8. `failed`：保存统一 error，清理临时文件。
9. `cancel_requested` / `cancelled`：在安全阶段边界停止。

百分比规则：

- 切分前不知道总 chunk 数，`percent` 可以是 `0.0` 到 `5.0` 的阶段性估计，但不要假装精确。
- 切分后，以 chunk 为主计算真实进度。建议公式：`percent = completed_chunks / total_chunks * 100`，或给 preprocessing/splitting/merging 留少量权重，但必须在文档里说明。
- 用户最关心“是不是在动”和“转到第几段”，优先保证 `completed_chunks`、`total_chunks`、`current_chunk` 准确。

### 数据结构要求

job 记录至少包含：

- `id`
- `status`
- `model`
- `backend`
- `language`
- `created_at`
- `started_at`
- `completed_at`
- `expires_at`
- `queue_position`
- `progress`
- `request` 的安全摘要：文件名、大小、参数、context hash，不保存完整 context 到普通日志。
- `split` 摘要。
- `result`：完成后保存与同步接口兼容的 JSON 结果。
- `error`：失败后保存统一错误对象。
- `warnings`
- `temp_paths`：用于清理。

### 测试要求

Mac/mock 环境必须能跑完整测试，不需要 CUDA、torch GPU 包、Qwen 模型缓存或真实模型下载。

至少增加这些测试：

- 创建 job 返回 `202`、`status=queued`、`status_url`。
- `GET /v1/jobs/{job_id}` 能看到 queued/running/completed 状态。
- mock adapter 延迟时，进度从 queued/running 变化到 completed。
- 多个 job 同时提交时，只运行一个，另一个显示 `queued` 和正确 `queue_position`。
- chunk 转录时进度按 `completed_chunks` 增长。
- job 完成后 `result.text` 非空，响应结构与同步接口兼容。
- adapter 抛错时 job 进入 `failed`，错误对象使用统一 code/message/details。
- queued job 可以取消为 `cancelled`。
- running job 取消时不强杀当前 chunk，当前 chunk 完成后进入 `cancelled`。
- 未知 job 返回 `404 job_not_found`。
- 超过 10 分钟的同步请求返回 `202` job，或明确返回 `422` 提示使用 job；优先实现 `202`。
- 不安装 GPU 依赖的 Mac 环境仍然 `uv run pytest -q` 通过。
- `uv run mypy asr_server tests scripts` 通过；WSL-only torch import 必须懒加载或做类型兼容，不能让 Mac 环境静态检查失败。

### WSL 真实验收要求

完成代码后在 WSL Arch Linux 内验收：

1. `uv run pytest -q`
2. `uv run mypy asr_server tests scripts`
3. CUDA torch 验收：确认 torch 不是 CPU 版。
4. 启动真实服务：

```bash
ASR_ADAPTER=qwen uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

5. 用 `test-fixtures/audio/test_short.wav` 创建 job，轮询到 `completed`，记录 `result.text` 前 200 字。
6. 用 `/path/to/your-long-audio.mp3` 创建 job，轮询过程中至少记录三次状态：queued 或 preprocessing、transcribing 中间状态、completed。
7. 验证 `progress.total_chunks`、`completed_chunks`、`current_chunk` 在长音频转录中真实变化。
8. 分别对 `qwen3-asr-0.6b` 和 `qwen3-asr-1.7b` 跑一次 job API，后端均为 `transformers`，返回非空文本。
9. 从 Mac mini 通过局域网入口请求：

```bash
curl --noproxy '*' http://YOUR_WINDOWS_LAN_IP:18080/health
curl --noproxy '*' http://YOUR_WINDOWS_LAN_IP:18080/v1/models
```

10. 从 Mac mini 创建一个 job 并轮询 `GET /v1/jobs/{job_id}` 到 completed，证明进度不是只在 WSL localhost 可见。

验收记录至少写下：

- 模型 ID。
- backend。
- torch 版本、CUDA 版本、GPU 名称。
- 音频文件、音频时长、chunk 数。
- job 状态流转时间线。
- 轮询样例 JSON。
- 总耗时、加载耗时、推理耗时、合并耗时。
- 文本前 200 字。
- 如果失败，写清楚错误 code、阶段、chunk index、下一步判断。

### 当前不进入范围

- vLLM 后端。
- WebSocket streaming。
- ForcedAligner / word-char timestamps。
- MOSS 跨 chunk speaker identity stitching。
- MiMo。
- Web UI。
- Redis、Celery、数据库持久化队列。
- 多 GPU、多 worker 并发推理。

## 你不要做什么

- 不要开放公网。
- 不要创建公网隧道。
- 不要把 worker 端口暴露给 Mac。
- 不要在活跃请求运行时强行卸载模型。
- 不要把未跑通的模型、后端或高级能力写进 `/v1/models`。
- 不要把下载的模型、缓存、私密音频或 token 提交进 Git。

## 交付标准

你交付时至少说明：

- torch 版本、CUDA 版本、GPU 名称。
- `transformers` 后端最小转录结果。
- `qwen3-asr-0.6b` 和 `qwen3-asr-1.7b` 的服务端 API 转录结果。
- 如果启用 MOSS，`moss-transcribe-diarize-0.9b` 的 `verbose_json` segments 转录结果。
- `/health` 和 `/v1/models` 的响应摘要。
- Mac mini 是否能通过 `http://YOUR_WINDOWS_LAN_IP:18080` 调用服务。
- 如果失败，给出具体命令、错误日志和下一步判断。

## 当前 job / 能力与鉴权合约

创建响应的任务标识字段是 `id`，路径中的 `{job_id}` 使用该值。轮询使用 `detail=compact`；completed 后请求一次 `detail=full` 读取 `result.text`。如需已完成 chunk 的文字，也使用 full。任务在内存中保存，重启后丢失，不能承诺服务端断点续算。

使用 `purpose=podcast` 时，REST 必须显式选择已声明的 MOSS 模型；它不会自动从默认 Qwen 切换模型。读取 `diarization.status` 和 `speaker_scope`，保留 partial / mixed / unresolved；4–5 小时整期一致性仍未完成完整实测。

启用 `ASR_API_TOKEN` 时，所有 `/v1/**` 请求提供 Bearer 凭据；客户端 CLI 用 `ASR_TOKEN`，MCP 用 `ASR_MCP_TOKEN`。`.env` 只有部署生成的 systemd unit 会读取，客户端和直接 Python 启动不自动加载。
