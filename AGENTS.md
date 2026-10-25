# 代理开发指南

## 官方 Agent Skill

- `skills/lan-asr/` 是官方客户端 Skill 的维护来源，沿用现有名称 `lan-asr`。
- `GET /skill` 返回其中的 `SKILL.md`；`/skill/lan-asr.zip` 和 `/skill/asr_batch.py` 分发同版本客户端。`/docs`、OpenAPI、README 和客户端提示词均链接该入口。
- 变更调用协议时同步官方 Skill 及配套客户端；文档不能替代 `/v1/models` 的运行时能力发现。发行包必须包含 `skills/lan-asr/`。
- Skill 不绑定个人 LAN IP；本地安装通过 `ASR_BASE_URL` 或 `--base-url` 配置地址。升级客户端保留既有批次数据库和请求快照。

## 2026-10-25 平台扩展契约

[平台运行契约](docs/asr-platform-runtime.md)补充本文件及PRD：同步、异步、工作台、编辑对齐和load共用有界FIFO；解析前正文准入与受控spool；有界PCM解码及可信资产复用；SQLite schema 3摘要/正文/chunk分离；JSONL关联诊断和内部引擎注册。旧身份隔离、保留期、默认quality_profile及模型质量门槛保持不变。预取默认关闭，性能与GPU验收以实际记录为准。

## 2026-10 工作台与 Qwen 质量流程契约

用户已批准本轮产品化扩展，以本节和 `docs/product-workbench.md` 为准，覆盖下文历史“无 Web UI / 仅内存队列 / 推理后删除全部上传”的限制。

- FastAPI 在唯一 LAN 端口 `18080` 同时提供 React 工作台和 API；不新增对外前端端口、云托管、vLLM 或 streaming。
- `/v1/**` 继续采用既有可选 Bearer 凭据、临时文件和内存 job 语义；新增 `quality_profile` 默认 `legacy`。MOSS 本轮不改。
- `/api/**` 使用独立用户名密码、HttpOnly 会话及服务端任务归属；旧 Bearer Token 不赋予个人任务访问权限。
- 工作台采用 SQLite、本地保留音频及检查点，与旧 job 共用单 worker FIFO；任务终态后保留 7 天，手动删除和到期清理覆盖音频、结果、编辑版本。推理临时文件仍及时清理。
- `quality_v1` 完整覆盖原音、单块最多 180 秒，默认 4096 生成上限（尊重显式值）；候选切分参数必须标记未质量验收，不能宣称已证明最优或 6 小时质量达标。
- 公共词库及本次热词按任务冻结快照；只有人工确认无歧义别名可自动规范，保留原稿与修改记录。不确定内容不得补写，时间戳不得伪造。
- 真实 GPU 与准确率验收在 WSL 进行；开发集和验收集分离，无真实长音频及人工参考文本时明确列出未验收项。
- `CLAUDE.md` 保持为指向本文件的兼容入口，不复制规则正文。

## 项目背景

这个仓库提供局域网 ASR 服务：Windows 主机的 NVIDIA GPU 在 WSL 中运行 Qwen3-ASR 与可选 MOSS，供局域网 Agent、应用和脚本通过 HTTP 或客户端 stdio MCP bridge 使用。

产品部署默认使用当前 WSL checkout，推荐克隆到 `~/services/asr-server`，不要放在 `/mnt/c`。HTTP 客户端和 MCP 默认地址为 `http://127.0.0.1:18080`；远端 Agent 必须显式配置 Windows 主机的局域网地址。

历史已验收环境是 WSL2 Arch Linux + RTX 5070 Ti 16GB，原部署 `/home/fragt/services/asr-server`、原入口 `http://192.168.31.137:18080` 仅作为历史验证信息。不得把它们硬编码为新用户的运行前提。Ubuntu 依赖说明不等于已完成对应实机验收。

Mac mini 和其他客户端只承担轻量开发与客户端验收。真实 GPU 推理、CUDA 验证、模型包安装、长期后台部署必须在 Windows PC 的 WSL 环境内完成；当前验收主路径仍为 WSL Arch Linux。

## 必读文档

实现服务行为前，必须先读：

```text
docs/asr-server-prd.md
```

把任务交给专门代理时，使用这些提示词：

```text
prompts/server-agent.md
prompts/request-client-agent.md
```

## 开发原则

- 公共 API 必须和 `docs/asr-server-prd.md` 保持一致。
- 使用 Python 3.12、uv、FastAPI 和 Uvicorn 实现服务。
- 服务入口监听 `0.0.0.0:18080`。
- WSL 项目位置不要放在 `/mnt/c`；推荐部署到 `~/services/asr-server`，部署脚本默认当前 checkout。
- 除非 PRD 明确变更，不要把 worker 端口 `8001` 暴露到局域网。
- 不要把历史测试端口 `8765` 写入实现、部署、防火墙或启动配置。
- API、生命周期管理、测试和第一条转录路径完成前，不要做 Web UI。

## 跨平台边界

- macOS 侧可以创建项目骨架、schema、测试、mock 适配器、客户端验证脚本和文档。
- macOS 侧不得要求 CUDA、NVIDIA 驱动、模型下载或大型本地模型缓存。
- WSL Arch Linux 侧负责 CUDA 检查、`nvidia-smi`、磁盘空间检查、真实 Qwen/MOSS 依赖安装和模型推理验证。
- 大模型依赖必须在适配器内懒加载，保证基础 import 和测试在无 GPU 环境也能运行。
- 路径处理保持 POSIX 兼容，不要在服务代码里硬编码 macOS 专用路径。

## RTX 5070 Ti 与 Qwen3-ASR 环境规则

WSL 侧安装真实 Qwen3-ASR 前，必须先把 RTX 5070 Ti 的 PyTorch CUDA 环境验收通过。这个显卡较新，最常见错误是 agent 直接执行普通 `pip install torch` 或被依赖解析成 CPU 版 torch。

硬性规则：

- 只在 WSL Arch Linux 内安装真实推理依赖；Mac mini 不安装 CUDA、torch GPU 包、Qwen/MOSS 模型包或模型缓存。
- 使用 Python 3.12 和 uv。不要混用多个 Python、conda 环境和系统 pip。
- 安装 torch 时必须使用 PyTorch 官方 CUDA wheel 源，例如 CUDA 12.8 对应的 `cu128` 源；不要使用没有 CUDA wheel 源的裸 `pip install torch`。
- `nvidia-smi` 只能证明 WSL 能看到驱动和显卡，不能证明 Python 环境里的 torch 是 CUDA 版；必须运行下面的 Python 验收脚本。
- 如果新显卡报 `no kernel image is available`、架构不支持或 CUDA capability 不匹配，不要退回 CPU 版 torch；应改用支持 RTX 5070 Ti 的更新官方 CUDA wheel 或 PyTorch nightly，并重新验收。
- 真实 Qwen adapter 使用 Hugging Face native `transformers` 路径；不要把 `qwen-asr` 包作为默认加载路径。
- vLLM 后续再做。只有明确切到 vLLM 任务时，才安装相关 vLLM 依赖，并且安装后必须再次验收 torch，防止依赖把 torch 替换成 CPU 版。
- `/v1/models` 中只能声明真实跑通过的模型、后端和能力。当前只声明 `transformers` 后端，不要声明 `vllm`。

推荐 WSL 安装顺序：

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

如果已经误装 CPU 版 torch，优先删除 `.venv` 后按上述顺序重建环境；不要在错误环境上继续叠装：

```bash
rm -rf .venv
uv sync
uv pip install --torch-backend cu128 -r requirements/wsl-gpu-cu128.txt
```

## Qwen3-ASR 后端预验收门槛

开发 WSL 服务端真实 adapter 前，必须先脱离本项目服务代码，跑通一次最小 Qwen3-ASR 转录流程：

- `transformers` 后端，用户口头说的 `tf` 在本项目里统一理解为 `transformers`，不是 TensorFlow。

推荐先用 `Qwen/Qwen3-ASR-0.6B-hf` 和 `test-fixtures/audio/test_short.wav` 验收，降低首次下载和显存压力。`transformers` 返回非空文本后，才能维护或启用 `asr_server` Qwen adapter。当前不要在 `/v1/models` 中声明 `vllm`。

验收命令：

```bash
uv run python scripts/qwen_asr_backend_smoke.py \
  --backend transformers \
  --model Qwen/Qwen3-ASR-0.6B-hf \
  --audio test-fixtures/audio/test_short.wav \
  --language auto
```

验收记录至少写下：模型 ID、后端、音频文件、torch 版本、CUDA 版本、GPU 名称、输出语言和文本前 200 字。

## MOSS-Transcribe-Diarize 启用门槛

`moss-transcribe-diarize-0.9b` 是可选 gate 模型。只有在 WSL 侧完成独立 MOSS smoke 并确认 CUDA torch 未被替换后，才能设置 `ASR_ENABLE_MOSS=1` 让它进入 `/v1/models`。默认模型仍是 `qwen3-asr-1.7b`。

验收命令：

```bash
uv run python scripts/moss_backend_smoke.py \
  --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --audio test-fixtures/audio/test_short.wav \
  --language auto
```

MOSS 只声明 `transformers` 后端。它的段级 speaker/timestamp 结果只在 `response_format=verbose_json` 的 `segments` 中返回，不等于 word/char timestamps，也不等于 forced alignment。长音频切分后 speaker label 只保证 chunk 内相对一致，不能当作全局说话人身份。

MOSS Anchor Replay 在调用方未显式传 `max_new_tokens` 时使用至少 `24000` 的自动生成预算，避免高语速多人播客按 `12 tokens/秒` 计算出的预算过早截断；显式请求值仍应保持原样，上限仍为 `65536`。

## API 与生命周期规则

- 模型能力发现必须来自 `GET /v1/models`；客户端不要硬编码模型能力。
- 模型状态必须使用 PRD 枚举：`unloaded`、`loading`、`loaded`、`unloading_scheduled`、`unloading`、`error`。
- 每个模型必须维护活跃请求计数和生命周期锁。
- 如果卸载请求到来时仍有活跃请求，设置 `unloading_scheduled`，新的同模型请求返回 `409 model_unloading_scheduled`，等活跃请求完成后再卸载。
- 推理仍在执行时，不要强制卸载模型。
- 异步转录使用 `POST /v1/audio/transcription-jobs`、`GET /v1/jobs/{job_id}`、`DELETE /v1/jobs/{job_id}`。
- 异步 job 采用内存 JobManager 和单 worker FIFO 队列；允许排队多个 job，但同一时间只运行一个真实模型转录。
- job 进度只承诺服务端阶段和 chunk 级真实进度，不要伪造单个模型 chunk 内部百分比。
- chunked job 每完成一个真实 chunk 后，在 `progress.chunk_text` 暴露该块稳定文字，在 `progress.text` 暴露累计临时文字，并在 `progress.chunks` 暴露可恢复的有序列表；native long-form 单次调用不得伪造这些字段。最终正文始终以 `result.text` 为准。
- 错误必须使用 PRD 错误信封：

```json
{
  "error": {
    "code": "model_not_found",
    "message": "unknown model: xxx",
    "details": {}
  }
}
```

## 网络与安全

- 唯一面向局域网的 API 端口是 `18080`。
- 客户端请求 Windows 主机的 LAN 地址时必须绕过本机代理，例如使用 `curl --noproxy '*'`。
- `ASR_API_TOKEN` 启用可选 Bearer 鉴权，保护 `/v1/**`；`/health` 和文档保持公开。CLI 使用 `ASR_TOKEN`，MCP 使用 `ASR_MCP_TOKEN`，不得打印或提交真实 token。
- `.env` 由部署生成的 systemd unit 读取；直接 Python 启动和 MCP 不会自动加载它。
- 不要假设服务会暴露到公网。
- 不要添加公网隧道、端口映射或互联网暴露。
- 上传音频应写入临时位置，并在推理结束后清理。
- 日志默认不要保存完整音频内容。

## 测试要求

至少覆盖：

- 健康检查。
- 模型列表和单模型状态。
- 模型加载和卸载行为。
- 卸载等待活跃请求完成。
- `unloading_scheduled` 状态下拒绝新请求。
- 转录接口参数校验。
- 异步 job 创建、状态轮询、队列位置、chunk 进度、完成结果、失败错误和取消语义。
- 未声明能力的错误处理，例如 timestamps、forced alignment 或 streaming 返回 `capability_not_supported`。
- `/v1/models` 中声明的所有模型和后端的转录路径。

先用 mock 适配器覆盖生命周期和 API 行为，再在 WSL Arch Linux 内接入真实模型。

## Git 规范

- 生成缓存、虚拟环境、模型缓存、上传文件和本地运行数据不要进 Git。
- 文档和提示词保持在稳定路径，方便代理交接。
- 提交要聚焦，提交信息要清楚。
- 不要提交机器专用 secret、token、下载模型或包含隐私内容的音频样本。
