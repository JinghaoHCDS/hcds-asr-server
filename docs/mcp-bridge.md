# Agent 侧 MCP Bridge

完整 Agent 调用流程见服务的 [`/skill`](/skill)（[仓库来源](../skills/hcds-lan-asr/SKILL.md)）。
它同时覆盖直接 HTTP 和批量客户端；本文补充 stdio MCP 的安装与配置。

MCP bridge 运行在 Agent 所在客户端（已实机验证 Mac；其他平台尚待验收），通过 stdio 向 MCP Host 提供工具，再使用现有 HTTP API 把音频上传到 WSL ASR Gateway。它不运行模型、不安装 CUDA，也不新增局域网端口。

```text
MCP Host -> stdio -> Local ASR MCP Bridge -> HTTP :18080 -> WSL GPU ASR
```

## 安装

在客户端仓库目录执行：

```bash
uv sync --extra mcp
```

MCP SDK 固定在稳定的 v1 版本线 `mcp>=1.27,<2`。不要在未迁移代码和测试前直接升级到 v2。

## 配置

Bridge 读取以下环境变量：

| 变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `ASR_MCP_BASE_URL` | 否 | `http://127.0.0.1:18080` | WSL ASR Gateway 地址 |
| `ASR_MCP_ALLOWED_ROOTS` | 是 | 无 | 允许读取音频的本地目录；POSIX 用 `:`、原生 Windows 用 `;` 分隔多个目录 |
| `ASR_MCP_TOKEN` | 否 | 无 | 发送给 ASR Gateway 的 Bearer Token |
| `ASR_MCP_MAX_FILE_MB` | 否 | `512` | bridge 侧上传大小上限 |
| `ASR_MCP_REQUEST_TIMEOUT_SECONDS` | 否 | `3600` | HTTP read timeout |

`ASR_MCP_ALLOWED_ROOTS` 是安全要求。Bridge 会先解析真实路径，再检查文件是否位于允许目录，因此指向目录外的符号链接也会被拒绝。不要把整个 home 目录或 `/` 加入允许列表。

可从 `.env.example` 复制配置，但 MCP Host 必须通过自身的 `env` 配置传入这些变量；bridge 不会自动加载 `.env`。

## MCP Host 配置

下面是通用 stdio MCP 配置形态，按 MCP Host 的配置格式放入对应文件：

```json
{
  "mcpServers": {
    "hcds-lan-asr": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/Users/your-name/path/to/hcds-asr-server",
        "--extra",
        "mcp",
        "python",
        "-m",
        "asr_mcp.server"
      ],
      "env": {
        "ASR_MCP_BASE_URL": "http://YOUR_WINDOWS_LAN_IP:18080",
        "ASR_MCP_ALLOWED_ROOTS": "/Users/your-name/Downloads:/Users/your-name/Desktop/audio"
      }
    }
  }
}
```

如果 ASR Gateway 启用了 Bearer Token，在 `env` 中额外设置 `ASR_MCP_TOKEN`。不要把真实 token 提交到 Git。

## 工具

- `asr_list_models`：发现运行时模型、后端、能力和加载状态。
- `asr_get_model_status`：查询一个模型的生命周期状态和活跃请求数。
- `asr_transcribe_file`：上传并转录本地文件；超过同步阈值时可能直接返回异步 job。
- `asr_create_job`：显式创建 FIFO 异步转录 job。
- `asr_get_job`：默认 `detail=compact` 获取真实阶段和 chunk 计数；任务结束后用 `detail=full` 取一次最终结果。
- `asr_cancel_job`：安全请求取消，不强杀正在执行的模型推理。
- `asr_load_model`：加载已声明模型。
- `asr_unload_model`：等待活跃请求完成后安全卸载模型。

调用方必须先使用 `asr_list_models` 做能力发现，不得硬编码 MOSS、timestamps、speaker resolution 或后端能力。最终转录正文始终以同步结果的 `text` 或 job 的 `result.text` 为准。

`asr_transcribe_file` 与 `asr_create_job` 支持 `purpose=oral_analysis|podcast`、逐素材 `context`、`hotwords` 和 `idempotency_key`。用途相关字段默认保持未设置，让 REST 服务应用一致的 profile；若 MCP 调用显式传入 `timestamps`、`response_format` 或 `speaker_resolution`，则以显式值为准并接受能力校验，不做静默降级。不要让一次 MCP tool call 等待数小时；长素材先创建 job，由 Mac 侧编排低频查询 compact 状态。

## 本地验收

先确认 HTTP 服务可达：

```bash
curl --noproxy '*' http://YOUR_WINDOWS_LAN_IP:18080/health
```

然后启动 MCP Inspector，并把 stdio command 配置为：

```bash
uv run --directory /Users/your-name/path/to/hcds-asr-server --extra mcp python -m asr_mcp.server
```

验收至少包括：

1. `tools/list` 返回全部八个 ASR 工具。
2. `asr_list_models` 与 HTTP `/v1/models` 一致。
3. 允许目录内短音频能返回非空文字。
4. 允许目录外文件返回 ToolError，且不会发出 HTTP 请求。
5. 长音频能创建 job，并可查询 chunk 进度与最终结果。

## 当前边界

- MCP bridge 运行在客户端 stdio，不在 WSL FastAPI 中挂载远程 `/mcp`。
- MCP 不替代现有 REST API；REST 和 `/v1/models` 仍是公共合约与能力真相来源。
- 不使用 base64 传大音频，也不接受任意远程音频 URL。
- 不新增 `8001`、`8765` 或其他局域网端口。
- 将来实现远程 Streamable HTTP MCP 前，必须先完成认证、`Origin` 校验和受控上传票据设计。
