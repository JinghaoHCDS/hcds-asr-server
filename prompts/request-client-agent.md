# 请求端 Agent 提示词

先请求目标服务的 `GET /skill` 阅读与其版本一致的官方 Agent 用法，再读取
`GET /openapi.json` 和 `GET /v1/models`。`/skill/lan-asr.zip` 提供指南及配套批量客户端。
旧服务返回 `/skill` 404 时，可按现有 OpenAPI 接入并说明版本差异；网络或鉴权失败不视为旧版本。

> 2026-10-25 平台扩展遵循 [平台运行契约](../docs/asr-platform-runtime.md)：所有 HTTP 推理共享 FIFO，短同步也会排队；工作台可显式选择支持的模型/profile，并提供摘要分页、revision/ETag；日志可按 job/trace 关联。历史内存-only限制只适用于旧 `/v1` owner。

> 2026-10 扩展：工作台 `/api/**` 使用 SQLite、个人 Cookie 和七天保留；旧 `/v1/**` 继续内存任务与原鉴权，新增 `quality_profile` 默认 `legacy`。历史“不做网页/数据库”仅适用旧阶段，以 [工作台契约](../docs/product-workbench.md) 和项目 AGENTS 为准。

部署路径和 LAN IP 由用户环境决定；示例 `YOUR_WINDOWS_LAN_IP` 与 `/path/to/` 必须替换。历史 Mac / Arch 验收不代表其他平台已实测。

你是 局域网 Agent 客户端项目的开发代理。你的任务是把项目接入一台局域网内的 WSL ASR 服务器。

ASR 服务地址：

```text
http://YOUR_WINDOWS_LAN_IP:18080
```

重要网络约束：

- 客户端可能配置了 `http_proxy`、`https_proxy`、`all_proxy` 等环境代理。
- 请求局域网 ASR 服务时必须绕过代理。
- 命令行测试使用 `curl --noproxy '*'`。
- 代码里设置 `NO_PROXY=YOUR_WINDOWS_LAN_IP,localhost,127.0.0.1`，或使用等价的单请求绕过代理配置。

请完成以下任务：

1. 先做连通性检查：

```bash
curl --noproxy '*' -v http://YOUR_WINDOWS_LAN_IP:18080/health
curl --noproxy '*' -v http://YOUR_WINDOWS_LAN_IP:18080/v1/models
```

2. 读取 `GET /v1/models` 返回值，不要硬编码模型能力。根据服务端声明选择模型。

3. 默认优先使用服务端声明的默认模型。当前通常是：

```text
qwen3-asr-1.7b
```

如果请求失败且错误码不是音频或网络问题，再尝试服务端声明可用的另一个普通转录模型，例如：

```text
qwen3-asr-0.6b
```

如果 `/v1/models` 声明了 `moss-transcribe-diarize-0.9b`，只有在产品明确需要段级说话人结果时才选择它，并优先请求 `response_format=verbose_json`。不要在服务端未声明 MOSS 时硬编码该模型。

MOSS 客户端必须读取响应 `execution.speaker_scope`：只有 `global` 才能把 `S01/S02` 当作整段一致身份；`chunk` 时必须保留 `chunk-NNNN:S01` 命名空间，不能自行去掉 chunk 前缀或推断跨块身份。展示降级状态时读取 `execution.automatic_chunk_fallback` 和 `warnings`，完整性诊断读取 `generation.segment_coverage_ratio`。

4. 实现一个最小同步请求函数，使用 `multipart/form-data` 调用：

```text
POST /v1/audio/transcriptions
```

字段：

- `file`：音频文件。
- `model`：默认 `qwen3-asr-1.7b`。
- `language`：默认 `auto`。
- `response_format`：默认 `json`。
- `timestamps`：默认 `none`，只有在服务端声明模型支持时才传 `word` 或 `char`。

5. 如果音频较长、前端需要进度条、或不希望 HTTP 长连接阻塞，实现异步 job 调用：

```text
POST /v1/audio/transcription-jobs
GET /v1/jobs/{job_id}
DELETE /v1/jobs/{job_id}
```

job 规则：

- 创建 job 后应收到 `202`、`id` 和 `status_url`。
- 用 `GET /v1/jobs/{job_id}` 轮询状态，不要用前端假进度代替服务端状态。
- 队列状态看 `status=queued` 和 `queue_position`。
- 转录进度看 `progress.phase`、`total_chunks`、`completed_chunks`、`current_chunk` 和 `percent`。
- 单个模型 chunk 内部没有真实百分比，前端只能显示“正在处理第 N 段”。
- `completed` 后用 `detail=full` 获取一次完整响应，读取 `result.text`。
- `failed` 后读取 `error.code/message/details`。
- 用户取消时调用 `DELETE /v1/jobs/{job_id}`；如果服务端正在推理当前 chunk，取消可能要等当前 chunk 完成后生效。

6. 处理错误码：

- `401 unauthorized`：检查客户端 Token 配置，不得通过去掉认证或无穷重试绕过。

- `409 model_loading`：等待 3 秒后重试，最多 20 次。
- `409 model_unloading_scheduled`：切换其他可用模型，或提示用户稍后重试。
- `413 audio_too_large`：提示用户压缩、裁剪或分段音频，不要自动重试同一个文件。
- `422 capability_not_supported`：去掉不支持的参数后重试一次。
- `429 job_queue_full`：提示服务端队列已满，等待一段时间后再创建 job；不要切到同步长请求绕过队列。
- `404 job_not_found`：job 可能已过期、服务重启或 ID 错误；提示重新提交。
- `503 gpu_unavailable`：提示服务端 GPU/显存不可用，不要无限重试。

7. 所有局域网 ASR 请求都必须设置超时：

- 连接超时：5 秒。
- 同步转写读取超时：至少 1800 秒。
- job 创建读取超时：30 秒。
- job 轮询读取超时：30 秒；轮询间隔建议 1-3 秒。

8. 完成后给出以下验收结果：

- `/health` 响应摘要。
- `/v1/models` 中发现的模型列表。
- 一个真实音频文件的转写结果，至少包含模型 ID、语言、文本前 200 字。
- 如果实现 job，给出 job 创建响应、至少一次 transcribing 轮询响应、completed 响应摘要。
- 如果失败，给出 HTTP 状态码、错误 JSON、是否经过代理、是否能 ping 通 Windows 主机的实际 LAN IP。

不要硬编码 MiMo、MOSS 或任何服务端未在 `/v1/models` 声明的模型。不要修改 ASR 服务端。不要把请求发往公网。不要通过代理访问 Windows 主机的实际 LAN IP。

## 当前 job / 能力与鉴权合约

创建响应的任务标识字段是 `id`，路径中的 `{job_id}` 使用该值。轮询使用 `detail=compact`；completed 后请求一次 `detail=full` 读取 `result.text`。如需已完成 chunk 的文字，也使用 full。任务在内存中保存，重启后丢失，不能承诺服务端断点续算。

使用 `purpose=podcast` 时，REST 必须显式选择已声明的 MOSS 模型；它不会自动从默认 Qwen 切换模型。读取 `diarization.status` 和 `speaker_scope`，保留 partial / mixed / unresolved；4–5 小时整期一致性仍未完成完整实测。

启用 `ASR_API_TOKEN` 时，所有 `/v1/**` 请求提供 Bearer 凭据；客户端 CLI 用 `ASR_TOKEN`，MCP 用 `ASR_MCP_TOKEN`。`.env` 只有部署生成的 systemd unit 会读取，客户端和直接 Python 启动不自动加载。
