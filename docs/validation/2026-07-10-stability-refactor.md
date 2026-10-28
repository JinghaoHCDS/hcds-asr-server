# ASR Server 稳定性审查与渐进式重构记录

> 历史摘要：以下结果属于原日期、环境和版本，本轮未重新验证。原始验收附件已按维护决定从当前树删除，未另存副本；Git 历史未改写。

日期：2026-07-10  
审查基线：`754b8844c4bbbad70a412c11e9d506eee439931d`  
环境：WSL Arch Linux、Python 3.12、RTX 5070 Ti 16GB、torch `2.11.0+cu128`

## 目标

本轮只处理会造成 OOM、永久卡死、状态错乱、错误转写或资源泄漏的问题。没有引入 Web UI、vLLM、数据库队列或跨 chunk 全局 speaker identity。

## 重构后的调用链

```text
UploadFile
  -> WorkspaceManager 按 1 MiB 流式写 job/request workspace
  -> ffprobe(Path, deadline)
  -> ffmpeg(upload Path -> normalized.wav Path, deadline)
  -> split_audio_path -> ChunkDescriptor(start, end, normalized_path)
  -> Job FIFO 或同步入口
  -> 全局单 GPU operation lock
  -> 必要时卸载上一模型
  -> ProcessRpcTransport(request_id, pid, deadline)
  -> Worker 接收 AudioPath，不接收完整音频 bytes
  -> Worker 只物化当前 chunk
  -> 模型推理
  -> chunk text merge
  -> MOSS segment 校验、overlap ownership、speaker scope
  -> 内存 Job result
  -> workspace cleanup / idle model unload
```

## 已建立的核心不变量

### 单 GPU

- 任意时刻最多一个模型执行 load、transcribe 或模型切换。
- 请求新模型时，旧模型即使仍处于 idle unload 等待期也会先卸载。
- 同步请求和异步 Job 使用同一个 `ModelLifecycleManager` GPU 锁。
- Worker fatal error 后，在旧进程被 reap 前不会开始下一次模型操作。

### Worker

- 所有请求都有单调 request_id。
- 启动必须通过 ping 握手。
- load、inference、shutdown 分别有 operation deadline。
- timeout、EOF、BrokenPipe、协议错乱和 CUDA OOM 都会 poison 当前 Worker generation。
- poisoned Worker 必须 terminate/kill 并 join；下一次 load 创建新进程。
- 收到 shutdown ACK 后先等待 Worker 的 finally 正常结束，再使用 terminate/kill 兜底。
- asyncio 请求取消不会提前把 active_requests 归零；当前底层操作仍由生命周期管理器持有。

### 音频与磁盘

- HTTP 生产路径不再调用 `UploadFile -> list[bytes] -> b"".join`。
- ffprobe 和 ffmpeg 直接接收 Path。
- splitter 返回 descriptor，不保留全部 chunk bytes。
- multiprocessing Pipe 传递 Path、start、end 和配置，不传音频内容。
- Worker 同时最多物化一个当前 chunk；真实 adapter batch 固定为 1。
- `split_strategy=none` 不允许超过 300 秒。
- 每文件最多 4096 个 chunk，窗口在物化前验收。
- 上传受单文件 bytes、进程级 spool bytes 和最小剩余磁盘空间三重限制。
- workspace 由父进程统一清理，Worker 被 kill 后的当前 chunk 也位于该 workspace。

### 配置生命周期

- `LoadConfig` 只包含 model_id、backend、device、dtype、revision。
- max_new_tokens、language、context、hotwords 属于请求/生成配置，不触发模型重新加载。
- 同一个 LoadConfig 的显式 load 幂等。
- 模型 status 输出当前 LoadConfig 和固定 revision。

### MOSS segment

- start/end 必须有限、非负且 `start <= end`。
- segment 不得超过当前 chunk；0.05 秒以内浮点误差可 clamp，严重越界丢弃。
- 空文本和非法结构丢弃并返回 warning。
- parser 抛错时保留 raw transcription，segments 为空并返回 warning。
- overlap segment 依据相邻 chunk ownership midpoint 归属，只保留一个 owner。
- speaker 输出为 `chunk-NNNN:S01`，并同时返回原始 `speaker_label`、`speaker_scope=chunk`、`chunk_index`。
- 存在有效 segments 时，顶层 text 从最终保留 segments 生成，避免 verbose_json 与 text 内容分叉。

### Job

- shutdown 立即取消 queued Job，不再排空整个队列。
- running Job 请求在 chunk 边界取消；超过 shutdown grace 后强制 abort Worker。
- TTL sweeper 真正从 `_jobs` 删除记录；删除后返回 `404 job_not_found`。
- Job workspace 在 completed、failed、cancelled、shutdown 全路径清理。
- progress 现在包含当前 chunk start/end。

## 对抗性验证结果

### 自动化

- 单元/API/故障测试：`123 passed, 2 skipped`，命令为 `uv run --frozen pytest -q`。
- 静态检查：`uv run --frozen mypy asr_server tests scripts` 通过，覆盖 45 个源文件。
- 新增覆盖：跨模型 GPU 互斥、取消 ownership、Worker hang/startup crash/OOM、六小时 sparse WAV descriptor、极小 chunk、Job TTL/shutdown、MOSS 非有限时间和 parser failure、依赖 revision pin。
- GitHub CI 固定 Python 3.12 与 uv 0.11.25，使用 `uv sync --frozen`，执行同一套 pytest 与 mypy 门槛；CI 不安装 CUDA 或模型依赖。

### 六小时低内存边界

使用 16kHz mono s16le 的六小时 sparse WAV 验证 fixed splitter：

- duration：21,600 秒。
- 120 秒窗口、2 秒 overlap：184 个 descriptor。
- splitter 不创建 `AudioChunk.audio` bytes。
- 该测试证明 descriptor 生成是常量音频内存，但不等于六小时真实模型转录性能验收。

### 固定依赖与模型 snapshot

- Transformers commit：`0bc355418bb265136a66c2dedc501066ffbc237d`
- MOSS package commit：`b5ad0f8386b155ddb89f9332ba3ca71891900357`
- Qwen 0.6B snapshot：`6aa69c382e2b426eee1f5870d4c95859a74b6445`
- Qwen 1.7B snapshot：`057a3b044fcd31c433e7971ab40d68d20e7eae6d`
- MOSS snapshot：`d7231bbae2587a4af278735eb765b318c4f64edd`

在 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 下：

- Qwen 0.6B fixed snapshot：通过，输出非空中文文本。
- Qwen 1.7B fixed snapshot：通过，输出非空中文文本。
- MOSS fixed snapshot、`language=zh`：通过，输出 2 个可解析 segments。
- MOSS fixed snapshot、`language=en`、中文音频：仍输出中文。

因此 MOSS 的公开 languages 已从 `auto/zh/en` 收缩为 `auto`。英文语言控制没有通过真实验收，不能声明支持。

### HTTP 真实模型路径

使用临时 `127.0.0.1:18081` 完成：

- FastAPI 上传短音频。
- Path-based FFmpeg 与 descriptor splitter。
- Qwen 0.6B Worker load/transcription，HTTP 200、文本非空。
- 随后请求 Qwen 1.7B，HTTP 200、文本非空。
- 切换后 0.6B 状态为 unloaded，1.7B 为 loaded。
- 1.7B status 中 revision 与固定 snapshot 一致。
- Ctrl-C shutdown 后 Worker 正常退出，无 multiprocessing semaphore leak warning。

## 仍需验证

以下项目没有伪装成已完成：

1. 真实 CUDA OOM fault injection 后的新 Worker 恢复。当前已有 mock/transport 测试，但未主动让生产 GPU OOM。
2. 六小时真实压缩音频的端到端 wall time、RSS、临时盘峰值和最终识别质量。
3. bounded streaming Silero。当前 Path 模式明确 warning 并 fallback 到流式 energy VAD，避免旧实现的 13.5 GiB 理论内存峰值。
4. MOSS 英文控制。当前真实反例已导致能力关闭。
5. systemd 正式部署目录 `/path/to/user/services/asr-server` 的升级、restart 和 Windows/Mac LAN 验收。
6. context/hotwords 在 Qwen HF native helper 中仍没有经过真实有效性验收；响应会保留 warning。

## 回滚边界

逻辑上保持以下独立模块边界，便于按提交拆分或回滚：

1. 故障测试。
2. 全局 GPU 生命周期。
3. ProcessRpcTransport。
4. Path/workspace 音频管线。
5. MOSS language/segment。
6. LoadConfig 分离。
7. Job shutdown/TTL。
8. API schema 提取。
9. 依赖 pin、部署参数和文档。

正式提交时应按上述边界拆成聚焦提交，不建议把整个工作树压成一个无法独立回滚的提交。
