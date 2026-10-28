# 口播、播客与批量交接工作流

本工作流保留“客户端管理素材和结果，WSL 做 GPU 推理”的边界。调用前先读取 `GET /v1/models`；`oral_analysis` 需要某个模型声明 `forced_alignment=true`，`podcast` 需要某个模型声明 `diarization=true` 且支持 `speaker_resolution=auto`。能力不存在时客户端明确失败，不会静默降级。

## WSL 开关

真实验收通过后，在服务环境中设置：

```bash
ASR_ADAPTER=qwen
ASR_ENABLE_QWEN_ALIGNMENT=1
ASR_ENABLE_MOSS=1
```

对齐器和 MOSS 都按需加载；单 GPU 锁保证它们不会绕过既有调度同时推理。不要在仍有生产任务时重启服务。

## 单条口播

```bash
uv run python scripts/asr_client.py \
  --base-url http://YOUR_WINDOWS_LAN_IP:18080 \
  job /path/to/media/oral-001.wav \
  --purpose oral_analysis \
  --language auto \
  --context '视频简介提到 Qwen3-ASR 和 RTX 5070 Ti；仅作词汇参考。' \
  --hotwords '["Qwen3-ASR","RTX 5070 Ti","Hugging Face"]'
```

口播结果的事实来源是 `result.timestamps`；每项包含真实 `start/end/granularity/source/status`。`result.timing.reading_segments` 只是便于阅读的分组，明确带 `semantic_sentence=false`。`rate.including_pauses` 的分母是原素材总时长，`rate.aligned_speech_only` 的分母是对齐单元覆盖的发声时长。停顿为声学候选，不是“刻意停顿”的内容判断。

## 单条长播客

```bash
uv run python scripts/asr_client.py \
  --base-url http://YOUR_WINDOWS_LAN_IP:18080 \
  job /path/to/media/podcast-001.m4a \
  --purpose podcast \
  --context '一期讨论 Apple Silicon、MLX 与 CUDA。' \
  --hotwords '["Apple Silicon","MLX","CUDA"]' \
  --poll-seconds 30
```

`podcast` 不默认做逐字对齐。最终必须检查 `result.diarization.status` 和 `speaker_scope`：只有 `global` 才能把标签当作整期一致；`mixed` 中 unresolved 段必须保留为未知，不能靠内容猜身份。标签只在本期任务内有效，最多分配 4 个稳定标签。

## REST 精简轮询

状态轮询默认不返回累计全文或最终正文：

```bash
curl --noproxy '*' -sS \
  'http://YOUR_WINDOWS_LAN_IP:18080/v1/jobs/JOB_ID?detail=compact'
```

任务结束后只获取一次完整结果：

```bash
curl --noproxy '*' -sS \
  'http://YOUR_WINDOWS_LAN_IP:18080/v1/jobs/JOB_ID?detail=full'
```

服务端 job 仍是内存态并按 TTL 清理。客户端批量工具保存 job ID；遇到 `job_not_found` 会按同一缓存键和幂等键重新提交。

## 批量清单

从 [batch-manifest.sample.json](../../examples/batch-manifest.sample.json) 复制。每个条目直接绑定路径和元数据，不依赖列表排序：

```json
{
  "items": [
    {
      "id": "creator-episode-001",
      "path": "/path/to/media/001.wav",
      "title": "样例标题",
      "source": "https://example.invalid/item/001",
      "description": "原始简介，仅供审计",
      "purpose": "oral_analysis",
      "language": "auto",
      "context": "外部 Agent 提炼后的简短背景",
      "keywords": ["Qwen3-ASR", "MLX"]
    }
  ]
}
```

首次运行及普通恢复使用同一命令：

```bash
uv run python scripts/batch_transcribe.py batch.json \
  --output output \
  --base-url http://YOUR_WINDOWS_LAN_IP:18080 \
  --max-in-flight 4 \
  --poll-seconds 10
```

显式重试已达到尝试上限的失败条目：

```bash
uv run python scripts/batch_transcribe.py batch.json \
  --output output \
  --base-url http://YOUR_WINDOWS_LAN_IP:18080 \
  --retry-failed
```

缓存键包含媒体内容 SHA-256、模型与 revision、运行时能力以及会影响输出的全部请求字段。文件名相同不会互相覆盖；Context 或关键词改变会生成新缓存键。提交幂等与结果缓存是两件事：前者阻止网络超时造成同一 server job 重复，后者决定本地完成项能否跳过。

## 输出目录

```text
output/
  index.md
  manifest.json
  items/<稳定ID-内容hash>/
    transcript.md
    transcript.json
```

所有文件先写同目录临时文件再原子替换。`transcript.json` 是结构化事实来源，Markdown 只由它生成。口播 Markdown 展示可回看的时间段；播客 Markdown 按发言顺序展示匿名 speaker。大批次先读 `index.md`，不要一次把 300 篇全文塞入分析 Agent 上下文。

仓库中的 [output-sample](../../examples/output-sample) 明确标记为合成样例，不是 GPU 验收结果。

## 回滚

代码回滚后删除服务环境中的 `ASR_ENABLE_QWEN_ALIGNMENT` 即可停止声明和调用对齐器；删除 `ASR_ENABLE_MOSS` 即可停止声明播客模型。回滚不要求删除模型缓存。恢复旧版本前先等待活跃 job 完成，再重启 systemd user service；不要强杀正在执行的 GPU worker。
