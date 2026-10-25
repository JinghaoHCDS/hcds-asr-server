# 2026-10-25 WSL 平台实机验收

本轮平台代码在独立 WSL 目录完成真实 GPU 验收，未替换线上服务、安装依赖或下载模型。Qwen 0.6B、1.7B、官方 ForcedAligner、MOSS 显式预算路径、自动模型切换和检查点恢复通过。MOSS 三分钟样本的原自动预算不足，按现有规则受控失败，详见下文。本记录证明运行行为，不提供识别准确率或六小时语音质量结论。

## 环境和边界

- 用户明确授权“本任务在 WSL 独立验收”。测试目录为 `/home/fragt/project/asr-platform-validation-20261025-01`，线上目录 `/home/fragt/services/asr-server` 仅提供现有解释器与只读模型缓存。
- Arch Linux / WSL2，Python 3.12.13，RTX 5070 Ti 16 GB，驱动 621.01；torch `2.11.0+cu128`、CUDA runtime 12.8、GPU capability 12.0。实际 CUDA tensor 运算通过。
- transformers `5.18.0.dev0`、FastAPI `0.138.1`、Starlette `1.3.1`、MOSS 包 `0.1.0`；FFmpeg `n8.1.2`。Qwen 使用 HF native transformers，没有安装 `qwen-asr`。
- `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；复用已有 Silero 6.2.3。独立 ASGI lifespan 通过 HTTPX 驱动，不新增监听端口，不读取线上工作台数据库。
- 基线从 `2fb6a4dca0375ad2cbc3a20b2e3aa5b4b71cf852` 解包；当前树来自本地源码包及验收工具覆盖。源码、音频和工具摘要保存在脱敏证据中。

| 模型 | 固定 revision |
| --- | --- |
| Qwen/Qwen3-ASR-0.6B-hf | `6aa69c382e2b426eee1f5870d4c95859a74b6445` |
| Qwen/Qwen3-ASR-1.7B-hf | `057a3b044fcd31c433e7971ab40d68d20e7eae6d` |
| Qwen/Qwen3-ForcedAligner-0.6B-hf | `c07281df297b9905d24a508279258cccf987a064` |
| OpenMOSS-Team/MOSS-Transcribe-Diarize | `d7231bbae2587a4af278735eb765b318c4f64edd` |

三个音频片段来自用户指定 Windows 目录的不同节目：开发 30 秒、验收 180 秒和验收 600 秒。Windows FFmpeg 8.0.1 提取，转移后核对 SHA256。[来源清单](validation-artifacts/2026-10-25-platform/samples.json)不含音频。原音、完整转录、账号哈希、数据库与原始响应仅保存在本次忽略数据目录，不进入 Git 或源码包。

## 真实模型与恢复

| 场景 | 结果 |
| --- | --- |
| Qwen 0.6B 独立 backend / 30 秒 / `zh` | 非空中文输出；随后 HTTP 同步和内存异步各通过，均为 131 字符 |
| 官方 ForcedAligner 独立 smoke | 真实对齐通过；补齐独立脚本的显式语言 prefill，两个语言模板单测在 Mac/WSL 均通过；服务推理 prefill 未改变 |
| Qwen 1.7B 工作台 / 180 秒 | 4 个 checkpoint，1123 字符，991 个官方时间戳；TXT/JSON/SRT/VTT、编辑 v2 后重新对齐、ETag、Range、owner/Bearer 隔离及删除通过 |
| Qwen 1.7B 工作台 / 600 秒 | 11 个 checkpoint，3425 字符，3091 个官方时间戳；同样覆盖四格式导出、编辑对齐、隔离、删除 |
| MOSS 独立 backend / 30 秒 | CUDA 推理及非空输出通过 |
| MOSS / 180 秒 / 自动预算 | `2160` tokens 触顶，返回 `generation_truncated`；未把截断结果标记 completed |
| MOSS / 同一 180 秒 / 显式 8192 | 2480 个生成 token、1109 字符、108 段；native long-form，speaker_scope 为 global，输出覆盖检查通过；覆盖率不等于准确率 |
| 自动切换 0.6B → 1.7B → MOSS | 两次切换均自动卸载并回收前 worker；中间没有手动 unload，始终只加载一个 ASR runtime；最后手动卸载通过 |
| 真实分块处理在途恢复 | 4 块中的第 1 块 checkpoint 原子提交后正常停止；新 app lifespan 和新 worker 续跑完成。命中原 checkpoint 一次，原块正文/摘要不变，规范音频 inode/mtime/大小/摘要不变，恢复后新归一化次数为 0 |
| WSL decoder 策略 | 十种普通容器兼容、concat/local HLS/network HLS 拒绝，共 13 项通过；未发现可接受的外部引用复现，不宣称已确认 SSRF |

恢复使用本地验收 hook 选择真实 checkpoint 提交后的安全边界，然后调用正常 shutdown；没有中断正在执行的 CUDA，也没有重启线上 systemd。这证明应用生命周期与 worker 重建后的持久恢复，未覆盖机器断电和 GPU 驱动重启。

MOSS 自动预算计算和触顶拒绝逻辑相对基线未变。单块 180 秒采用 `ceil(duration × 12)=2160`；Anchor Replay 的 24000 下限只适用于相应多块说话人流程。平台改进计划要求保留模型预算策略，因此本轮记录该现有限制，并用客户端显式预算验证成功路径。

## 显存与耗时

以下是独立验收命令的完整进程墙钟，包含启动、模型加载、HTTP 检查和清理，不能当作单次纯推理耗时。显存通过 nvidia-smi 每 0.5 秒采样整张卡，包含 Windows 和其他进程；它不是 worker 的 torch allocated memory。

| 场景 | 进程墙钟 | 全卡采样峰值 | 退出后 |
| --- | ---: | ---: | ---: |
| Qwen 0.6B backend | 29.00 s | 3389 MiB | 1240 MiB |
| Qwen 0.6B 同步 + 异步 | 32.17 s | 3501 MiB | 1240 MiB |
| 1.7B 工作台 180 秒 | 36.91 s | 7801 MiB | 1240 MiB |
| 1.7B 工作台 600 秒 | 70.34 s | 7857 MiB | 1240 MiB |
| MOSS 180 秒显式 8192 | 67.02 s | 3915 MiB | 1240 MiB |
| 在途恢复两次 lifespan | 49.05 s | 8019 MiB | 1240 MiB |
| 三模型自动切换 | 22.89 s | 5353 MiB | 1240 MiB |

各项测试前全卡基线为 1240 MiB。卸载时另检查了 worker 不再存活、runtime unloaded、活跃请求与临时预留归零；没有把服务退出后的进程回收替代 API unload 检查。

单独的 MOSS native 180 秒资源补测通过，每 0.5 秒读取本次进程组的 `/proc` RSS：网关峰值 **604.48 MiB**，确认的模型 worker 峰值 **3375.38 MiB**，所有子进程同时存活的 RSS 合计峰值 **3389.13 MiB**。它们是绝对 RSS，包含解释器与依赖；共享页会重复计数，不是处理增量。MOSS 结果另报 torch allocated VRAM 峰值约 **2300.57 MiB**。该轮全卡基线/收尾均为 1283 MiB，采样峰值 3958 MiB；不同时间的桌面基线有变化，不能将全卡总量直接归因于 worker。MOSS native 整段输入的内存单独列示，不套用 Qwen 路径管线的 256 MiB 网关增量预算。

| 单次任务 | started→completed | created→completed（含排队） | RTF_processing / RTF_end_to_end |
| --- | ---: | ---: | ---: |
| Qwen 1.7B / 180 秒 | 29.496 s | 29.742 s | 0.1639 / 0.1652 |
| Qwen 1.7B / 600 秒 | 62.094 s | 62.535 s | 0.1035 / 0.1042 |

这两项是冷模型、暖磁盘缓存的单次任务样本，不是三轮统计。created 从上传完成建立 durable job 时计算，排除之前上传以及 completed 后的编辑、导出、删除；started 从持久开始时间计算。规范化可能在 scheduler prepare 阶段完成，因此 created→completed 是较完整的任务耗时，pipeline 返回的 decode_ms=0 表示借用既有规范资产。不能将重叠阶段简单求和。

## 配对性能比较

基线和当前版使用同一份 180 秒验收音频、相同工作台 smoke helper、模型/revision/profile/检测器/热词摘要、现有解释器与暖磁盘缓存。各轮独立冷 model 进程，真实 GPU 串行运行。工作负载包括上传、识别、官方对齐、编辑重对齐、导出和删除。`RTF_workflow = 完整进程墙钟 / 180 秒`，包含处理与控制流程；它不是单 job 的纯 inference RTF。

| 源码 / 日志 | 三轮中位数 | 范围 | workflow RTF 中位数 |
| --- | ---: | ---: | ---: |
| 基线 | 30.690 s | 29.241–31.062 s | 0.1705 |
| 当前 INFO | 31.543 s | 31.142–32.148 s | 0.1752 |
| 当前 CRITICAL | 31.722 s | 30.895–32.051 s | 0.1762 |

当前 INFO 中位数相对基线增加 **2.78%**，本负载没有端到端加速证据。九轮模型、profile、prompt、词库和运行时身份一致，输出正文 SHA256 相同，均有 991 个官方时间戳；一致输出不等于人工准确率。固定按基线→INFO→CRITICAL 分组测量，顺序和小样本会影响结果。

INFO 相对 CRITICAL 中位数为 **−0.57%**，两组范围重叠，未观察到明显的完整工作负载日志惩罚；这不能估计稳定的负开销，也不能保证所有场景都低于计划中的 3%。配对与预取结果见[脱敏 GPU 证据](validation-artifacts/2026-10-25-platform/gpu-summary.json)。

预取另使用同一开发音频连续排队三个 Qwen 0.6B legacy 任务，显式 `zh`、4096 tokens，off/on 交替各三轮。每轮新网关/worker，事先预读同 revision 权重与音频，十八项任务均完成；每项归一化一次、每轮模型只加载一次、FIFO 不变、正文 SHA256 全部一致，最终卸载及临时配额回收通过。

| 预取 | 批次墙钟中位 / 范围 | 批次 RTF 中位 | 两次转录 RPC 间隔合计中位 / 范围 |
| --- | ---: | ---: | ---: |
| off | 12.810 / 12.408–14.961 s | 0.1423 | 93.45 / 88.88–93.60 ms |
| on | 12.126 / 12.090–13.738 s | 0.1347 | 12.03 / 11.48–12.51 ms |

批次中位数下降 **5.34%**，范围重叠；间隔约减少 81 ms，远小于批次耗时波动，不把全部下降归因于预取。该间隔由 DEBUG 中实际 transcribe RPC 的完成 UTC 时间减单调 duration 推算，包含 IPC/调度开销，是应用层代理指标，不是 CUDA kernel idle。测试开启同等 DEBUG 以取得区间，不与上表 INFO/CRITICAL 实验混用。默认继续保持 `ASR_PREPROCESSING_PREFETCH=0`。

这组 off 实验第一项冷加载的处理 RTF 中位数为 0.2591，随后两项复用模型为 0.0745 / 0.0755；每轮 load 事件约 3.60–3.80 秒。on 模式每项 started 在预取准备后发生，跨模式比较 started→completed 会漏计重叠解码，因此收益判断使用含上传/排队/准备的批次墙钟。所有逐任务时间和 pipeline timings 均保留在 JSON。

## 线上未变与交付

验收前后 `asr-server.service` 均为 active，MainPID 均为 **211**，WorkingDirectory 未变；`/health`、`/v1/models` 均为 200，三个线上模型均 unloaded、active_requests=0。选定线上源码与依赖文件 SHA256、torch/transformers 版本一致。测试结束后检查无本次 Python 进程、无 CUDA compute PID；全卡回到末轮测试前的 **1283 MiB**，媒体策略 scratch 已删除。独立源码、私有原始证据和三段输入保留供复核。

脱敏 JSON 包含测试源码逐文件 SHA256，已与最终本地对应文件核对。原始证据归档 SHA256 为 `6f38f636a80eb7435157f7f150f46a00fdca6234af0bbf35d0a0ba3476c90235`；其 313 个证据文件在下载后逐个核验。最终源码/前端包和本地演练数据库已重新配对相同 source_digest，并在新目录完成恢复核验；演练不操作线上账号、任务或音频。

## 复现与未验收项

在已满足项目 GPU 门槛的隔离 WSL 目录，使用现有 Python，禁止把本例指向线上数据：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ASR_QUALITY_VAD_DIR=/path/to/existing/silero-vad-6.2.3
python -m scripts.platform_gpu_validation \
  --dev-audio /path/to/dev-30s.wav \
  --acceptance-audio /path/to/acceptance-180s.wav \
  --output-dir /path/to/new-private-run \
  --phase all --enable-moss --moss-max-new-tokens 8192 --restart-inflight
python scripts/platform_gpu_compare.py \
  --baseline-source /path/to/baseline-source \
  --current-source /path/to/current-source \
  --audio180s /path/to/acceptance-180s.wav \
  --output-dir /path/to/new-comparison --repeats 3 --compare-logging \
  --quality-vad-dir "$ASR_QUALITY_VAD_DIR"
python -m scripts.platform_prefetch_probe \
  --audio30s /path/to/dev-30s.wav \
  --output-dir /path/to/new-prefetch-comparison --repeats 3
```

这些工具的 `--mock` 仅供无 GPU 的验收工具检查，不能提供真实吞吐证据。完整本地行为回归为 380 passed / 4 skipped，前端构建与发布依赖检查通过，详情见[本地记录](validation-2026-10-25-platform.md)。

没有人工参考文本，未验收 CER、术语准确率、字幕语义准确性、六小时真实语音质量、其他硬件/发行版、长期并发运行、Windows 原生 runtime、vLLM 或原生 GPU batching。600 秒结果通过说明该素材运行完整，不将输出字符数、对齐来源或 MOSS 覆盖检查当作质量达标。
