# 已接入模型的推理实践与来源

核对日期：2026-10-25。适用 Qwen3-ASR 0.6B/1.7B、Qwen3-ForcedAligner 0.6B 和 MOSS-Transcribe-Diarize 0.9B。以下是官方资料与当前固定版本的代码对照，不表示所有参数都已取得本项目最优准确率。

## Qwen3-ASR

依据：[官方 HF native 模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)、[官方仓库](https://github.com/QwenLM/Qwen3-ASR)、[当前固定 Transformers processor](https://github.com/huggingface/transformers/blob/ea31b0c6ab0d1368130b0482aee7293a014c3a38/src/transformers/models/qwen3_asr/processing_qwen3_asr.py)。

- 保持 native `AutoProcessor` / `AutoModelForMultimodalLM`，BF16、`eval()`、`inference_mode()` 和 `do_sample=False`。只解码新增 token，使用 processor 的 parsed 输出。
- 明确语言交给官方 helper 规范成完整语言名称，再作为 assistant prefill；context/hotwords 放 system prompt。固定模型模板遗漏 assistant 消息的兼容补丁继续保留。自动语言路径使用 generation prompt，避免旧模板的空 assistant prefill 问题。
- 长输入需要足够生成预算。`quality_v1` 默认 4096；显式值保持原样；旧 API 默认 512 保持兼容。达到上限且未遇 EOS 时报告 `generation_truncated`，不能把截断稿当完整结果。
- 16 kHz、单声道 PCM 输入与[官方音频规范化实现](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py)一致。保留原音时间坐标和完整覆盖；不采用会静默改写重复文字的通用清洗。

代码：`asr_server/adapters/qwen.py`、`audio/assets.py`、`quality/pipeline.py`。测试：`test_qwen_adapter.py` 的两尺寸加载、语言 prefill/auto、上下文和生成路径；`test_quality.py` 的预算、完整覆盖、复核与检查点场景。

## 强制对齐

依据：[官方 ForcedAligner 模型卡](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)、[官方输入窗口实现](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py)。

先转录，再将真实音频、文本、语言交给 `prepare_forced_aligner_inputs`，调用模型并 `decode_forced_alignment`。时间来自模型输出；跨块加回原媒体偏移，不能按字数均分。官方模型卡说明支持五分钟，SDK 的对齐切分采用更保守的 180 秒；本项目 quality 单块最多 180 秒，legacy 保留原 300 秒硬上限，不能据此声称两种窗口质量等价。对齐器语言能力少于 ASR，仍须单独验收目标语言。

代码：`adapters/qwen.py::_align`、`quality/splitter.py`、`audio/merger.py`；测试：`test_forced_alignment_smoke.py`、`test_quality.py` 和 `test_audio_splitter.py`。

## MOSS-Transcribe-Diarize

依据：[官方 Transformers 示例与长音频说明](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize#python-usage)、[固定 helper 实现](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/b5ad0f8386b155ddb89f9332ba3ca71891900357/moss_transcribe_diarize/inference_utils.py)、[固定模型代码](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/d7231bbae2587a4af278735eb765b318c4f64edd/modeling_moss_transcribe_diarize.py)。

- 使用官方 `build_transcription_messages`、`generate_transcription`、`parse_transcript`；BF16、`eval()`、greedy generation。固定 helper 内部已有 inference mode 和 autocast。
- 本次显式设为 `attn_implementation="sdpa"`。官方建议 SDPA 或 FlashAttention，避免长音频 eager attention 的二次方内存增长；固定模型代码已声明 SDPA 支持。加载不静默降级 eager，也不新增 GPU 依赖。
- 保留按输入时长计算生成预算、Anchor Replay 自动预算下限 24000、显式值优先及 65536 上限。官方短样例的 2048 不能当作长播客通用预算。
- `[start][speaker]text[end]` 是段级时间。本次字幕直接使用经过范围、顺序、完整文字检查的段级时间，标记 `segment_timed`，不声称逐字对齐。旧结果也可由模型能力授权后导出。
- 官方新版描述更长输入及其他服务后端；当前部署仍保留已验收的 1801 秒 native 门槛和后续切分/Anchor Replay，不据上游宣传扩大本机能力。

代码：`adapters/moss.py`、`execution.py`、`diarization/anchor_replay.py`、`workbench/service.py`；测试：`test_moss_adapter.py`、`test_execution_policy.py`、`test_review_regressions.py`。

## 有意保留的选择

`vLLM`、SGLang、FlashAttention 安装、`torch.compile`、真实 batching 均需新的显存、输出一致性和生命周期测试，当前不启用。上游吞吐数字不能直接外推到本机单 GPU FIFO。模型和依赖 commit pins 不变；来源见 `registry.py` 和 `requirements/wsl-gpu-cu128.txt`。

本次修改已部署到 WSL 线上服务，并完成真实三模型转录、有限参考文本 WER/CER、跨切块音频及字幕导出验收。独立探针确认 MOSS 的语音和文本注意力模块实际采用 SDPA。详见[本轮部署验收](validation-2026-10-25-review-fixes.md)；样本范围不支持六小时真实语音质量或普遍性能提升结论，全卡显存峰值本轮未测得。


## 本轮本地验证

- `pytest -q`：393 passed、4 skipped；包含新增 13 项审查回归。用真实 FFmpeg 验证约 16 秒 probe 估算的 MP3 能完整解码约 70 秒。
- `mypy asr_server asr_mcp tests scripts`：116 个文件通过。
- `python -m scripts.check_release_dependencies`：通过；模型及依赖 pins 未变。
- `cd frontend && npm run build`：TypeScript 和生产构建通过。
- 取消/删除覆盖运行中的准备阶段与排队预取；检查 partial 文件、资源余量及终态。队列测试验证容量释放后较早任务优先于新请求。字幕测试包括旧结果、原始段级时间、非法时间与 chunk 窗口拒绝。
