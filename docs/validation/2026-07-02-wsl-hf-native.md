# WSL HF Native Qwen3-ASR 验收记录

> 历史摘要：以下结果属于原日期、环境和版本，本轮未重新验证。原始验收附件已按维护决定从当前树删除，未另存副本；Git 历史未改写。
> 本页短样本结果对应已退出的 13.06 秒旧录音；同路径现改为合成样例，不能用它复现旧样本的质量或计时数字。

## 环境

- 日期：2026-07-02 15:29:19 +08:00
- WSL 发行版：Arch Linux，kernel `6.6.87.2-microsoft-standard-WSL2`
- 项目路径：`/path/to/user/project/asr_Server`
- Git commit：`4885355`，工作区有本次迁移改动
- Python 版本：`uv run python --version` -> `Python 3.12.13`
- uv 版本：`uv 0.11.25`
- torch 版本：`2.11.0+cu128`
- torch CUDA 版本：`12.8`
- transformers 版本：`5.13.0.dev0` from Hugging Face Transformers main
- GPU 名称：`NVIDIA GeForce RTX 5070 Ti`
- GPU capability：`(12, 0)`
- `nvidia-smi` 摘要：driver `596.49`，CUDA runtime `13.2`，VRAM `16303 MiB`

## 依赖门槛

- 初始 `transformers 4.57.6` 不包含 `AutoModelForMultimodalLM`，不能运行 Qwen3-ASR HF native。
- 已安装 `transformers @ git+https://github.com/huggingface/transformers`。
- Transformers main 曾解析到 `numpy 2.5.0`，与 `numba 0.65.1` 不兼容；已 pin `numpy<2.5` 并回到 `numpy 2.4.6`。
- 重新验收后 torch 仍为 CUDA wheel，CUDA 可用。

## 后端预验收

### qwen3-asr-0.6b

- 命令：`uv run python scripts/qwen_asr_backend_smoke.py --backend transformers --model Qwen/Qwen3-ASR-0.6B-hf --audio test-fixtures/audio/test_short.wav --language auto`
- 模型 ID：`Qwen/Qwen3-ASR-0.6B-hf`
- 后端：`transformers`
- loader：`hf-native`
- 音频文件：`test-fixtures/audio/test_short.wav`
- 输出语言：`Chinese`
- 历史转录输出：非空；原始正文已移除。
- 是否通过：通过

### qwen3-asr-1.7b

- 命令：`uv run python scripts/qwen_asr_backend_smoke.py --backend transformers --model Qwen/Qwen3-ASR-1.7B-hf --audio test-fixtures/audio/test_short.wav --language auto`
- 模型 ID：`Qwen/Qwen3-ASR-1.7B-hf`
- 后端：`transformers`
- loader：`hf-native`
- 音频文件：`test-fixtures/audio/test_short.wav`
- 输出语言：`Chinese`
- 历史转录输出：非空；原始正文已移除。
- 是否通过：通过

## 服务端 API 验收

- 启动命令：`ASR_ADAPTER=qwen ASR_IDLE_UNLOAD_SECONDS=0 uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080`
- `/health` 摘要：`status=ok`，GPU available，`NVIDIA GeForce RTX 5070 Ti`，VRAM `16302 MB`
- `/v1/models`：只声明 `qwen3-asr-1.7b` 和 `qwen3-asr-0.6b`，能力仍为 offline transcription + `backends=["transformers"]`
- `qwen3-asr-0.6b` + `transformers` 历史输出：非空；原始正文已移除。
- `qwen3-asr-1.7b` + `transformers` 历史输出：非空；原始正文已移除。
- `timestamps=word` 未声明能力错误码：`422 capability_not_supported`
- `backend=vllm` 未声明后端错误码：`422 capability_not_supported`
- 服务停止验证：加载 `qwen3-asr-0.6b` 后 Ctrl-C 停止服务，worker 不再打印 `KeyboardInterrupt` traceback。

## 测试

- `uv run pytest -q`：`83 passed, 2 skipped`
- `uv run mypy asr_server tests scripts`：通过

## 未完成项

- 未执行 Mac mini 到 `http://YOUR_WINDOWS_LAN_IP:18080` 的局域网验收。
- 未执行长音频异步 job 的真实 Qwen HF native 转录验收。
- `torch.compile` 未启用，第一阶段保持关闭。
