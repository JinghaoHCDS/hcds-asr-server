# 2026-10-22 Qwen quality_v1 与工作台实机验收

> 历史摘要：以下结果属于原日期、环境和版本，本轮未重新验证。原始验收附件已按维护决定从当前树删除，未另存副本；Git 历史未改写。
> 本页短样本结果对应已退出的 13.06 秒旧录音；同路径现改为合成样例，不能用它复现旧样本的质量或计时数字。

## 结论与范围

在 WSL2 Arch Linux、RTX 5070 Ti 上，通过了独立 Qwen3-ASR 0.6B 后端 smoke，以及默认 1.7B 的工作台完整短音频流程：私有任务、真实转录与对齐、字幕导出、编辑指定文本后局部重对齐、音频 Range 和删除。

这是当时 13.06 秒旧录音样本的功能验收。`quality_v1` 仍明确返回 `validated=false`；本记录不能证明 6 小时质量、术语召回率、最优切分、边界错误率或字幕 P95 已达标。

线上部署没有替换、没有重启、没有安装或同步依赖。源码仅放入独立目录 `/path/to/user/project/asr-quality-validation-20261022`；验证采用进程内 ASGI 调用，没有额外监听 LAN 端口。

## 环境与输入

- Python：3.12.13；解释器复用 `/path/to/user/services/asr-server/.venv/bin/python`。
- OS：Arch Linux / WSL2，Linux `6.6.87.2-microsoft-standard-WSL2`。
- GPU：NVIDIA GeForce RTX 5070 Ti 16GB，CUDA capability `(12, 0)`。
- torch：`2.11.0+cu128`；CUDA runtime：`12.8`；`torch.cuda.is_available()` 为 true。
- transformers：`5.18.0.dev0`。
- 全程 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，使用已有模型缓存。
- 历史输入：已退出的 13.06 秒旧录音；原始音频和正文不再分发。本页的计时、对齐单元和文本比较只对应该旧输入。
- 0.6B 模型：`Qwen/Qwen3-ASR-0.6B-hf@6aa69c382e2b426eee1f5870d4c95859a74b6445`。
- 1.7B 模型：`Qwen/Qwen3-ASR-1.7B-hf@057a3b044fcd31c433e7971ab40d68d20e7eae6d`。
- 对齐器：`Qwen/Qwen3-ForcedAligner-0.6B-hf@c07281df297b9905d24a508279258cccf987a064`。

## 独立后端预验收

在独立目录执行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /path/to/user/services/asr-server/.venv/bin/python \
  scripts/qwen_asr_backend_smoke.py --backend transformers \
  --model Qwen/Qwen3-ASR-0.6B-hf \
  --audio test-fixtures/audio/test_short.wav --language auto
```

- exit code：0；输出语言：Chinese。
- 历史转录结果非空；正文已移除。
- 包含加载和退出的墙钟时间：11.99 秒。
- 每 250 ms 采样 `nvidia-smi` 全 GPU 已用显存：起始 2853 MiB，峰值 4814 MiB。

## 默认 1.7B 工作台完整路径

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /path/to/user/services/asr-server/.venv/bin/python \
  scripts/workbench_smoke.py --model qwen3-asr-1.7b \
  --output data/validation/workbench-17b-smoke.json
```

- exit code：0；包含创建应用、模型加载、转录、编辑对齐及退出的墙钟时间：13.06 秒。
- 服务转录 `total_ms=8096.44`，其中单次 ASR `inference_ms=2590.58`，ASR 后处理与对齐 `postprocess_ms=1279.20`；这些不是长音频速度承诺。
- 全 GPU 显存起始 2853 MiB，采样峰值 8994 MiB；相较起始增加 6141 MiB。
- 返回 `quality_profile=quality_v1`、`profile_version=quality_v1.1`、`validated=false`。
- 历史转录结果非空；正文已移除。
- 返回 35 个真实对齐单元、3 个阅读／字幕片段，实际 timestamp source 仅 `qwen3_forced_aligner`；未执行自动规范，`changes=[]`，因此不把本样本作为真实别名规范收益证明。
- 请求包含 `Qwen3-ASR` 及反例 `ABSENTTERMZXQ`；反例没有进入正文。此单一反例也不足以证明整体术语正确率。

| 验证操作 | 结果 |
| --- | --- |
| 随机密码创建用户并登录 | 200 |
| 分片上传、重复片段重放、后续片段续传 | 均成功，offset 一致 |
| 私有 job 创建并完成 | completed，正文非空 |
| 从旧 `/v1/jobs/{id}` 读取私有 job | 404 |
| 版本 1 TXT / SRT / VTT / JSON 导出 | 均 200 |
| 修改已有片段并创建版本 2 | 200 |
| 版本 2 对齐期间导出 SRT | 409 |
| 指定修改文本的真实局部重对齐 | `alignment_status=aligned` |
| 原音 `Range: bytes=0-31` | 206，返回 32 字节 |
| 删除 job | 204，后续读取 404 |

显存数据是整张 GPU 的观测值，包含 Windows／其他进程占用，不等于本模型的 PyTorch allocated 峰值。

## 已确认别名的真实重新对齐

在另一个临时工作台数据库中，将 `Qwen3-ASR` 的已确认别名设为 `千问 ASR`、`auto_normalize=true`，继续使用包含 `Qwen3-ASR` 的本次热词请求。没有修改线上词库。

- 原稿：历史记录为非空，正文已移除。
- 交付稿：确认别名规范后与原稿不同，正文已移除。
- `quality.changes` 记录一项 `confirmed_alias`，保留原字符范围、原词、标准词及音频范围。
- 规范后的实际响应包含 **33 个真实时间单元**，source 仅 `qwen3_forced_aligner`；没有沿用原稿的 35 个时间单元。
- 四种格式导出、版本 2 编辑对齐、待对齐阻止字幕导出、Range 和删除同样通过。
- 墙钟时间 13.06 秒；全 GPU 采样峰值 8994 MiB，退出后回到 2853 MiB 的起始基线。

这验证了该明确语境、明确别名的规范与重新对齐路径，不代表术语总体召回率或误规范率达标。

验收中修复了一处真实发现：本次热词与公共词库具有相同 canonical 时，原实现用空别名请求项覆盖公共已确认别名，导致规范未发生。现在本次输入只提升同名词的提示顺序，保留该词的公共规范元数据。该修复后重新执行默认与别名两条真实路径，均通过。

临时别名验证脚本初次因缺少 multiprocessing `__main__` guard 而重入，已修正验证脚本；这次初始失败没有加载模型，也没有修改产品或线上服务。本页保留该失败原因；不承诺远端原始日志仍可访问。

## 线上前后状态与证据

验证前、验证后均读取线上 `/health`、`/v1/models` 和三个模型的 `/status`：服务健康、模型全部 `unloaded`、活跃请求全部为 0；验证后 `systemctl --user is-active asr-server.service` 仍为 `active`。验证 CLI 退出后，检查独立工作目录下没有残留 Python／Qwen worker 进程，整卡已用显存回到 2853 MiB 基线。

原始日志、JSON 和一次性别名脚本已从当前树删除。本页保留方法、计时、显存、功能与失败摘要；未另存私有副本。

后续仍需许可明确的公开测试集和代表性长素材，按开发／独立验收集分离的方式评测切分候选、同源长短 CER 差异、术语召回与误规范、字幕时间偏差及 30 分钟至 6 小时的完整性。

### 实机验证后的局部修正

当时精确源码身份记录在现已删除的环境附件中，本页不足以还原该完整工作区。实机完成后，本地进一步修正了 `quality/splitter.py` 与 `quality/pipeline.py` 的空结果分支：只有 PCM 全零输入才可豁免空转录错误，不再把低于能量阈值的非零低音量输入当成数字静音。声学空洞候选检测继续使用原能量阈值。

该修正新增 amplitude 为 0、1、20 的回归，`pytest tests/test_quality.py -q` 为 **18 passed**。本轮实机 fixture 返回非空正文，不经过该空结果分支，因此没有为这项分支修正重复运行 GPU；不得把较早实机 SHA 表述为包含该后续修改。
