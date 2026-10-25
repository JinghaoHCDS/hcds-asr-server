# 平台改进：验证组合与测试去向

日期：2026-10-25。基线：`2fb6a4dca0375ad2cbc3a20b2e3aa5b4b71cf852`。本表记录本轮实际删改和替代证据；整轮的用例数、耗时与实机结果以最终验收记录为准。

## 保留的默认检查

```bash
uv run --frozen pytest -q --durations=20
uv run --frozen mypy asr_server asr_mcp tests scripts
uv run --frozen python -m scripts.check_release_dependencies
```

`integration`、`gpu`、`performance`、`release` 已登记为 pytest markers。没有通过 `addopts` 排除任何现有本地行为测试；`test_api.py` 标为 integration 后仍在默认套件内运行。已有真实 HTTP smoke 仍须显式设置 `ASR_BASE_URL`。GPU backend smoke、批次性能测量和依赖发布检查继续使用明确命令，不能把新增标签解释为已获得测试提速。

独立 GPU 验收入口为 `scripts.platform_gpu_validation`（真实 API/工作台/生命周期/安全恢复）、`scripts.platform_gpu_compare`（同源工作负载的基线和日志对照）和 `scripts.platform_prefetch_probe`（同序列预取开关）。只在已有合格环境的独立目录执行；具体命令与证据见 [WSL 验收](validation-2026-10-25-wsl-platform.md)。独立 ForcedAligner smoke 的 `zh`/`auto` prefill 回归位于 `test_forced_alignment_smoke.py`，不要求安装 GPU 依赖。

## 原用例到替代证据

| 原位置 / 场景 | 实际处理 | 当前证据 |
| --- | --- | --- |
| `test_api.py::test_docs_endpoint_uses_current_capability_notes` | 删除英文句子、换行、标题和启动文案的逐字断言；改名为文档路由 smoke | `/docs`、`/openapi.json`、`/docs/mcp` 状态及响应类型，OpenAPI 可解析且包含转录入口 |
| `test_api.py::test_unsupported_capability_returns_422` | 删除重复的普通 422 测试 | `test_unsupported_capability_is_rejected_before_audio_decode` 同时检查错误码和未调用转录管线 |
| `test_api.py` 三处 vLLM 拒绝及 MOSS 不支持参数 | 合入同一参数矩阵，原 MOSS 三个请求拆为独立参数行 | load、同步转录、异步创建各有实际路由；覆盖 Qwen/MOSS backend、timestamps、MOSS 显式语言拒绝；拒绝不进入 decode |
| `test_api.py` 上传过大及 `test_jobs.py` 同条件重复 setup | API 用例按同步/异步两个入口参数化；删除 job 文件中的重复用例 | 每文件上限返回 413、释放 parser/工作区账本。另由 `test_admission.py` 验证无 Content-Length 流式正文在复制前截止，覆盖的是不同资源边界 |
| `test_api.py` 整份 Qwen/MOSS 能力常量副本 | 合成启用/停用 MOSS 两行公共发现检查 | 响应与运行时同源 `ModelDefinition.capabilities`、revision、default 一致；关键模型窗口仍在 `test_execution_policy.py` 保留产品阈值验证 |
| `test_config.py` 六个同形非法环境值测试 | 合为六行参数表 | 各非法类别及错误中的字段定位均保留；执行次数未因此减少 |
| `test_qwen_adapter.py::test_qwen_model_repos_use_hf_native_checkpoints` | 删除孤立 repo 常量复制 | 实际 load 边界对两种 Qwen 模型检查 processor/model 收到 registry 指定的 repo 和 revision；同时保留 HF native 路径和显式语言回归 |
| `test_moss_adapter.py::test_moss_model_repo_uses_fixed_hf_model_id` | 删除孤立 repo 常量复制 | 实际 load/helper 场景检查 processor/model 收到 registry 指定的 repo 和 revision；保留 generation、缺尾、speaker 等回归 |
| `test_qwen_adapter.py` 四个卸载场景 | 公共 setup 合成三行参数矩阵，正常关闭/CPU 转移合并 | 模型及 aligner 引用清空、关闭、CPU 转移先于 cache 释放、model close/CUDA synchronize 失败仍执行后续释放。删除辅助函数完整顺序快照 |
| `test_dependency_pins.py` 三项静态依赖检查 | 整体迁入发布检查，删除该测试文件 | `scripts.check_release_dependencies` 检查 Git full commit、模型/aligner snapshot、MCP optional/dev/lock metadata 约束一致及 locked 版本；CI 每次运行该命令 |
| `make_wav` 重复实现 | API、两个 audio 模块复用 `tests/helpers.py`；job 模块同样可直接引用 | 保留原采样率、声道及信号默认值；属于 fixture 去重，不算删除行为验证 |

上述五个原测试文件的顶层测试函数由 79 个变为 61 个，包含合并、改名和三项发布迁移。此数字不是 pytest 展开用例数；参数化保留或增加了真实边界，不能宣称减少了 18 次执行。真正删掉的低价值内容主要是重复 422、孤立常量断言和完整辅助调用顺序快照。

账号隔离、GPU 互斥、活跃卸载、worker 回收、删除意图、清理后终态、检查点身份、空稿证据、显式语言、MOSS 完整性、备份恢复等原行为保护继续保留。

## 媒体根因与本地观察

| 根因 | 新行为 | 已观察的本地证据 | 仍需的实机证据 |
| --- | --- | --- | --- |
| 窗口不前进或累计数无界 | 按输入采样整数推进；追加前共享 4096 上限；循环检查同一取消信号 | `test_audio_splitter.py`：显式 0、NaN/Infinity、近等 overlap 均受控拒绝；跨 VAD 区间累计上限和中途取消有限退出；原六小时 sparse WAV 仍只生成描述符 | 真实长音频识别质量另行评测 |
| parser 和上传副本资源遗漏 | parser 与 workspace 两份副本共同计费；lease 幂等归还 | `test_admission.py`：两入口未读完整超限正文、未进入 upload copy、parser 文件与账本清空；`test_workspace.py`：双副本耗尽额度及删除目录后重复释放 | 部署文件系统与并发负载观测 |
| 解码结束后才判断体积 | 原始 PCM stdout 固定块写 WAV，写入前检查实际输出额度，取消/deadline 回收 decoder | `test_audio_preprocess.py`：约 1 MiB 预算超限，删除前 partial 从未越界；同步包装同样拒绝；恰好达到上限成功；取消与超时回收且无 partial | 各目标 FFmpeg 构建、六小时资源和运行时吞吐 |
| 输入能否引用二级资源尚未验证 | probe/decode 共用 file 协议及普通媒体 demuxer 白名单，拒绝 playlist/script | Mac FFmpeg 8.1.1 和 WSL n8.1.2：10 种普通容器完成 probe+normalize；隔离 scratch 中 concat、本地 HLS、loopback URL HLS 均 probe 不接受、decode 失败；MOV 的 `enable_drefs` / `use_absolute_path` 帮助输出均为默认 false | 当前没有接受外部引用的复现，不宣称修复已确认 SSRF；其他目标构建需分别验证 |
| 重复规范化及不安全复用 | 内部 `NormalizedAudioAsset` 冻结规格、帧数、PCM SHA256、版本和 owner；借用方没有删除操作 | 资产 metadata roundtrip、PCM 篡改拒绝；上层集成验证首次只规范化一次和正常重试零次 | 升级前旧资产/检查点的明确迁移或拒绝记录 |

新的 `pcm16-wav-v2` 使用确定的 44 字节 WAV 头。旧 FFmpeg 容器头的文件摘要可能不同，不能仅凭相同 PCM 就混入旧检查点；缺少版本身份时应保留旧记录并要求从源素材新建任务。

## 发布检查与 CI 固定版本

2026-10-25 使用各官方仓库 `git ls-remote` 核实当前既有 major tag，CI 固定为对应 commit：

| Action | 上游 tag | 固定 commit |
| --- | --- | --- |
| `actions/checkout` | `v4` | `11d5960a326750d5838078e36cf38b85af677262` |
| `actions/setup-python` | `v5` | `a26af69be951a213d495a4c3e4e4022e16d87065` |
| `astral-sh/setup-uv` | `v6` | `d0d8abe699bfb85fec6de9f7adb5ae17292296ff` |

本地发布依赖检查结果：2 个 GPU Git commit pins、4 个模型及 aligner snapshots、MCP lock 为 `1.28.1`，optional/dev/lock metadata 的 `<2,>=1.27` 一致。此结果不代表在 Mac 安装或运行 GPU 包。
