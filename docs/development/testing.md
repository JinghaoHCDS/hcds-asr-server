# 测试与验证

## 当前检查入口

本地使用 CPU／mock 和独立开发数据目录；不设置真实 HTTP smoke 的 `ASR_BASE_URL`。先冻结依赖，再执行与改动相称的测试，收尾跑一次全量。源码支持、mock、真实 GPU 与生产实际操作分别报告。

```bash
uv sync --frozen
uv run --frozen python -m scripts.check_release_dependencies
uv run --frozen pytest -q
uv run --frozen mypy asr_server asr_mcp tests scripts
npm --prefix frontend ci
npm --prefix frontend run build
```

`build` 已包含 TypeScript 检查，无需再重复 typecheck。当前树／PR diff、Git 与忽略冲突、Markdown 与图片、发行包和官方 Skill 的检查属于交付验证。运行时 Markdown 有缺文件降级，因此 HTTP 200 之外还要核对内容与包成员。原始验收附件已从当前树删除；包内仅收录维护的摘要、许可明确的合成样例与正式资源，仍须检查引用闭包。

## 2026-10-25 测试合并与替代证据

以下是当时版本的去向记录，不是当前 HEAD 已通过或提速证明。日期：2026-10-25。基线：`2fb6a4dca0375ad2cbc3a20b2e3aa5b4b71cf852`。本表记录本轮实际删改和替代证据；整轮的用例数、耗时与实机结果以最终验收记录为准。

## 保留的默认检查

```bash
uv run --frozen pytest -q --durations=20
uv run --frozen mypy asr_server asr_mcp tests scripts
uv run --frozen python -m scripts.check_release_dependencies
```

`integration`、`gpu`、`performance`、`release` 已登记为 pytest markers。没有通过 `addopts` 排除任何现有本地行为测试；`test_api.py` 标为 integration 后仍在默认套件内运行。已有真实 HTTP smoke 仍须显式设置 `ASR_BASE_URL`。GPU backend smoke、批次性能测量和依赖发布检查继续使用明确命令，不能把新增标签解释为已获得测试提速。

独立 GPU 验收入口为 `scripts.platform_gpu_validation`（真实 API/工作台/生命周期/安全恢复）、`scripts.platform_gpu_compare`（同源工作负载的基线和日志对照）和 `scripts.platform_prefetch_probe`（同序列预取开关）。只在已有合格环境的独立目录执行；具体命令与证据见 [WSL 验收](../validation/2026-10-25-wsl-platform.md)。独立 ForcedAligner smoke 的 `zh`/`auto` prefill 回归位于 `test_forced_alignment_smoke.py`，不要求安装 GPU 依赖。

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

## 工具输出和真实验证

`scripts/` 保持相互导入和仓库根推导位置。`platform_resource_probe`、`platform_gateway_memory_probe` 等原始测量默认进入忽略的 `artifacts/platform/`，正文与请求快照按私有数据处理，不直接提交测量目录。

模型／工作台 smoke 默认 stdout 为状态、字符数、SHA-256 和数值指标；需要正文时显式 `--include-content --output`，仅写新建私有文件，stdout 不变。既有内部运行函数仍可供私有比较工具取得完整结果，不能把其返回值直接当公开报告。

## 仓库与发行资源检查

```bash
uv run --frozen python -m scripts.check_repository
uv run --frozen python -m scripts.check_secrets --gitleaks /path/to/gitleaks
uv run --frozen python -m scripts.check_secrets --gitleaks /path/to/gitleaks --ref HEAD
```

凭据扫描固定 Gitleaks 8.30.1，下载版本与校验摘要见 [CI](../../.github/workflows/ci.yml)。此命令要求目标是自身带 `.git` 的 checkout；无 Git 发行目录不能继承祖先仓库并把空扫描当通过。默认只读取 Git 跟踪内容、可见新文件及工作区增删 diff，不读取忽略的运行数据；`--ref` 可重复指定需要检查的本地引用，只检查这些提交可达的历史。CI 获取完整提交历史并检查 HEAD；发布前还需显式检查拟发布的其他分支与标签，不扫描 Agent 私有辅助引用。

扫描输出仅含规则和位置，临时报告使用同样的字段投影并在退出时删除，不保存原始命中、作者或消息。历史源码 manifest 曾包含 `asr_server/auth.py` 的 SHA-256 行；相关扫描例外只用于该摘要格式，不豁免实际凭据字段。模式未命中不等于隐私或素材公开授权。

准入检查覆盖生成目录、跟踪／忽略冲突、超过 1 MiB 的新增或改动文件、5 MiB 上限及本地 Markdown／图片／锚点。音频样例的来源、生成方法、规格和摘要以[样例说明](../../test-fixtures/audio/README.md)为准；更新时同时复核资源准入与 smoke 身份，不自动沿用旧样例的真实 GPU 验收。检查器不是完整 CommonMark 渲染器，不验证外链或全部动态 JavaScript 资源。

`workbench_release bundle` 会在生成归档前检查所收文档的资源闭包；原始验收材料不收进新包，历史记录只保留有版本和限制的精选摘要；其附件已按维护决定移除，不能把摘要改写成本轮复测。解包后可用 `check_repository --package-root /path/to/extracted-release` 复核 manifest、关键资源、前端引用及本地文档；目录内只放待检查发行内容。

恢复工具 `scripts/transcription_resilience_smoke.py` 覆盖部分结果、Range、排空、恢复和清理，使用隔离数据目录。`scripts/workbench_audio_smoke.py` 区分 mock 工程行为与真实 MP3／模型路径；GPU 工具只在当前任务授权后连接 WSL，不能在 Mac 安装模型来补验收。

[验收模板](validation-template.md)规定证据粒度；未关闭项见 [roadmap](roadmap.md)。
