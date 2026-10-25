# 文档索引

网页使用者从 [README](../README.md#使用工作台) 开始；部署者从 README 和 WSL 指南开始；Agent 接入者从 MCP / 批量工作流开始。模型和能力始终以运行中的 `GET /v1/models` 为准。

## 使用与维护

- [局域网工作台与 Qwen 质量流程](product-workbench.md)：账号、断点上传、七天任务保留、回听改字与字幕导出。
- [Qwen 质量评测](../evaluation/README.md)：开发/验收集、切分候选对照、CER/术语/字幕门槛。
- [WSL 部署](wsl-deployment.md)：Windows / NVIDIA / WSL 前提、首次 GPU 验收、systemd、Windows 启动与局域网访问。
- [MCP bridge](mcp-bridge.md)：客户端 stdio 配置、音频允许目录和八个工具。
- [口播、播客与批量交接](batch-workflow.md)：逐素材 Context、恢复 / 重试、JSON + Markdown 输出。
- [API 合约](asr-server-prd.md)：模型能力、生命周期、异步 job、错误信封与鉴权。
- [Swagger 说明](docs-endpoint-capabilities.md)：`/docs` 与当前 API 能力说明文本。
- [验收模板](validation-template.md)：真实 WSL smoke 的记录要求。
- [仓库清理记录](repository-cleanup.md)：产品化整理中删除、替代和保留的文件。
- [性能与多 ASR 改进计划](asr-platform-improvement-plan.md)：资源边界、调试日志、性能、模型接入与测试精简的设计背景；实现行为见[平台运行契约](asr-platform-runtime.md)，检查结果见[平台验收](validation-2026-10-25-platform.md)。[Agent 任务包](asr-platform-agent-tasks.md)保留实施职责记录。

## 验收记录

这些是对应版本和环境的历史证据，不代表所有硬件、发行版、模型或音频质量都已验证。

- [2026-10-25 平台改进](validation-2026-10-25-platform.md)：A–E 实现、本地控制面性能、六小时资源、浏览器、迁移和回滚演练。
- [2026-10-25 WSL 平台实机](validation-2026-10-25-wsl-platform.md)：真实 Qwen/MOSS、官方对齐、十分钟工作台、检查点恢复及 GPU 对比。
- [2026-10-22 工作台本地工程验收](validation-2026-10-22-local-workbench.md)：mock 回归、浏览器流程、账号隔离、备份恢复与未验收项。
- [2026-10-22 Qwen 质量流程与工作台实机验收](validation-2026-10-22-quality-workbench.md)：13.06 秒样本的真实转录、对齐、别名规范、编辑与字幕导出，不代表长音频质量达标。
- [2026-07-02 Qwen HF native](validation-2026-07-02-wsl-hf-native.md)：0.6B / 1.7B 的 transformers 后端。
- [2026-07-10 MOSS](validation-2026-07-10-wsl-moss.md)：独立 smoke、HTTP 路径与卸载。
- [2026-07-10 稳定性审查](review-2026-07-10-stability-refactor.md)：单 GPU、worker、临时目录和 job 可靠性。
- [2026-07-15 MOSS long-form](validation-2026-07-15-wsl-moss-long-form.md)：10 / 30 / 60 / 90 分钟、缺尾保护与自动分块。
- [2026-07-15 Anchor Replay](validation-2026-07-15-wsl-moss-anchor-replay.md)：四说话人跨块解析与对抗性检查。
- [2026-07-15 四人播客与实时 chunk](validation-2026-07-15-live-chunks-four-speaker-podcast.md)：真实长素材的 partial / mixed 边界与待复验项。
- [2026-09-19 升级](validation-2026-09-19-upgrade.md)：Qwen 对齐、Context、批量交接及长播客限制。

## 验证工具

在 WSL 内运行真实 GPU 工具；无 GPU 环境只运行 mock / unit checks。

- [Qwen 独立 backend smoke](../scripts/qwen_asr_backend_smoke.py)
- [Qwen 官方对齐器 smoke](../scripts/qwen_forced_alignment_smoke.py)
- [Qwen 完整音频 / Context / 对齐 pipeline smoke](../scripts/qwen_pipeline_smoke.py)
- [MOSS 独立 backend smoke](../scripts/moss_backend_smoke.py)
- [真实 HTTP smoke](../tests/test_http_smoke.py)：配置 `ASR_BASE_URL`；自备音频用 `ASR_HTTP_SMOKE_AUDIO`，Token 用 `ASR_HTTP_SMOKE_TOKEN`。

## 维护约定

新增或关闭能力时，同步更新 README、部署指南、Swagger 说明和代理提示词。真实验收另存新的 `validation-YYYY-MM-DD-*.md`，不覆盖旧记录。已完成的临时计划通过 Git 历史追溯，不继续作为现行交接指令。

- [平台运行契约](asr-platform-runtime.md)
- [平台实现验收](validation-2026-10-25-platform.md)
- [日志与诊断](asr-diagnostics.md)
- [新增 ASR 引擎](adding-asr-engine.md)
- [测试精简去向](asr-platform-test-map.md)
