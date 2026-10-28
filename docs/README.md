# 文档索引

使用者从 [README](../README.md)开始；开发者与 Agent 先读 [AGENTS](../AGENTS.md)。运行时模型和能力以 `GET /v1/models` 为准，精确接口 schema 读取 `/openapi.json`；文档不能替代真实环境验收。

## 使用指南

- [WSL 部署](guides/wsl-deployment.md)：支持边界、CUDA 门槛、启动与排障。
- [升级与回滚](guides/upgrading.md)：checkout／发行目录更新，当前 schema 4→5 保留数据及更早版本边界。
- [MCP bridge](guides/mcp-bridge.md)：客户端 stdio、允许目录与工具。
- [口播、播客与批量交接](guides/batch-workflow.md)：逐素材 Context、恢复和正式结果目录。
- [日志与离线诊断](guides/asr-diagnostics.md)：默认不含正文的关联事件和诊断包。
- [官方 hcds-lan-asr Skill](../skills/hcds-lan-asr/SKILL.md)：部署后先读 `/skill`，ZIP 下载同源客户端并动态带入服务地址。

## 当前参考

- [API／产品契约](reference/asr-server-prd.md)：模型能力、生命周期、参数、错误和内存 job。
- [工作台契约](reference/product-workbench.md)：账户隔离、schema 5、音频／容量、恢复、编辑与导出。
- [平台运行契约](reference/asr-platform-runtime.md)：资源准入、FIFO、资产、持久化和就绪。
- [OpenAPI 说明资源](reference/docs-endpoint-capabilities.md)：服务实际读取的 `/docs` 描述；[MCP 指南](guides/mcp-bridge.md)由 `/docs/mcp` 读取。

## 开发与维护

- [测试指南](development/testing.md)：CPU／mock、GPU、发布检查及既有测试合并／替代证据。
- [平台设计](development/platform-design.md)：资源、调度、I/O、注册与诊断的理由。
- [音频与容量设计](development/audio-storage.md)：持久 MP3、操作 PCM、互斥账本和历史迁移背景。
- [转录恢复设计](development/transcription-recovery.md)：有界策略、稀疏结果、fence 与自愈。
- [模型实践](development/asr-model-practices.md)：固定上游实现与选用依据。
- [新增引擎](development/adding-asr-engine.md)：stub、注册和真实启用门槛。
- [质量评测](../evaluation/README.md)：独立开发／验收集、CER、术语和字幕。
- [验收模板](development/validation-template.md)：版本、方法、指标、限制及脱敏证据要求。
- [未完成事项](development/roadmap.md)：质量、持续运行、平台及历史隐私整改的具体关闭条件。

## 历史验收

以下记录绑定各自原始版本和环境，不能视为当前 HEAD 的测试证明。文档整理所用源码基线为 `0d329c4c621408f4f49fc4b6b426da3ada61ad0a`，它只标识记录来源，不替代记录中的真实运行 ref／文件摘要。已有未提交差异无法补齐时保持不确定。

原始验收日志、JSON、截图和一次性验证脚本已按维护决定从当前源码树删除，不另做私有副本。精选记录保留方法、指标、失败与限制，并标明其原版本；删除不清除 Git 历史，也不代表已收回外部副本。仓库只分发许可明确的合成样例，自有代码使用 [MIT](../LICENSE)，外部组件与音频生成器见[第三方归属及许可](../THIRD_PARTY_NOTICES.md)。

- [2026-10-27-workbench-console](validation/2026-10-27-workbench-console.md)
- [2026-10-26-workbench-audio](validation/2026-10-26-workbench-audio.md)
- [2026-10-26-transcription-resilience](validation/2026-10-26-transcription-resilience.md)
- [2026-10-25-wsl-platform](validation/2026-10-25-wsl-platform.md)
- [2026-10-25-workbench-ui](validation/2026-10-25-workbench-ui.md)
- [2026-10-25-skill-address](validation/2026-10-25-skill-address.md)
- [2026-10-25-review-fixes](validation/2026-10-25-review-fixes.md)
- [2026-10-25-platform](validation/2026-10-25-platform.md)
- [2026-10-25-hcds-rename](validation/2026-10-25-hcds-rename.md)
- [2026-10-25-agent-skill](validation/2026-10-25-agent-skill.md)
- [2026-10-22-quality-workbench](validation/2026-10-22-quality-workbench.md)
- [2026-10-22-local-workbench](validation/2026-10-22-local-workbench.md)
- [2026-10-10-repository-cleanup](validation/2026-10-10-repository-cleanup.md)
- [2026-09-19-upgrade](validation/2026-09-19-upgrade.md)
- [2026-07-15-wsl-moss-long-form](validation/2026-07-15-wsl-moss-long-form.md)
- [2026-07-15-wsl-moss-anchor-replay](validation/2026-07-15-wsl-moss-anchor-replay.md)
- [2026-07-15-live-chunks-four-speaker-podcast](validation/2026-07-15-live-chunks-four-speaker-podcast.md)
- [2026-07-10-wsl-moss](validation/2026-07-10-wsl-moss.md)
- [2026-07-10-stability-refactor](validation/2026-07-10-stability-refactor.md)
- [2026-07-02-wsl-hf-native](validation/2026-07-02-wsl-hf-native.md)

## 维护约定

当前规则只在对应 reference 与 AGENTS 维护，其他位置使用链接和必要摘要。指南更新随行为变更；设计说明保留取舍，不保留已完成派发清单。历史验收保留失败与限制，新增证据另建绑定版本摘要，不覆盖旧记录。

移动／删除前检查源码、测试、CI、动态资源、部署和产品包。生成报告写入忽略的 `artifacts/` 或批准的私有位置；只有精选且获准公开的摘要进入 `validation/`。不创建持续增长的公开原始 archive。
