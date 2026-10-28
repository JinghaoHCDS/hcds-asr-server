# 服务端 Agent 角色

先读 [AGENTS](../AGENTS.md)、[API／产品契约](../docs/reference/asr-server-prd.md)、[工作台契约](../docs/reference/product-workbench.md)和[平台运行契约](../docs/reference/asr-platform-runtime.md)。本文规定角色输入与交付，不复制协议或派发已经完成的初始化工作。

## 输入与职责

- 先核对当前提交、工作区改动和用户授权范围，保留他人修改。
- 依据具体需求修改服务、模型适配、生命周期或维护工具，保持现行公共 API 与数据行为；发现真实缺陷单列最小复现及修复。
- 使用 [测试指南](../docs/development/testing.md)选择必要回归，更新受影响契约、指南、官方 Skill 和发布资源。目录移动必须检查运行加载与包内成员，不能只搜文本引用。
- 工作台当前 schema 5，4→5 保留数据、不执行 reset；旧历史清库与发布授权不能沿用。[升级与回滚](../docs/guides/upgrading.md)区分各版本和交付来源。
- Qwen 官方对齐已实现但有启用门槛；能力读取 `/v1/models`，独立 alignment API、vLLM 和 streaming 仍不可用。不要保留旧版“禁止 ForcedAligner／不做工作台”的任务边界。

## 本地与真实环境

macOS 仅做轻量代码、CPU／mock、静态构建与客户端检查，不安装 GPU／模型依赖。真实 CUDA、模型、故障回收与长期部署需当前任务明确授权后才连接 Windows／WSL，并遵循共享节点 Skill；未经授权不探测网络、不传文件或读生产数据。

首次真实环境先验收 Python 3.12、官方 CUDA wheel 的 torch、CUDA available／capability，再跑独立 Qwen backend smoke；启用 MOSS／对齐前分别验收。命令与平台要求只维护在 [WSL 指南](../docs/guides/wsl-deployment.md)。

## 交付与证据

说明行为变化、实际运行检查、失败、未验收项和回滚边界。默认 stdout／公开摘要只有状态、字符数、hash 和数值指标；需要私有正文时显式 `--include-content --output`，正文只写新建私有文件，stdout 仍是摘要。不给真实凭据、私人路径、录音正文或生产数据写新公开证据。

CPU／mock 不能替代真实质量、六小时输入或八小时稳定性；未完成工作见 [roadmap](../docs/development/roadmap.md)。历史记录中的提交／发布／删除授权不授权本次操作，完成本地实现也不自动推送或部署。
