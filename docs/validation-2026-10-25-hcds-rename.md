# 2026-10-25 仓库与官方 Skill 改名

GitHub 仓库从 `JinghaoHCDS/asr_Server` 改为 `JinghaoHCDS/hcds-asr-server`，仓库 ID 保持一致。本地 Git origin 和当前部署、接入文档的仓库链接同步更新。旧验收记录保留其当时名称和路径。

官方 Skill 改为 `hcds-lan-asr`，版本 1.4.0，来源移动到 `skills/hcds-lan-asr/`。下载端点为 `/skill/hcds-lan-asr.zip`，包内根目录与 SKILL.md 的 name 都使用新名称；旧 `/skill/lan-asr.zip` 以 307 跳转到新入口。`/skill`、部署地址自动注入、server.json、客户端 API 和鉴权语义继续使用原约定。

为保持恢复能力，客户端的 `~/.Agent/state/lan-asr/` 数据目录与提交幂等命名空间不随展示名称改变。Mac 上的安装应从实际服务下载新包后替换旧 Skill 目录，并修正依赖工作流的旧脚本路径，不重建批次数据库。

## 本地验证

- 完整回归：**422 passed / 4 skipped，24.77 秒**。
- mypy：**119 文件**通过；Skill 验证器通过。
- 文档、客户端、发行包和 MCP 相关检查：先行 **38 项**通过。
- 同一检查覆盖新包目录、Skill name、旧 URL 跳转、生成的部署地址、旧本机状态兼容，以及服务发布包不混入旧 Skill。

本轮代码不变更模型实现、GPU 依赖、工作台数据库 schema、账号或任务保留。已有服务器的工作目录与 systemd unit 名称保持现有配置；新部署示例使用新仓库目录名称。此文件记录本地工程验证，实际发布和 Mac 安装结果需以随后线上检查为准。
