# 2026-10-25 官方 LAN ASR Skill 与文档分发验收

> 历史摘要：以下结果属于原日期、环境和版本，本轮未重新验证。原始验收附件已按维护决定从当前树删除，未另存副本；Git 历史未改写。

本轮沿用现有 `lan-asr`，将其更新为 1.2.0，纳入 `skills/lan-asr/` 并同步本机共享安装。服务入口新增 `/skill`、`/skill/lan-asr.zip` 和 `/skill/asr_batch.py`。本记录验证本地工程行为；截至本轮本地验收结束，没有使用 Windows 节点、部署线上或执行真实模型推理。

## 交付行为

- `/skill` 以 `text/markdown` 返回仓库中的 `SKILL.md`。ZIP 仅含 `lan-asr/SKILL.md` 与 `lan-asr/scripts/asr_batch.py`，客户端另提供固定源文件下载；不允许任意路径下载。缺少分发文件时返回 `503 skill_unavailable`，不提供伪完整包。
- `/docs` 初始 HTML 就有 `/skill` 链接，无需执行 Swagger JavaScript。Swagger 和 OAuth redirect 继续可用；OpenAPI 描述、README、文档索引、MCP 教程与请求端提示词提供入口。文档保持公开，`/v1/**` 原有 Bearer 验证不变。
- Skill 以当前运行时能力发现为准，说明统一 FIFO、Qwen quality_v1、MOSS 段级时间及匿名标签、生成截断和质量未验收项。明确区分 `/v1` 内存任务与 `/api` 个人持久任务；修正旧 `/docs` 中不支持工作台或数据库的过时表述。
- 批量客户端增加 `--quality-profile` 与 `--max-new-tokens`，随请求快照持久化并按原值恢复。声明不足时提前拒绝；显式模型和要求的能力不静默替换。已有数据库采用加法迁移。
- 新批次冻结源文件 SHA256；丢失任务重提前重新校验输入和能力，最多重提一次。旧批次缺少原始 SHA256，只能先比较已保存大小和修改时间再建立摘要，不能追溯检测历史上保持二者不变的修改。
- 每份结果文件先写同目录临时文件，再原子替换；两份输出成功后才写入本地 completed。磁盘写入失败或取消不作为成功批次。
- 地址优先级为 `--base-url`、`ASR_BASE_URL`、本机 `~/.Agent/state/lan-asr/config.json`、loopback 默认地址。配置只用于新批次，恢复使用已冻结服务器。公开文件不包含个人 IP；本机通过独立配置保留原服务地址。
- `scripts.workbench_release` 的发行来源包含官方 Skill，避免仅在开发 checkout 中可用。

## 检查结果

| 检查 | 结果 |
| --- | --- |
| 完整本地回归 | 414 passed、4 skipped，25.55 秒 |
| 静态类型 | mypy 119 个文件通过 |
| 发布依赖检查 | 通过；未新增服务依赖或更改 GPU pins |
| Skill 验证器 | 官方目录通过；兼容的 YAML frontmatter，版本位于 metadata |
| 客户端行为检查 | 19 项通过，覆盖旧数据库迁移、请求快照、能力拒绝、丢失任务、输入变化、输出失败、取消及本地配置优先级 |
| 真实 HTTP + mock adapter | 从 `/docs` 发现 `/skill`，下载 ZIP，运行下载的客户端；Qwen 普通转录和 MOSS 播客均 completed，TXT 与 JSON 正文一致，保留 mock 标记 |
| 本机安装同步 | `SKILL.md` 与客户端逐字节匹配仓库；CLI help 含新参数；没有打开既有批次数据库 |

默认四项跳过沿用现有条件：两项未配置真实服务地址的 HTTP smoke，以及两项缺少可选 Jinja2 的语言模板用例。本轮未修改模型模板，不扩大这两类验收。

本机原 Skill 和脚本备份保存在 Git 忽略的 `data/skill-upgrade/`。临时 mock 服务、测试媒体、下载包和演练批次数据库已随临时目录清理；本机原批次状态未参与测试。

## 复现

```bash
uv run --locked pytest -q
uv run --locked mypy asr_server asr_mcp tests scripts
uv run --locked python -m scripts.check_release_dependencies
python3 skills/lan-asr/scripts/asr_batch.py run --help
```

发布后应从实际服务验证 `/docs` 的原始 HTML、`/openapi.json` 的 Skill 入口、三项分发端点及 ZIP 内容与发布源的一致性，同时确认原 `/v1` 鉴权和模型状态。当前本地成功不代表现有线上服务已经包含这些路由。
