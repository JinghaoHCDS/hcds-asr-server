# 2026-10-10 仓库整理

> 历史摘要：以下结果属于原日期、环境和版本，本轮未重新验证。原始验收附件已按维护决定从当前树删除，未另存副本；Git 历史未改写。

基于 `d36ba7d` 的产品化整理。先清点引用，再删除明确脱离现有运行路径的文件；历史实机记录保留。

## 已清理

| 文件 | 原因 / 替代入口 |
| --- | --- |
| `environment.yml` | 无代码、测试或文档引用的旧 conda 配置；统一使用 Python 3.12 + uv |
| `scripts/batch_transcribe_qwen.py` | 无入站引用的旧直连 GPU 批处理；由 `scripts/batch_transcribe.py` 的 HTTP 清单工作流替代 |
| `docs/moss-long-form-podcast-optimization-handoff.md` | 已完成的临时交接计划，包含私人素材路径；历史可从 Git 追溯 |
| `.DS_Store` | Finder 生成的忽略文件 |

进入任务时 `test-fixtures/audio/test_long.mp3` 已被本地删除，保留该状态。测试改用在内存中生成的 210 秒 tone / silence MP3，继续验证实际 FFmpeg 解码和 energy 分块，不再依赖这份长媒体。新用户的长音频验收使用自备样本。

## 保留及整理

- `scripts/export_transcript_artifacts.py`：仍有测试引用，保留兼容导出路径。
- 独立 backend smoke、forced alignment smoke、完整 Qwen pipeline smoke：验证层次不同，全部保留。
- `docs/validation-*.md` 和稳定性审查：保留真实成功、失败与限制记录；文档索引补充四人播客记录。
- `examples/output-sample/`：保留明确标记的合成样例，展示 Agent 文件夹交接契约。
- `prompts/`：保留代理交接入口，改为用户配置的路径和 LAN 地址。
- `CLAUDE.md`：仍为指向 `AGENTS.md` 的兼容符号链接。
- `.venv/`：保留开发环境；迁移目录留下的旧 shebang 已通过按锁文件重装修复。没有安装 torch、CUDA 或模型依赖。

当前不跟踪运行日志、上传、模型缓存和转录产物。测试与类型检查生成的缓存属于忽略文件，验收收尾时清理。

## 产品入口修正

- 默认部署当前 WSL checkout，systemd unit 渲染实际路径；Windows 启动脚本读取用户的发行版、用户与目录参数。
- 默认允许首次模型下载，离线模式需显式选择并预热所有要用的模型。
- 服务器与客户端默认地址改为 loopback；远端调用显式配置 Windows LAN IP。
- 补齐 PRD 已约定的可选 Bearer 验证，更新客户端 Token 和 Swagger Authorize。
- PRD 和代理提示统一任务创建响应 `id`、compact / full 轮询，移除未实现的单请求文件数组能力描述。
- 全套回归发现 job 终态先于临时音频清理、取消清理时 spool 预算泄漏；修复清理完成后才发布终态并开始结果 TTL。

本轮本地验证覆盖 mock/API、部署模板与命令、鉴权、清理竞态、类型和静态 README 渲染。没有在 Mac 安装 GPU / 模型依赖，也没有替换在线 WSL 服务；Windows 启动与目标 WSL GPU 验收仍需对应机器运行。

最终检查：`pytest -q` 为 212 passed、2 skipped（需显式在线服务配置的 HTTP smoke）；`mypy asr_server asr_mcp tests scripts` 为 66 个源文件通过；bash 语法与 `git diff --check` 通过。另启动实际 loopback Uvicorn mock 服务，验证公开 health/docs、401 / Bearer、同步文件上传、异步 `id` 与 compact / full 结果，随后停止该进程。README 静态 SVG、相对文档链接与 900 / 360 px 布局检查通过。该结果不等于 WSL GPU 或 Windows 启动验收。
