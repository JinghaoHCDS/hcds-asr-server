# 2026-10-28 仓库治理与 WSL 发布

状态：本轮治理已推送 `main`，既有 WSL 服务于 2026-10-28 14:05:53（UTC+08:00）完成切换，短合成样例的真实工作台流程通过。schema 保持 5，未迁移或清空任务，GPU 环境、systemd unit 与已有配置保持不变。

## 版本与交付

- 运行源码：`027057f040acbc1c3e83a83d7e4c6f084244a32c`；治理主体为 `8c20c29`，后续提交修正音频测试对文件系统时钟的依赖。
- 运行包：`hcds-asr-server-20261028-027057f.tar.gz`，238 个文件。
- 包 SHA-256：`94d47eca6995c15402f50671edbe0a7a9536d3e35902b69e6e055ed0ebe22ef3`。
- 源码摘要：`36ed41faa148cee82304649b19e83ecfefb4069b35c14ae6d27e6a834a234511`；以逐文件 manifest 核对实际交付内容。

文档按指南、当前参考、开发维护和历史验收整理；错误及验收输出默认不携带输入正文或凭据。打包和部署排除运行产物，补齐资源闭包、仓库准入及凭据检查。自有代码采用 MIT，第三方许可另行保留；原始验收附件和旧录音从当前源码树移除，替换为来源明确的合成样例。线上按已知基线精确清理 78 个过期受管文档和附件，未使用无边界目录删除。

## 工程验证

| 范围 | 实际结果 |
| --- | --- |
| macOS | 569 passed、4 skipped；mypy 139 文件通过；前端构建、干净 CPU 依赖安装和隔离 mock 流程通过 |
| WSL 回归 | 首轮 570 passed、2 skipped、1 failed；失败是既有 PCM 篡改测试依赖同一时钟刻度内时间戳自动变化。显式推进 mtime 一秒后，音频模块 19 项通过；mypy 139 文件通过。未将两次结果写成一轮完整绿灯 |
| 发行目录 | 包摘要、逐文件资源、Markdown 链接、依赖约束及隔离 mock 上传／转录／导出／回听／编辑／删除通过 |
| LAN 入口 | 网页与静态资源字节匹配运行包；`/health`、`/readyz`、`/v1/models`、`/`、`/docs`、`/docs/mcp`、`/openapi.json`、`/skill`、客户端脚本和 Skill ZIP 均为 200；ZIP 按实际下载 origin 生成服务地址 |

GitHub CI 原为 `disabled_manually`，本轮已重新启用；各提交结果见 [Actions](https://github.com/JinghaoHCDS/hcds-asr-server/actions)。

## 短样例真实 GPU 流程

使用[合成样例](../../test-fixtures/audio/README.md)，10.105375 秒，SHA-256 为 `56717d7e62e209e426d939d33334643cc682fac6b0bbd1f436cea090b37e5ad1`。环境为 WSL2 Arch Linux、RTX 5070 Ti，torch `2.11.0+cu128`、CUDA `12.8`、capability `12,0`，保留已有安装。

- 通过真实 HTTP 工作台使用 Qwen3-ASR 1.7B、`quality_v1` 与官方 ForcedAligner，任务完成；本次端到端 18.135 秒包含加载等开销，不是稳定性能基准。
- 参考和识别文本按小写单词及标点归一化后均为 28 词，编辑距离 0、WER 0；结果 160 字符、2 个片段、28 个真实时间戳，状态 `aligned`。
- TXT／JSON／SRT／VTT 导出均为 200；MP3 Range 回听为 206；修改合成任务文字后，版本 2 重新对齐为 `aligned`，字幕可导出。
- 会话权限、参数报错不回显密码及容量接口检查通过。测试使用临时维护会话，token 未保存；合成测试任务已删除。

这些结果只验证此短合成样例的工程流程，不代表时间戳人工准确率、听感或独立质量集通过；未重验 MOSS、真实六小时质量、八小时稳定性及其他客户端平台。

## 数据与回滚

原账号、5 个历史任务、正文版本、MP3 与检查点按持久数据摘要核对保留；除 SQLite 外，1102 份原有数据文件逐一 SHA-256 一致。启动刷新的 `jobs.revision`、数据库自增序号及本次测试正常分配 FIFO 后前进 2 的 `metadata.recovery_submission_sequence` 属于运行计数，不当作原内容变化；正文、编辑版本与归属仍严格核对。

首次切换把该刷新计数也纳入严格比较，触发保护性运行代码回退；旧服务的 health／ready 恢复，未恢复或覆盖工作台数据。调整验收比较后再次切换成功。这是当前 schema 5 的实际运行代码回退，不能推及跨 schema 回滚或新增数据恢复。

本轮私有离线备份已核验 1103 个数据文件和 126 个运行文件，保存在 WSL 的 `~/services/backups/repository-governance-20261028-027057f`；匹配包、清单与操作脚本保存在 `~/services/releases/repository-governance-20261028-027057f`。独立 `rollback.py` 已部署并通过只读输入校验，未执行；实际执行过的是上述激活脚本自动回退路径。备份不进入 Git、公开 Release 或 CI artifact；没有为本次回滚备份原始公开验收附件或旧 WAV。Git 历史和外部旧副本未改写或收回。
