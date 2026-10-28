# 升级与回滚

本指南适用于当前 schema 5。部署、远程连接、数据重置和备份都必须在当前任务已授权的范围内执行；历史记录中的授权不是本次操作授权。真实部署前先在独立代码和数据目录验收。

## 版本适用范围

| 起点 | 当前处理边界 |
| --- | --- |
| schema 4 → 5 | 事务迁移，保留账户、任务、MP3、版本、检查点与保留期限；**不运行 `workbench_reset_tasks`** |
| 当前 schema 5 更新 | 核对源码／前端／依赖／流程身份；保持数据，检查点不匹配明确拒绝，用户可显式从头重试 |
| schema <4 | 历史 MP3 改造不提供旧 WAV／任务直接迁移。含任务或上传的旧库会返回 `schema_reset_required`；先制定与该版本匹配的数据处理方案。旧清库授权仅适用当次发布，不能自动沿用 |
| 新库 | 按当前版本初始化；不导入未经验证的旧任务或检查点 |

schema <4→4 的历史选择见[音频设计](../development/audio-storage.md)及[当时验收](../validation/2026-10-26-workbench-audio.md)。离线重置工具仍保留供明确授权的人工重置，但不是当前升级前置步骤。

## 安装来源与更新

- **Git checkout**：在 Linux 原生文件系统维护 checkout；确认工作区改动后更新源码，再按 [WSL 指南](wsl-deployment.md)安装匹配依赖。不能在更新中自动升级未变更的 GPU 环境。
- **解包发行目录**：它可能没有 `.git`，不要执行 `git pull`。在新的独立目录解包并核对 manifest，准备匹配前端及依赖后再切换。
- `frontend/dist/` 是运行包资源，不进入 Git；安装源码时先构建，使用运行包时校验包内资产。官方 Skill、运行时 Markdown、样例与锁文件都属于交付资源。

## 已授权的服务切换

1. 核对目标源码／前端／依赖／schema／流程身份和已有本地修改；在隔离数据目录完成相称验证。
2. 使用管理员 drain 停止新推理准入，等待活跃操作在安全边界退出；停止服务。读取、回听等降级可用性按[工作台契约](../reference/product-workbench.md#转录恢复与服务就绪)处理。
3. 在已批准的私有位置保存一致 SQLite、MP3、检查点、版本与配套旧源码／前端。备份包含账户密码哈希与用户数据，不能进入 Git、公开 Release 或公开 CI artifact。
4. 安装目标版本并迁移；4→5 不清任务。重启后核对 health、ready、登录、队列、音频 Range、完整／部分导出和容量，再恢复准入。
5. 记录包与逐文件摘要、实际验证范围、失败及未验收项，不记录私有正文或凭据。

备份工具取得离线 `service.lock`，拒绝覆盖既有备份；恢复目标必须是新目录或空目录。以下是离线工具形态，路径需替换为已批准的私有位置：

```bash
uv run python -m scripts.workbench_release backup --data-dir /path/to/workbench --output /path/to/new-private-backup
uv run python -m scripts.workbench_release restore --backup /path/to/new-private-backup --data-dir /path/to/new-restored-workbench
```

恢复时核对 manifest 的 `source.source_digest` 与配套源码／前端；commit 不能单独描述未提交改动和前端覆盖。工具只报告匹配状态，不自动重启或切换线上服务。

## 回滚边界

普通文档／路径治理可回滚对应源码和前端提交；不改数据库 schema。已实际迁移的数据不能只回退代码：旧程序不认识 schema 5 时必须拒绝运行，不能绕过守卫。

需要回到旧数据格式时先停写并保存新版本运行期间产生的数据，再恢复匹配的源码、静态资源和数据快照。不能用旧备份静默覆盖新增上传、编辑与结果，也不承诺历史上已删除的临时备份仍存在。由 manifest 确认过期受管文件后再清理，不对部署目录直接使用无边界删除。

尚未实际执行的生产回滚只能标为方案或隔离演练。历史[console 验收](../validation/2026-10-27-workbench-console.md)也没有证明生产回滚已执行。
