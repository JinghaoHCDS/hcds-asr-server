# ASR 关联事件与离线诊断

应用事件使用 JSONL。`trace_id` 串起一次任务，`request_id` 标识提交请求，`job_id` 是服务任务，`rpc_request_id` 是进程内递增的 worker 调用号。后台 owner 必须显式保存并重新绑定 `ExecutionContext`；仅依靠请求的 ContextVar 不能保持 FIFO 任务关联。

`ASR_LOG_LEVEL=INFO`、`ASR_LOG_FORMAT=json`；设置 `ASR_LOG_DIR` 时由网关单写者输出 `events.jsonl`，按实际 UTF-8 字节最多 10 MiB、至少每天轮转，保留当前文件及 9 个轮转文件，总量最多 100 MiB。写入线程空闲时每分钟维护一次，按文件首条事件时间整体过期并提前两分钟清理，保留时间不超过七天；可能整批提前删除同文件内较新的事件。维护仅查看这十个固定文件。目录权限 0700、文件 0600。服务关闭期间不运行维护，下次启动立即清理。未设置目录时输出到 stderr，由运行环境管理保留期。应用配置不控制 journald 的保留。

日志队列默认 4096 条，给关键状态及失败预留槽位。饱和时丢弃事件，`log_stats()` 及管理员 `/api/admin/status.logging` 返回丢失总数与队列长度，下一条成功入队的事件携带 `log_queue_dropped_total`。持续饱和时失败事件也可能丢失；业务和 GPU 不等待日志写盘。单行上限 8 KiB，未知异常仅保存异常类型和有限栈位置，无 message、源码行、locals 或完整路径。默认不记录转录、热词、密码、Cookie、Token 或音频。

在项目根目录运行：

```bash
uv run python -m scripts.asr_diagnostics trace --job job_example --log-dir /path/to/logs
uv run python -m scripts.asr_diagnostics trace --job wb_example --log-dir /path/to/logs --format text
uv run python -m scripts.asr_diagnostics bundle --job wb_example --log-dir /path/to/logs --data-dir /path/to/workbench --output /path/to/new-diagnostics.tar.gz
```

`trace` 显示关联时间线、最后成功阶段和 worker 错误链。每遍最多读取十个日志文件、100 MiB，总时间线内容最多 8 MiB；截断在结果中标记。`bundle` 上限 16 MiB，包含过滤后的日志、任务参数摘要、运行时版本和检查点文件清单，不包含正文、音频、数据库、环境变量或凭据。仅用 SQLite 只读连接投影任务摘要。工具不启动模型、不重试、不连接远程节点；输出必须是新文件。旧 `/v1` 任务重启后可能只剩日志，不能由日志恢复音频。

worker 的 `diagnostics` 是 IPC 内部字段，父进程重新按白名单过滤后写事件，不并入客户端 `AsrError.details`。协议版本为 1；生产 worker 在 ping 时声明版本。历史自定义 worker 缺失版本时仍兼容，显式不匹配则回收并拒绝继续。
