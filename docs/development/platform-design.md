# 平台执行与资源设计

本文提炼 2026-10-24 规划、2026-10-25 A–E 实现的长期设计理由，替代已完成任务包。当前行为以[平台运行契约](../reference/asr-platform-runtime.md)为准；原始基线为 `2fb6a4dca0375ad2cbc3a20b2e3aa5b4b71cf852`，测量和失败边界见[平台验收](../validation/2026-10-25-platform.md)及[WSL 验收](../validation/2026-10-25-wsl-platform.md)。目标值不是已取得结果。

## 边界与取舍

- 保持 FastAPI、SQLite、独立模型进程和单 GPU 串行，不引入 DAG 引擎、事件总线、ORM 或分布式队列。
- 所有推理入口共享 FIFO，解决同步路径绕过队列的问题；模型互斥与公平提交是两个不同不变量。卸载保留活跃请求保护，不能等整批任务完成后才登记卸载意图。
- 请求关联、资源 lease、音频 owner、调度票据和引擎注册各承担窄职责；后台 owner 显式携带关联上下文，不能依赖 ContextVar 自动跨队列传播。

## 有界媒体与资源所有权

正文准入在 Form／JSON 解析前执行，Content-Length 只是提前拒绝提示，实际按 receive 字节累计。parser spool 与上传副本同时存在时各计峰值；异常、断连及取消都释放文件和 lease。

音频窗口按整数采样推进，拒绝非有限数和无法前进的 overlap；多个 VAD 区间共享追加前的 4096 块上限。所有扫描与窗口循环观察同一取消事件，不能先生成无界列表再检查。

解码按固定块从 FFmpeg stdout 取得 PCM，在落盘前检查实际输出预算、时长、deadline 和取消；stderr 有界消费。恰好达到上限还需检查 EOF，不能把截断输入当完整成功。Qwen 16 kHz 单声道 PCM16 每秒为 32000 字节，六小时约 659 MiB；这不是其他引擎强制规格，上传源和其他存活副本另计。

内部 `NormalizedAudioAsset` 冻结规格、帧数、PCM 摘要、版本和 owner；下游只借用。路径和 inode 只用于操作内核验，不作跨操作检查点身份。工作台持久 MP3 与临时 PCM 的分工见[音频设计](audio-storage.md)。

probe／decode 共用输入协议和普通媒体 demuxer 策略。实际 FFmpeg 构建需要分别验证二级引用行为；协议白名单不等于文件系统沙箱，历史没有接受外部引用的复现不能被写成已确认 SSRF。

## I/O、预取和模型扩展

SQLite 采用有界 executor 执行完整 repository 操作，密码计算单独限并发；摘要／revision 与正文、chunk 分离。高频列表只读摘要，终态正文不固定重复获取；源摘要在每上传锁内算，短全局事务复查提交身份。容量查询读互斥分类账本，不递归扫描每个文件。

预取默认关闭。开启时只准备紧邻一项，预处理并发／深度为 1、GPU 并发仍为 1；拿不到输出 lease 则退回正常路径，取消和停机由 owner 清理。真实批次耗时范围重叠，不能宣称普遍吞吐收益。

引擎注册统一 factory、worker、pipeline、输入规格、固定模型 revision 和运行配置身份。stub 不进入生产列表；新模型先独立 smoke 再启用。Qwen batch 当前如实声明串行，GPU batching 只能在另行实验有收益时加入。

## 诊断与验证

JSONL 关联 HTTP、任务、chunk、worker 和阶段耗时，默认排除自由文本、正文和凭据；日志有字节、队列、保留期及文件权限边界。未知异常只保留类型和有限栈位置。诊断入口见[诊断指南](../guides/asr-diagnostics.md)。

性能优化不能靠减小显式生成预算、丢弃输入区间、跳过必要对齐或关闭质量复核获得。CPU／mock 验证资源、调度和协议；真实 CUDA、吞吐、显存释放与质量另行验收。[测试指南](testing.md)保留原用例的合并／替代证据；未完成工作见 [roadmap](roadmap.md)。
