# 请求端 Agent 角色

先请求目标服务的 `GET /skill`，阅读与其版本一致的官方用法，再读取 `GET /openapi.json` 与 `GET /v1/models`。ZIP `/skill/hcds-lan-asr.zip` 提供同源客户端和无凭据服务地址；已有批次数据库和请求快照保留。

旧服务明确返回 `/skill` 404 时按实际 OpenAPI 接入并说明版本差异，网络／鉴权失败不能当作旧版本。仓库中的补充入口为 [AGENTS](../AGENTS.md)、[MCP 指南](../docs/guides/mcp-bridge.md)与[批量指南](../docs/guides/batch-workflow.md)。

## 输入与职责

- 使用用户配置的服务 origin；LAN 请求绕过代理（如 `curl --noproxy '*'` 或 HTTP 客户端 `trust_env=false`），设置有限连接／读取超时。
- 先发现模型、后端、timestamps、speaker scope 和 pipeline，不能硬编码未声明能力。REST 的 `purpose=podcast` 不会自动从默认 Qwen 切换 MOSS，调用方须显式选择已声明的 MOSS。
- 完整文件用同步或异步上传；长任务优先创建 job，轮询 `detail=compact`，完成后请求一次 `detail=full` 读取 `result.text`。只显示真实阶段／chunk 进度，不伪造模型内部百分比。
- `/v1` job 在内存中，重启后需重新提交；本地批次恢复不等于服务端断点续算。工作台 `/api` 另用个人 Cookie 和归属检查，Bearer 不授予个人任务权限。
- MOSS 保留实际 global／chunk／mixed、partial／unresolved，不把相对标签改成真人身份或全局一致标签。Qwen 时间粒度取实际对齐返回值，不按文字均分时间。
- 处理稳定错误码，按 Skill 对网络／队列暂态做有界重试；鉴权、能力、内容、超限或完整性错误不得无穷重试或静默降级。取消用现有接口，等待安全模型边界。
- Token 从客户端配置环境读取，不写日志、结果、命令样例或工具返回。MCP 只读允许根目录内解析后的普通文件，不扩大到用户 home 或文件系统根。

## 验证与交付

用合成或已获准素材验证发现、提交、状态、取消和最终输出；本地结果包含源路径、请求与正文，按私有运行数据维护。默认分享摘要只含模型、语言、长度、hash、阶段和错误类别，不复述转录正文或 Token。不要为连通性检查擅自连接 Windows 节点、修改服务或读取其他用户任务。

指出实际平台、版本和未测试项；MOSS 4–5 小时整期一致性等限制不能用短样本替代。修改协议或客户端行为时同步[官方 Skill](../skills/hcds-lan-asr/SKILL.md)，不复制另一套长期 API 契约。
