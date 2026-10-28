# 代理开发指南

## 当前入口

- 修改行为前先读 [API／产品契约](docs/reference/asr-server-prd.md)、[工作台契约](docs/reference/product-workbench.md)和[平台运行契约](docs/reference/asr-platform-runtime.md)。[文档索引](docs/README.md)区分指南、当前参考、设计与历史验收。
- 本文件是现行开发指引；`CLAUDE.md` 保持指向本文件的兼容符号链接，不复制正文。历史计划、任务包、验收中的授权和命令不构成新任务授权。
- 服务端和客户端角色入口为 [server-agent](prompts/server-agent.md)、[request-client-agent](prompts/request-client-agent.md)。实现范围由当前用户指令及现行契约确定，不重复执行项目初始化任务。

## 产品与数据不变量

- 使用 Python 3.12、uv、FastAPI、Uvicorn。唯一 LAN 端口 `18080` 同时提供工作台与 API；不新增公网隧道、端口映射，不暴露 worker 端口 `8001` 或历史测试端口。
- `/v1/**` 使用可选 Bearer、内存 job 和临时输入，默认 `quality_profile=legacy`；重启后不承诺恢复服务端输入。`/api/**` 使用独立个人账户、HttpOnly 会话及服务端归属检查，Bearer 不赋予工作台访问权。
- 同步、异步、工作台、编辑对齐和模型 load 共用有界单 GPU FIFO，不用同步调用绕过队列。取消等待安全边界；清理文件和资源 lease 后才发布终态。活跃模型卸载进入 `unloading_scheduled`，新同模型调用返回既有 409，不强卸载活跃模型。
- 工作台当前 schema 为 **5**。**schema 4→5 原位保留账户、任务、MP3、版本、检查点与保留期限，不执行 `workbench_reset_tasks`。** 更早格式的历史边界见[升级指南](docs/guides/upgrading.md)，不能套用旧清库操作。
- 工作台终态默认保留七天。持久音频为经完整校验的 MP3 192 kbps CBR；ready 元数据提交后才删除原始上传。操作内按需创建受控 PCM，退出时清理；重试不重复编码 MP3。删除不得被晚到结果撤销。
- 检查点核对 MP3／PCM 摘要、解码规格、模型／流程／参数／词库身份，不匹配明确拒绝，可显式从头重试。保留成功块及恢复预算账本，普通重试不重置额度；缺口结果保持 `failed/quality_incomplete`，完整真实对齐前不提供字幕。
- 进度只报告阶段与完成 chunk，最终正文以 `result.text` 为准；native long-form 不伪造 chunk 内百分比。能力来自 `/v1/models`；错误保持 `error.code/message/details` 信封。

## 模型、质量与平台边界

- 默认 Qwen3-ASR 1.7B，Qwen 与可选 MOSS 使用 HF native `transformers`；不把 `qwen-asr` 包、vLLM、streaming 或未验收后端声明为可用。
- Qwen 官方 ForcedAligner 是已实现的可选路径，真实验收后才启用 `ASR_ENABLE_QWEN_ALIGNMENT=1`。不得伪造时间戳；独立 `/v1/audio/alignments` 仍是预留接口。
- MOSS 需独立 smoke 并复核 CUDA torch 后启用。段级时间不等于字词对齐；跨块标签读取实际 `speaker_scope`，保留 mixed／partial／unresolved。Anchor Replay 自动生成预算在未显式指定时至少 24000，显式值保持，上限 65536。
- `quality_v1.4` 基础身份和 `recovery_v1.0` 恢复策略分开记录；词库按任务冻结，只自动规范人工确认的无歧义别名，不补写不确定内容。
- 六小时是输入上限，`quality_v1.validated=false` 保持。六小时质量、独立参考集、八小时稳定性等未完成项见 [roadmap](docs/development/roadmap.md)；工程或 mock 不能替代。
- macOS 只负责轻量开发、CPU／mock、客户端和静态构建，不安装 CUDA、GPU torch、模型包或大模型缓存。适配器内懒加载依赖。
- 真实 CUDA、模型推理和常驻部署在获授权的 Windows／WSL 环境进行；当前已验证主环境为 WSL2 Arch Linux + RTX 5070 Ti 16GB。部署推荐 Linux 原生文件系统 `~/services/hcds-asr-server`，不放 `/mnt/c`；其他平台独立验证。
- **连接 Windows 前须取得当前任务明确授权**，包含探测、SSH、命令和传输；随后读共享 `windows-node` Skill，保持严格主机密钥校验，不擅改防火墙、权限或 WSL。历史授权不延续。

## 官方 Skill 与配置

- `skills/hcds-lan-asr/` 是正式客户端 Skill 来源。`GET /skill`、`/skill/hcds-lan-asr.zip`、`/skill/asr_batch.py` 同版本分发；协议变化同步 Skill 与客户端，发行包必须包含两份源文件。
- 模板不绑定机器 IP。下载 ZIP 动态生成 `server.json`：优先 `ASR_PUBLIC_BASE_URL`，空值／loopback／通配地址取下载 origin。客户端依次取命令行、环境、随包地址、本机配置、loopback；升级保留批次数据库及请求快照。
- `ASR_API_TOKEN` 保护 `/v1/**`，CLI 使用 `ASR_TOKEN`，MCP 使用 `ASR_MCP_TOKEN`，不得打印或提交真实凭据。systemd 读取 `.env`，直接 Python 与 MCP 不自动加载；LAN 客户端绕过代理。

## 验证

本地默认不设置真实 HTTP smoke 的 `ASR_BASE_URL`，使用独立开发数据目录：

```bash
uv sync --frozen
uv run --frozen python -m scripts.check_release_dependencies
uv run --frozen pytest -q
uv run --frozen mypy asr_server asr_mcp tests scripts
npm --prefix frontend ci
npm --prefix frontend run build
```

工具和覆盖边界见[测试指南](docs/development/testing.md)。只为明确行为风险补必要回归，不为文案或可逆移动写逐字断言。真实部署需验证官方 CUDA wheel 来源、torch CUDA／capability、Qwen／MOSS 独立 smoke；命令见 [WSL 指南](docs/guides/wsl-deployment.md)。`nvidia-smi` 不能单独证明 Python CUDA 环境正确。

## 文件准入与维护

- 保留源码、测试、锁文件、正式客户端、部署工具、必要公开资源和合成样例。`plan`、`smoke`、`probe`、日期或自动生成本身不构成删除理由。
- `docs/guides/` 写操作；`reference/` 写当前行为；`development/` 写设计理由、验证和未完成工作；`validation/` 只放绑定版本的精选摘要。完成任务提炼后退出当前实施导航，历史失败与限制不抹去。
- `artifacts/`、`scratch/`、运行目录与客户端 `asr-transcripts/` 默认不进 Git；真实录音、正文、数据库、备份及敏感原始证据只用获准私有位置。忽略目录不等于私有权限，也不代替部署／发布准入。
- 删除、合并或迁移前检查源码、测试、文档、CI、动态加载、部署和发行清单，更新替代入口及必要兼容。不要机械移动相互导入或按层级推导根目录的脚本、官方 Skill。
- 仓库自有代码采用 [MIT](LICENSE)，模型、依赖和外部素材分别遵守其许可。公开样例只使用来源与许可明确的合成内容；原始验收附件已从当前树移除，不新增公开原始 archive。历史重写、推送、发布和生产操作仍须当次授权。
- 默认日志、错误及可分享摘要不记录密码、Token、正文、热词或私人路径。验收只留脱敏位置、方法、版本、指标与限制；正文仅在明确内容选项及获准私有范围保存。公开 diff 的删除行同样检查。
- 发布分别核对 Git、部署复制和产品包；`frontend/dist/` 不进 Git但运行包需要。源码、锁、构建与逐文件 manifest 必须可对应。回滚使用匹配源码、前端和数据快照，不盲目 `rsync --delete`。
