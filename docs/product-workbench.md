# 局域网工作台与 Qwen quality_v1

2026-10-26 音频与存储扩展：工作台使用 schema 4 和持久 192 kbps CBR MP3，识别及对齐 PCM 在操作退出后清理。该扩展覆盖此前持久 WAV 与旧任务自动迁移行为；旧 `/v1/**`、官方客户端 Skill、质量参数与模型能力保持原契约。产品决策及验收清单见[实施方案](workbench-audio-storage-plan.md)。

2026-10-25 平台实现扩展：解析前资源准入、统一同步/异步推理 FIFO、一次归一化资产、schema 3 摘要分页与正文分离、关联日志及显式引擎注册，以[平台运行契约](asr-platform-runtime.md)为准。真实 GPU 与质量验收状态见对应验收记录。

批准日期：2026-10-22。本文是 `docs/asr-server-prd.md` 的产品化扩展，覆盖旧版“不做 Web UI / 不引入数据库”的阶段性边界。

## 产品与运行边界

默认 Qwen 1.7B、transformers、官方 ForcedAligner；现有 MOSS 保持原策略。网页在 API 的同一 `18080` 端口提供上传音视频、排队、个人任务、回听改字及 TXT/SRT/VTT/JSON 导出，不依赖公网或云服务。前端为 React/TypeScript/Vite 静态产物，生产机器无需运行 Node 服务。

2026-10-25 界面更新：沿用用户 HCDS 雾蓝／卡布里蓝／麦黄／墨灰／纸白主题，GSAP 动效尊重减少动态效果设置。长结果采用固定视口、独立正文／工具滚动、动态高度虚拟列表；支持片段全文搜索、时间跳转及播放跟随，完整复制和导出不受可见行限制。手机端在正文与工具间切换。实现与验收见[工作台 UI 验收](validation-2026-10-25-workbench-ui.md)。

面向中文口播、课程、访谈、播客和中英混说。最大六小时是输入上限，不代表六小时识别质量已验收。`quality_v1` 初始 60 秒是实验候选；真实开发集、独立验收集通过发布门槛前，旧调用默认保持 `legacy`。

网页 `/api/**` 以个人用户名密码和 HttpOnly Cookie 认证；管理员创建、停用账户及重置密码。普通用户只能访问自己的任务、音频、上传和版本，服务端逐次检查归属。原 `/v1/**` 的可选 Bearer Token 不赋予个人工作台数据访问权；CLI/MCP 保持原行为。

## 质量流程

1. 选择首条音轨，生成或严格校验复用标准 MP3，保留有效时间轴；识别时从同一 MP3 创建临时 PCM，视频原文件在音频提交 ready 后删除。
2. 完整覆盖切分，在目标附近寻找低能量切点，不因检测到静音就丢弃输入。候选目标 30/60/90/120/180 秒，比较零重叠和边界两侧上下文，单次输入不超过 180 秒。
3. 每块使用任务冻结的词库快照；默认不传播上一块全文。自动生成预算 4096，EOS 提前结束；显式限制保持原值。
4. 截断、疑似循环、空白语音区、边界冲突及疑似术语问题有界复核：最多两轮，总复核音频预算不超过首次输入时长的 30%。未完成区间不得静默变成 completed。
5. 使用真实字词对齐确定边界归属；不以字符串模糊匹配直接删掉真实重复。真实 timestamps 来源始终可见，mock 不作为时间精度证据。
6. 返回模型原稿、规范稿、改词记录和待复核区间。对齐成功不等于文字正确，不提供未经校准的准确率百分比。

`quality_v1.2` 修复空稿区间判断：每个区间及复核子区间仍先运行 ASR，纯零 PCM 可以交付空文本；只在源音频尾部，且 PCM16 峰值不超过 4、非零采样比例不超过 5%、末尾全零至少持续 1 秒及该区间的一半时，允许把空结果作为近数字静音尾段交付。后者必须保留 `near_silent_tail` 待复核标记及采样证据。常量非零低音量、更强瞬时信号和非尾部输入不通过该豁免；这不是通用语音检测，也不据此跳过低能量音频。

流程版本参与检查点指纹。升级后不得拼接旧流程结果；检查点版本不匹配时明确拒绝恢复，用户可显式从头重试，以保留的标准 MP3 重新处理。

`quality_v1.3` 修复 Qwen 显式语言输入：仍由原生 `apply_transcription_request` 处理语言代码与音频，仅为该请求提供能保留 assistant 语言前缀的兼容模板，不修改模型缓存。自动检测路径不变；生成正文没有语言标签时，按请求语言进行后续对齐，不将空语言转换为字符串 `None`。

`quality_v1.4` 增加空稿后的独立语音证据：ASR 与有界复核仍返回空稿时，用固定版本 Silero 6.2.3 的 CPU 模型逐 512 个采样检查整个区间，同时检查原音及峰值增益视图。沿用该模型默认 0.5 语音阈值，任何一帧检出语音、所需增益超过 100 倍、依赖/模型不匹配或检测异常时仍不得豁免失败。仅两视图均未检出语音时，可交付空正文并保留 `empty_without_detected_speech` 待复核标记、检测器身份与处理证据；不裁剪原音，不补写文字，不声称证明无语音。检测器配置和模型文件 SHA256 同样进入检查点身份。该路径不修改旧 API 的默认 legacy 流程或 MOSS。

检测器模型单独安装到 `data/quality-vad/silero-vad-6.2.3`，可用 `ASR_QUALITY_VAD_DIR` 指定其他目录。该目录不加入 Python 模块搜索路径，避免启用旧流程中的可选 Silero 分块。常规 GPU 部署会安装此隔离模型包，已有环境可用 `--skip-gpu-deps --install-quality-vad` 补装；不替换已验收的 torch。

公共词库分通用和 AI/技术领域。任务本次热词优先于选定领域，再优先于通用词库。词条包含 `canonical / aliases / domain / auto_normalize`；只有已人工确认且无歧义的别名允许自动规范，疑似同音词通过回听与复核处理。不存在于正文的热词不补入。词库版本更新不改变正在执行或已完成任务。

## 持久化、恢复与容量

工作台使用 SQLite 和本地数据目录，保存账号、会话、上传、队列、检查点、词库快照和结果版本。旧 API 内存 job 不变，但两者共用一个 FIFO worker 和模型管理器，GPU 仍串行。

上传块大小 8 MiB，服务端顺序确认 offset；断线后查询已接收 offset，用户重选同一文件后续传。提交使用幂等键，重复提交不产生重复任务。浏览器关闭后任务继续。

每个有效分块原子保存检查点，包含标准 MP3 摘要、解码规范、PCM 摘要、模型 revision、流程版本、词库和参数。重启恢复队列顺序；未结束模型调用重做，已完成分块仅在重建 PCM 身份一致时复用。临时文件路径及 inode 不作跨操作稳定身份；版本不一致返回 `checkpoint_version_mismatch` 或 `resume_version_unavailable`，通过 `POST /api/jobs/{id}/retry` 的 `{ "from_start": true }` 清除旧进度并从头处理。

持久音频为 `audio.mp3`，界面显示“高质量回听 · MP3 192 kbps”。源采样率为 44.1/48 kHz 时保持，其余转为 48 kHz；单/双声道保持，多声道使用固定规则混为双声道。保存实际音轨索引、有效时长、编码规格、实际大小、SHA256、时间轴及工具身份；多音轨素材提示所用首条音轨。此处理不额外降噪、响度归一化、增强增益或裁剪静音，不恢复低质源已丢失的细节。

MP3 必须完整解码并通过帧、规格、尾部与有效时长校验；符合标准的 MP3 可直接复用，判断不能只依赖扩展名或平均码率。标准文件经 partial 原子提交并持久化 `audio_asset.state=ready` 后才删除唯一 source；转换失败或取消尚未 ready 时保留 source，重试重新准备。ready 后失败或取消仍保留 MP3 和有效检查点。首次转录、重试、自动复核及初始字幕对齐共用一次操作内的 PCM；编辑对齐独立预留空间，完整受控解码后按整数采样窗口取段，不做 MP3 随机截取。操作结束清理 PCM、分片、partial 和 lease 后才发布终态，下一次操作按需重建，MP3 不重复编码。启动恢复检查 ready 文件身份并回收无 owner 临时文件。

默认容量：单个上传 8 GiB、每用户保留区 16 GiB、全局 64 GiB、磁盘余量 5 GiB、最长六小时。限制均可配置。到达容量限制拒绝新上传，不删除未到期任务来凑空间。已完成/失败/取消任务在终态七天后清理，可提前删除。未提交上传也需要过期回收，避免永久占用配额。

容量账本分为 `saved_bytes`（保留文件）、`text_bytes`（文字与版本）、`temporary_bytes`（已落盘临时占用）和 `reserved_bytes`（未兑现预留）；`used_bytes` 为前三者之和，`allocated_bytes` 再加未兑现预留，可分配余量为配额减 allocated。分类互斥，不能把已落盘预留再次计费。转换峰值计入 source、MP3 及操作 PCM，192 kbps 仅用于估算，边界由实际字节及帧数检查；删除失败不得提前结算释放。轮询直接查账本，不扫描文件目录。管理端物理磁盘余量单独展示。上传、提交、终态、取消、对齐完成、删除和页面重新可见时刷新轻量统计。

保留数据与推理临时目录分离；临时分片和预处理副本用后删除。主动取消等待模型安全边界，不强杀当前推理。删除等待活跃消费者安全退出，再清音频、检查点、规范稿、原稿和全部编辑版本，不能让晚到的模型结果复活已删除任务。

文本按片段修改，携带 `base_version` 做乐观锁。只重新对齐受影响片段，新版本对齐未完成时禁止导出新 SRT/VTT；可回到已完成的旧版本导出。编辑不隐式改变公共词库。

## API 扩展

- `/v1/audio/transcriptions`、`/v1/audio/transcription-jobs` 增加可选 `quality_profile=legacy|quality_v1`，默认 legacy；MOSS 不接受 Qwen 专属 profile。
- `/api/auth/me|login|logout`：个人会话；`/api/capabilities`：启用模型、质量流程、领域、输入及保留限制。
- `/api/uploads`：创建；`GET/PATCH /api/uploads/{id}`：查询/顺序续传，PATCH 用 `Upload-Offset`；`POST /api/uploads/{id}/complete`：冻结参数并提交任务。
- `/api/jobs`：个人列表及音频实际大小；`/api/jobs/{id}`：查询/删除，详情包含 `audio_asset` 准备状态与编码规格；`POST .../cancel|retry`：取消/重试，retry 可传 `{ "from_start": true }`。
- `GET /api/jobs/{id}/audio`：ready 后返回 `audio/mpeg`，支持 Range、归属隔离及私有缓存；准备中返回 `409 audio_not_ready`，不回退到 source 或临时 PCM。
- `GET /api/storage`：个人分类占用、配额和剩余可分配量，不查询模型能力；管理端 `/api/admin/status.storage` 返回全局分类与物理磁盘余量。
- `PATCH /api/jobs/{id}/segments/{segment_id}`：带版本的改字；`GET .../export?format=txt|srt|vtt|json&version=N`：指定版本导出。
- `/api/admin/users`、`/api/admin/users/{id}`：账户；`/api/admin/glossary`：词库；`/api/admin/status`：队列与容量。

错误沿用现有 error envelope。前端不显示服务器文件路径、模型内部状态细节或虚构模型内部进度。实际模型能力仍从 registry 的 `/v1/models` 同源发现。

## 开发、运行与管理员

```bash
uv sync
cd frontend
npm ci
npm run build
cd ..
uv run python -m scripts.workbench_admin --help
uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

管理员使用本地 CLI 创建，密码通过交互或 stdin 输入，不预置默认密码、不打印会话 Token。CLI 与服务必须使用相同的 `ASR_WORKBENCH_DATA_DIR`。真实环境保持现有 WSL CUDA 安装流程；Mac 不安装 GPU 模型依赖。

新增环境配置：

| 设置 | 默认 |
| --- | --- |
| `ASR_WORKBENCH_ENABLED` | `1` |
| `ASR_WORKBENCH_DATA_DIR` | `data/workbench` |
| `ASR_WORKBENCH_RETENTION_DAYS` | `7` |
| `ASR_WORKBENCH_MAX_UPLOAD_BYTES` | `8589934592` |
| `ASR_WORKBENCH_USER_QUOTA_BYTES` | `17179869184` |
| `ASR_WORKBENCH_TOTAL_QUOTA_BYTES` | `68719476736` |

`ASR_MIN_FREE_DISK_MB` 同时保护临时数据与保留数据。`.env` 仅由部署生成的 systemd unit 自动读取，直接 Python 和管理员 CLI 需显式设置环境。

## 发布与回滚

先运行 mock 回归、类型检查、前端构建、真实浏览器流程，再在 WSL 独立代码目录和数据目录完成 CUDA、Qwen backend smoke、1.7B+ForcedAligner及完整工作台验收。线上服务不在测试过程中被替换。

切换前核对线上与本地文件差异，检查目标 FFmpeg 提供 MP3 编码器；在隔离代码与数据目录验证编码、真实转录、PCM 清理后回听及编辑对齐。进入维护窗口后关闭新上传和任务准入，等待活跃操作安全结束、停止服务，使用 SQLite backup API 保存一致数据备份及配套旧源码、前端和依赖记录。

schema 4 不迁移旧工作台任务。离线清理工具先列出任务、上传和文件数量及大小，执行后清除旧任务、关联与未提交上传、版本、chunk、检查点、legacy payload、音频及工作台专属临时文件并重建账本；账户、密码哈希和公共词库及版本保持。禁止清模型缓存或其他服务文件。旧 schema 存在任务/上传时启动拒绝并返回 `schema_reset_required`；清空后可升级。发布后确认登录、词库、空任务列表及容量，再完成新样本端到端验收后恢复准入。

回滚必须同时恢复匹配旧代码、静态产物和迁移前完整数据，不能让旧程序读取 schema 4。旧任务清除授权只作用于在线任务，发布备份保留至验收完成；完成后清除本次临时备份中的旧任务数据并核对磁盘释放。新流程失败时旧 API 默认值仍保持 legacy。真实质量门槛见 [评测说明](../evaluation/README.md)，未试听不记为音质已验收。

### 发布包与离线数据备份命令

```bash
# 必须先完成frontend构建；产物含逐文件SHA-256清单，不含数据/secret/node_modules
uv run python -m scripts.workbench_release bundle --output /path/to/new-release.tar.gz
# 停止使用该data-dir的服务后备份；若仍持有service.lock则拒绝操作
uv run python -m scripts.workbench_release backup --data-dir /path/to/data/workbench --output /path/to/new-backup
# schema 4 升级：同样要求服务停止；默认只预览，确认本次清除范围后执行 --apply
uv run python -m scripts.workbench_reset_tasks --data-dir /path/to/data/workbench
uv run python -m scripts.workbench_reset_tasks --data-dir /path/to/data/workbench --apply
# 恢复到新目录或空目录；逐文件校验、SQLite integrity_check，并报告源码匹配状态
uv run python -m scripts.workbench_release restore --backup /path/to/new-backup --data-dir /path/to/restored-workbench
```

备份包含账号密码哈希、保留音频与结果，应保存在 Git 之外。首次恢复需核对 `required_source.source_digest` 与配套源码包；它包含未提交改动，不能只看 Git commit。工具拒绝覆盖既有备份、不自动重启服务或切换线上目录。


## 转录恢复与服务就绪

工作台 schema 5 从 schema 4 事务迁移，保留全部现有任务、MP3、检查点、正文版本与保留期限；**此次升级不得运行 `workbench_reset_tasks`**。更早 schema 的音频迁移限制仍按对应旧版发布说明执行。旧程序会拒绝 schema 5，回滚必须保留新运行数据并使用匹配源码/前端/数据快照，不能用旧备份覆盖发布后的新增结果。

基础 `quality_v1.4` 身份保持正常模型生成和切分不变，恢复记录以独立 `recovery_v1.0` 绑定原基础指纹。旧成功块摘要核对后只读复用，旧失败调用由持久错误及原时间窗口显式导入。恢复预算继承旧账本：每原始块最多三个动作（有依据的语言恢复、二分、四分；正常合并复核也占同一动作上限），总复核音频不超过首次输入的 30%。预算先落盘再调用；短音频额度不足时明确停止。普通重试和服务重启不会重新赠送额度。

仅 auto 空稿且最近三个已确认非空块语言一致、合计至少 30 秒、没有冲突或推断语言标记时尝试一次显式语言。记录来源索引、请求/有效语言、调用参数和策略版本；非空候选还需截断、重复、真实对齐与明显有声遗漏检查，保留待复核标记。该启发式不替代人工语言或文字确认。

失败子段保留范围和原因并继续检查其余子段，不能拼接局部成功掩盖缺口。原始块质量恢复耗尽后继续处理后续块，检查点支持稀疏成功区间。全部区间通过才为 completed；有缺口仍为 failed / quality_incomplete，但保存部分 result、新 version、completeness 和 gaps。部分 TXT 标明“不完整”和时间缺口，JSON 保留结构；有缺口或缺少有效真实时间的版本拒绝字幕导出。原始文字在对齐前持久化，对齐失败重试只补对齐，不重识别。

任务摘要返回有界 `retryability`、`next_action`、`next_attempt_at` 和完整性计数，正文和完整缺口仅在所属任务详情读取。前端按后端动作显示恢复、从头处理或回听。执行用持久 attempt/generation 和版本 fence，取消、删除、编辑或新尝试接管后不能提交晚到结果。

基础设施恢复每操作最多一次、任务累计三次，强退留下的活跃尝试也计费。默认从 5 秒退避，后续按累计次数增长（上限 60 秒）；等待释放调度槽位，到期作为新提交进入 FIFO。连续三个 fatal worker 错误冷却 60 秒，真实推理成功才清零；无法确认旧进程退出时全局隔离 GPU，不启动第二个 worker。识别空稿不触发 worker 熔断。

`GET /health` 保持原存活语义。新增公开 `GET /readyz`，200 表示可接新推理，503 返回有限原因；不公开任务、路径或账户。正常模型 unloaded 不影响就绪，队列满仍为 429。推理降级时已有文字、回听、导出、登录、取消继续可用。管理员 `POST /api/admin/drain` 接收 `{ "draining": true|false }`，停止新推理提交、在安全边界保存活跃进度，并返回活跃操作/回听消费者；恢复准入前核对数据和队列。管理员 status 同时返回恢复统计（保留任务窗口及分母）、worker/维护/排空状态。

分阶段超时配置 `ASR_WORKER_CHUNK_TIMEOUT_SECONDS`、`ASR_WORKER_ALIGNMENT_TIMEOUT_SECONDS`、`ASR_WORKER_NATIVE_TIMEOUT_SECONDS` 未指定时继承旧 inference deadline。300 秒观测只记日志，不自动中止；收紧阈值需目标机器实测。维护单轮异常退避后继续；调度监督只能在没有残留调用 owner 时重建。systemd 使用 Restart=on-failure、300 秒内最多 5 次启动和 120 秒停机上限；独立线程 watchdog 仅监测事件循环持续失联，不根据 GPU 利用率或 UI 百分比中断长调用。

本轮方案及验收范围见 [专项方案](transcription-resilience-plan.md)。真实推理、故障注入、持续运行和六小时质量分开报告；未测项目不能由源码或一次 smoke 替代。

声音检测没有取得阴性证据不等于确定存在人声或漏词；音乐、音效、呼吸及检测不可用须保持未分类/具体失败原因。声音疑点在界面显示“待回听确认”，不把整个处理块当作已证明遗漏的台词区间。离线 `inspect_speech_evidence` 提供双视图概率摘要和原因，不授权将未知区间放行为完整结果。见[本轮验收](validation-2026-10-26-transcription-resilience.md)。
