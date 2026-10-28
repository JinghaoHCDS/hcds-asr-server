# 平台执行与资源契约

当前参考：schema 5、统一单 GPU FIFO、有界正文与 PCM、持久 MP3、分离摘要／正文／chunk 和关联诊断。设计理由见[平台设计](../development/platform-design.md)与[音频设计](../development/audio-storage.md)；历史测量和限制见[平台验收](../validation/2026-10-25-platform.md)。模型策略及质量启用门槛保持各自契约，工程实现不等于 GPU／听感／长音频质量验收。

## 请求、调度与取消

应用仅在原 `18080` 提供工作台及 API。请求生成可信 `X-Request-ID` 和 `X-Trace-ID`；客户端输入不能覆盖服务 ID。关联事件、轮转和离线诊断见[诊断文档](../guides/asr-diagnostics.md)。

所有 HTTP 推理入口使用同一个有界 `OperationScheduler`：短同步请求等待 Future，旧异步 job 保持内存 owner，工作台及编辑对齐保持个人持久 owner；管理员 load 同样排队。FIFO 按提交序号，不按模型重排；空位释放后立即补入较早的持久任务，新请求准入前也先补队列，60 秒维护仅作恢复兜底；运行和等待项均计入 `ASR_MAX_QUEUED_JOBS`。同步接口原先超过十分钟自动返回 `202` job 的协议保留。卸载意图不排在整段转录后，活跃调用继续使用原 `unloading_scheduled` 保护。

排队请求取消后移除票据。正在模型调用的取消等待当前调用安全结束；同一线程可见事件传到解码、扫描、分块及每个后续模型调用边界。业务取消不会终止调度 worker，任务终态在文件清理和配额释放后发布。

`ASR_PREPROCESSING_PREFETCH=0` 为默认。开启时预备紧邻的一项，解码并发和预取深度均为 1，GPU 并发仍为 1；预备先取得初始输出 lease，不能获取余量时退回正常执行。准备失败、取消及关停均由资产 owner 清理。容器时长仅用于初始估算；实际输出超过估算时，在落盘前重新检查个人/全局配额、临时额度与磁盘余量并扩大 lease，实际帧数始终受六小时上限约束。它不提供吞吐保证；本轮三轮真实对照观察到 RPC 间隔缩短，批次耗时范围仍重叠，详见 [WSL 验收](../validation/2026-10-25-wsl-platform.md)。

## 解析与音频资产

纯 ASGI 正文限额在 JSON/Form 解析前执行，顺序为关联、相应鉴权、准入、解析、业务。工作台上传块在读 body 前检查归属。总字节以每次 receive 累计为准；Content-Length 仅用于提前拒绝。停止读取超限正文，超时、断连及取消关闭所有 parser 文件。受控 multipart parser 是唯一的框架版本适配边界，仍保留 Form 参数和 OpenAPI。

默认限额：旧上传为 `ASR_MAX_UPLOAD_MB` 加 256 KiB 元数据；单文件、最多 32 个字段，字段与头的累计元数据也受限。工作台 PATCH 8 MiB、普通 JSON 256 KiB、公共词库 1 MiB。接收槽位默认 4，等待单次正文块默认 30 秒，槽位在正文接收完成后释放。所有 parser spools 和旧 API workspace 位于服务专属临时根，解析副本及上传副本同时存活时分别计费。

`MediaPreprocessor` 统一 probe/decode、同步/异步/bytes 包装，默认只允许普通上传媒体 demuxer 与 `file` 输入协议，排除 HLS/DASH/concat 等播放列表。格式行为依赖实际 FFmpeg 构建；兼容和二级引用证据见验收记录，不能将协议白名单描述成文件系统沙箱。

解码输出为受选中模型 `AudioSpec` 约束的 PCM16 WAV，以固定小块读取 stdout，在每次落盘前检查实际 PCM 字节预算；有限 stderr 同步消费，超限、取消和 deadline 回收 decoder 和 partial。恰好到达预算仍须检查 EOF，截断输入不得作为完整音频交付。逻辑额度及各文件系统尚未落盘预留分开计算；持久上传未写余量和处理预留与临时预留合计检查，长期承诺不占临时 spool 配额。PCM 在同一资源锁内从持久承诺转交 decoder lease，落盘后增量兑现预留，EOF 归还未使用部分，owner 删除后释放保留部分。硬链接及借用不重复计物理文件。

窗口按整数采样位置前进，显式零及非有限数受控拒绝。多个 VAD 区间共用追加前的 4096 块上限；扫描、合并和窗口循环均检查同一取消信号。

工作台以 `audio.mp3` 作为持久音频：192 kbps CBR；44.1/48 kHz 源保持，其余转 48 kHz，单/双声道保持，多声道固定混为双声道，记录首条音轨的实际索引及数量。标准化模块与模型预处理分离，不附加降噪、增益或静音裁剪。符合标准的 MP3 需完整检查 MPEG 帧、编码、采样率、声道、解码有效帧数和尾部后才能直接复用；平均码率不能证明 CBR。输出校验考虑编码延迟和尾部填充，不能只比较容器 duration。编码/解码沿用协议白名单、取消、deadline、磁盘与实际字节预算；超过六小时或异常尾部不得提交假 ready。

MP3 使用 partial 和原子重命名，数据库记录 preparing/ready/failed、版本、SHA256、大小、有效时长、时间轴及工具身份；ready 持久化后才删除唯一上传源。失败保留 source，回收 owned partial，重试可重新准备。启动恢复核对 ready 文件存在性与身份，清理无 owner 操作临时文件与无效预留，不从旧 WAV 恢复播放。

每次转录或编辑对齐从同一标准 MP3 按选中模型规格生成受控 PCM。`NormalizedAudioAsset` 记录规格、帧数、PCM SHA256、owner 和 `pcm16-wav-v2` 版本；转录、自动复核和初始对齐在一次操作内借用同一资产。借用时核对文件身份、大小及纳秒修改时间，借用者不得删除资产；owner 等所有消费者退出，删除 PCM、分片并结算 lease 后发布终态。重试和编辑对齐重新解码，不重新编码 MP3。编辑对齐先完整解码，再按整数采样窗口取段，避免 MP3 随机截取的帧边界偏移。

新 WAV 使用确定的 44 字节头。质量 checkpoint 身份包含标准 MP3 摘要、解码规范、PCM 摘要、规范化、引擎、模型、pipeline、输入规格、请求、词库、aligner 和检测器身份；临时路径及 inode 不作跨操作身份。每次重建 PCM 验证摘要，不一致时返回 `checkpoint_version_mismatch` 或 `resume_version_unavailable`，不得拼接旧 chunk；用户可用 `POST /api/jobs/{id}/retry` 的 `{ "from_start": true }` 从头重试。旧 `/v1/**` 的临时输入、模型能力与质量默认值保持原行为。

## 工作台存储与查询

SQLite 使用 WAL；完整 repository 操作通过有界 executor 在线程内建连、执行事务和关闭。密码计算使用独立两槽执行器，等待数量有上限；登录失败地址记录按 TTL 和容量清理。

数据库 schema 5 保存音频元数据、分类容量账本和恢复状态，延续摘要、revision、状态索引、正文 versions 和已完成块 job_chunks 的分离。schema 4→5 事务迁移保留所有数据，不执行 reset；schema <4 的历史格式边界见[升级指南](../guides/upgrading.md)。分块 checkpoint 写入后才发布完成块；普通显示进度至多每秒落盘一次，chunk、取消/删除、版本和终态立即持久化。提交源摘要在每上传锁内计算，离开全局锁，再在短事务中复查上传状态及幂等身份。

分类为保留文件 `saved_bytes`、文字及版本 `text_bytes`、已落盘操作文件 `temporary_bytes`、未兑现 `reserved_bytes`；前三项相加为 used，used 加未兑现预留为 allocated。准入使用完整 allocated 与磁盘未兑现承诺，不重复加入已经落盘的预留。源文件、MP3 与 PCM 同时存活的峰值受预算保护，删除失败保留实际占用。容量日常查询读账本，目录扫描仅在启动恢复和低频维护。`GET /api/storage` 返回个人分类、quota 和 available；管理员 status 返回全局分类及独立 `disk_free_bytes`。任务摘要同时返回 `audio_size_bytes` 和轻量音频规格；播放接口只提供 ready MP3，`audio/mpeg`、Range、owner 与私有缓存保持。

`GET /api/jobs?view=summary&limit=50&before=<sequence>` 返回 `{jobs,next_cursor}`，不读取正文；limit 上限 100。`ids` 可限制活动任务轮询，保留 owner 条件。旧列表调用仍可用。详情返回 revision/ETag，`If-None-Match` 未变化时 `304`。前端只轮询活动任务，隐藏页降低频率，终态停止固定拉正文，编辑后主动失效刷新。管理员队列只含任务/模型/状态/时间/队列位置/chunk计数/错误码白名单。`/health` 返回缓存 GPU 快照及 `gpu_snapshot_at`，请求不触发 CUDA 查询。

`POST /api/uploads/{id}/complete` 可选 `model`、`quality_profile`；不传时沿用可用 Qwen quality 默认。只展示和接受注册表声明的组合。基础第三引擎可只支持 legacy，不能接受对齐、字幕或 quality 请求；能力不足在解码前拒绝。更换模型/profile/引擎或规格不会复用旧幂等结果。新增引擎接口与 stub 见[接入模板](../development/adding-asr-engine.md)，stub 不进入生产模型列表。现有 Qwen batch 如实声明串行。声明 `segment_timestamps` 的模型可使用真实段级时间导出字幕，状态为 `segment_timed`，不等于 word/char `aligned`；历史 `unavailable` 结果在范围、来源和完整文字核对通过后同样可导出。chunk 窗口不得充当字幕时间。模型推理依据及选用范围见[模型实践对照](../development/asr-model-practices.md)。

## 配置与升级

新增设置见 [.env.example](../../.env.example)：`ASR_RECEIVE_SLOTS`、`ASR_RECEIVE_TIMEOUT_SECONDS`、`ASR_MULTIPART_METADATA_BYTES`、`ASR_PREPROCESSING_PREFETCH`、`ASR_TEMP_DIR`、`ASR_LOG_LEVEL/FORMAT/DIR`。`ASR_ADAPTER=registered` 是通用注册器入口，旧 `qwen` 配置保持兼容；不会自动启用新生产模型。`.env` 仍仅由部署生成的 systemd unit 读取。

升级与回滚按[版本适用指南](../guides/upgrading.md)执行：隔离验证、排空、等待安全退出、停服务、获准私有备份，再切换匹配源码／前端。当前 4→5 不清任务；旧程序拒绝新 schema，回滚保护新增数据。源码改动不等于线上已切换。测试及发布检查见[测试指南](../development/testing.md)。

## 转录恢复与服务就绪

schema 5 从 schema 4 原位迁移，保留所有现有任务、音频、检查点、正文版本和保留期限。此次升级禁止使用历史 `workbench_reset_tasks`；旧代码拒绝 schema 5，回滚须保存发布后的新增数据并恢复匹配快照。详细恢复协议以[工作台契约](product-workbench.md#转录恢复与服务就绪)为准。

工作台持久 attempt/generation fence、基础设施恢复预算、到期重新排队顺序和独立 recovery_v1.0 质量账本共同约束恢复；所有入口仍共用单 GPU FIFO。`/readyz` 使用调度、维护、数据写入及磁盘余量缓存、worker隔离/熔断和排空状态；读取接口继续可用。管理员排空和恢复统计、阶段deadline、维护监督及systemd设置见同一契约与 [.env.example](../../.env.example)。
