# 工作台 MP3 与存储验收

> 历史摘要：以下结果属于原日期、环境和版本，本轮未重新验证。原始验收附件已按维护决定从当前树删除，未另存副本；Git 历史未改写。
> 本页短样本结果对应已退出的 13.06 秒旧录音；同路径现改为合成样例，不能用它复现旧样本的质量或计时数字。

日期：2026-10-26。实施依据：[执行方案](../development/audio-storage.md)。工程流程、本地回归、WSL 短音频 GPU 验证和原 18080 入口的线上验收均通过；本页保留线上切换、旧任务清理和最终检查结果的历史摘要。

## 实现

工作台保留经过完整解码和帧校验的 `audio.mp3`（192 kbps CBR）；44.1/48 kHz 及单/双声道保持，其他采样率转 48 kHz，多声道按已记录的固定规则混为双声道。所有操作都从同一 MP3 重建模型 PCM。源上传仅在 MP3 原子提交且 ready 元数据持久化后删除；转换失败保留唯一上传源。

转录、复核、首次对齐、重试及编辑对齐退出后，删除操作 PCM/分片，释放 lease 并结算容量，再保存终态。清理失败保留占用并进入待清理状态；播放和推理消费者退出前不移除音频。重启回收无主操作文件、未提交 MP3 及 partial，完整核对 ready 文件身份。检查点同时验证 MP3、当前 FFmpeg 构建、PCM 摘要、模型、请求和质量流程；不匹配时明确提供从头重试。

schema 4 分类账本区分持久音频、文字/版本、临时已写与未兑现预留。`/api/storage` 独立刷新，列表不扫描目录。旧任务由离线工具清除，用户、密码哈希、会话和公共词库完整身份保留。旧 `/v1`、模型能力、官方客户端 Skill 与质量默认参数保持原契约。

## 本地检查

- `uv run pytest -q`：454 passed，4 skipped（未配置在线 HTTP smoke 与本机缺少 rsync 的可选集成项）。
- `uv run mypy asr_server asr_mcp scripts tests`：125 个文件通过。
- `npm --prefix frontend run build`：TypeScript 与 Vite 通过。
- `uv run python -m scripts.check_release_dependencies --media`、`git diff --check`：通过。
- 真实 FFmpeg 测试覆盖 WAV/AAC/视频、首音轨与多声道、CBR 复用和 VBR 转码、逐帧与尾部完整性、低估时长、时长/容量/磁盘边界、取消/超时、缺编码器及恢复身份。
- WSL 最终候选包的媒体、生命周期、存储、接口与发布回归：79 passed。
- 生命周期测试覆盖失败后保留 MP3、重试和编辑重建 PCM、检查点拒绝与从头重试、删除/播放并发、删除失败保留账目、预取 MP3 取消与启动回收。
- 合成工程标记在开头、中段、末尾的重建时间偏移小于 100 ms。该结果不表示模型对齐精度。

## WSL 实机

隔离源码与全新数据目录，不另开监听端口，使用生产虚拟环境。FFmpeg n8.1.2 提供 libmp3lame；torch 2.11.0+cu128、CUDA runtime 12.8、RTX 5070 Ti。没有改装模型或 GPU 依赖。

历史 GPU 摘要：当时的 13.06 秒旧录音样本、Qwen3-ASR 1.7B、quality_v1 和官方 ForcedAligner。四块上传后形成 315,117 字节的 48 kHz 双声道 MP3。首次转录、改字对齐、重试的 MP3 SHA256 与重建 PCM SHA256 均一致；三个操作后 source/WAV/partial/lease 均清理。Range 206、`audio/mpeg`、私有缓存和跨用户 404 通过；TXT/JSON/SRT/VTT 初始及编辑后均 200。删除后分类账本全部为零。

历史同源对照：原 48 kHz/24-bit 双声道 WAV、旧 16 kHz 单声道 PCM、新 MP3 在同参数 quality_v1 下文本相同，均有 37 个真实对齐单元。这里统计的是模型输出之间的字符差异，不能当作人工参考 CER。

可重复运行：

```bash
# 使用全新/空数据目录；不连接在线数据、不开放测试端口
ASR_ADAPTER=qwen ASR_ENABLE_QWEN_ALIGNMENT=1 .venv/bin/python -m scripts.workbench_audio_smoke \
  --data-dir /path/to/new-isolated-data --audio test-fixtures/audio/test_short.wav \
  --output /path/to/new-report.json --real-gpu
```

脚本中的重试通过明确记录的 completed→failed 工程状态切换触发，不宣称 GPU 故障注入。mock 模式只验证工程行为，不产生质量证据。

## 浏览器

使用 Ego Lite 在本机 mock 和已部署真实服务验证上传说明、独立容量更新、MP3 ready 后自动挂载播放源、13.06 秒媒体完整加载、8 秒跳转和 390×844 移动布局；文档宽高与视口一致。线上改字后 v2 对齐通过，四种导出均 200；测试任务删除后列表为空、容量自动归零。本机另用工程状态验证“从头重试”按钮。原始服务截图已删除，当前不提供该画面证据。窄屏为 Chromium 模拟，不代表实机 iOS/Safari 验收。


## 质量边界

本次没有覆盖口播/背景音乐/低音量/混说完整人工参考集，也没有人工对照试听和真实六小时质量验收。短音频同源对照、工程时间标记和 CUDA 流程通过，不等同于完整音质、CER 或长音频对齐验收；既有 `quality_v1.validated=false` 保持。

## 线上发布结果

`asr-server.service` active，原 18080 单端口入口保持；244 个候选包文件在安装后逐一校验 SHA256。schema 3 的 7 个旧任务、7 个版本、7 个 legacy payload、7 个关联上传及 583 个文件（818,907,430 字节）清除，schema 升为 4。账户、会话、完整公共词库身份在清理前后相同。线上测试任务已删除，各分类及未兑现预留归零，SQLite integrity_check 通过。

旧 `/v1` 在线 HTTP smoke 2 passed（真实转录和官方时间戳能力）。完整源/数据备份在切换前生成；线上验收完成后已校验并移除本次临时旧任务备份，物理磁盘可用量增加 856,375,296 字节。配套旧源码包与发布记录保留，未触及历史备份、模型缓存和无关服务。
