# MOSS Anchor Replay 四说话人 WSL 验收记录

> 历史摘要：以下结果属于原日期、环境和版本，本轮未重新验证。原始验收附件已按维护决定从当前树删除，未另存副本；Git 历史未改写。

## 结论

2026-07-15，`moss_anchor_replay` 在 RTX 5070 Ti 16GB 的真实 MOSS `transformers` 后端上完成 4 个不同声音、2 个连续块的跨块回连。第二块的本地 `S01`、`S02`、`S03`、`S04` 全部映射回第一块建立的 `speaker-0001` 至 `speaker-0004`，结果为 `complete/global`，没有冲突或未解析段。

这是 opt-in 能力，默认 `speaker_resolution=off`。验证证明当前实现至少支持 4 人，不证明任意录音条件、重叠说话或无限人数都可靠。

## 环境与输入

- 部署目录：`/path/to/user/services/asr-server`
- API：`http://127.0.0.1:18080`
- GPU：NVIDIA GeForce RTX 5070 Ti 16GB
- 模型：`OpenMOSS-Team/MOSS-Transcribe-Diarize`
- 后端：`transformers`
- 输入：4 个本地不同声音片段顺序拼接成 69.8 秒 block，再重复一次，总时长 139.6 秒
- 请求：`split_strategy=fixed`、`max_chunk_seconds=69.8`、`overlap_seconds=0`、`speaker_resolution=auto`

原始声音、组合音频、完整文本与响应 JSON 不进入 Git，验收后已从 WSL 临时目录删除。

## 结果

- HTTP：200
- chunk：2
- `diarization.method`：`moss_anchor_replay`
- `diarization.status`：`complete`
- `diarization.speaker_scope`：`global`
- `speaker_count`：4
- `unresolved_segments`：0
- `conflicts`：0
- `anchor_budget_limited`：false
- `candidate_speakers`：0
- `missing_segment_chunks`：0
- 第二块映射：`S01→speaker-0001`、`S02→speaker-0002`、`S03→speaker-0003`、`S04→speaker-0004`
- segment：18
- 尾部覆盖：139.48 / 139.60 秒，99.91%
- 两次调用生成 token：730
- 单次 `max_new_tokens`：2048
- peak allocated VRAM：1937.27 MiB
- 服务验收后仍为 `active`

## 对抗性审查

- 同一本地标签命中两个全局锚点时，不做合并，标记 conflict；`required` 返回 422。
- 锚点缺失或覆盖不足时，只保留能唯一证明的映射，其余 segment 为 unresolved，scope 为 mixed。
- 后加入说话人先标为 `new_candidate`，只有在后续块作为锚点重新命中后才确认。
- 说话人暂时缺席不会删除其锚点，后续重新出现仍可回连。
- 锚点总预算为 60 秒，正文最多 1740 秒；不会把前缀叠加到 1800 秒正文后突破 1801 秒已验证边界。
- 锚点前缀计入 MOSS 动态 token 预算；锚点 transcript 被丢弃，正文时间戳减去前缀长度。
- 输出保留 `source_speaker=chunk-NNNN:Sxx`，不伪造数值置信度，也不把匿名 ID 宣称为真实姓名。

## 尚未覆盖

- 真实多人同时重叠说话。
- 强噪声、远场、严重串音下的锚点稳定性。
- 超过 4 人的真实验收；60 秒预算在默认 8 秒锚点下理论可容纳 7 人，但 `/v1/models` 只声明已真实通过的 4 人。
- 跨任务身份和姓名绑定；这需要显式 enrollment 或独立声纹系统，不属于 Anchor Replay。
