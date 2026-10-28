# MOSS 长音频 model-aware 执行策略 WSL 验收记录

> 历史摘要：以下结果属于原日期、环境和版本，本轮未重新验证。原始验收附件已按维护决定从当前树删除，未另存副本；Git 历史未改写。
> 本页短样本结果对应已退出的 13.06 秒旧录音；同路径现改为合成样例，不能用它复现旧样本的质量或计时数字。

> 后续 opt-in 跨块 speaker 解析曾使用 Anchor Replay 完成四说话人验收，见[Anchor Replay 历史摘要](2026-07-15-wsl-moss-anchor-replay.md)。本文的 chunk-local 结论仍适用于默认 `speaker_resolution=off`。

## 结论

`moss-transcribe-diarize-0.9b` 在当前 RTX 5070 Ti 16GB 与固定 snapshot 上，10 分钟和 30 分钟单次 native long-form 可以完整覆盖尾部并保留全局 `S01/S02`。60 分钟和 90 分钟单次推理不会立即 OOM，但模型都在约 3044 秒停止输出；90 分钟还接近耗尽 16GB 显存。因此不能把“请求完成生成”当作“转写完整”。

服务的生产 `auto` 策略据此收敛为：不超过 1801 秒走 native long-form；更长音频明确降级为 1800 秒 fixed chunks。显式 `split_strategy=none` 仍可用于研究性挑战，但缺尾会返回 `422 incomplete_transcript`，不会再返回表面 `completed` 的残缺正文。

## 环境

- 日期：2026-07-15
- 服务目录：`/path/to/user/services/asr-server`
- API：`http://YOUR_WINDOWS_LAN_IP:18080`
- GPU：NVIDIA GeForce RTX 5070 Ti 16GB，compute capability `(12, 0)`
- torch：`2.11.0+cu128`
- torch CUDA：`12.8`
- transformers：`5.14.0.dev0`
- 模型：`OpenMOSS-Team/MOSS-Transcribe-Diarize`
- revision：`d7231bbae2587a4af278735eb765b318c4f64edd`
- 后端：`transformers`
- 语言：`auto`

原始播客约 5406.07 秒、86,498,322 bytes。原始音频、完整逐字稿和响应 JSON 均未加入 Git；下列记录只保留可复查指标。

## 阶梯实测

| 输入 | 执行模式 | 结果 | 推理耗时 | 生成 token / 上限 | 峰值 allocated VRAM | segment / speaker | 尾部覆盖 |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| 10 分钟 | native | 完成 | 83.09 秒 | 3426 / 7201 | 3028.19 MiB | 66 / `S01,S02` | 599.98 / 600.01，99.995% |
| 30 分钟 | native | 完成 | 325.29 秒 | 9706 / 21601 | 5531.08 MiB | 158 / `S01,S02` | 1799.99 / 1800.02，99.998% |
| 60 分钟 | explicit `none` | 旧实现表面完成，实际缺尾 | 706.39 秒 | 16477 / 43201 | 9006.26 MiB | 243 / `S01,S02` | 3044.37 / 3600.01，84.57% |
| 90 分钟 | explicit `none` | `422 incomplete_transcript` | 约 2415.47 秒 | 未触及 64873 上限 | 进程显存约 15.95–15.97GB | 最后 segment 3044.39 秒 | 缺失 2361.68 秒 |
| 90 分钟 | `auto` fallback | 完成 | 1058.67 秒 | 合计 31271 / 单次上限 21600 | 5850.11 MiB | 543 / 8 个 chunk-scoped 标签 | 5405.88 / 5406.07，99.9965% |

10 分钟 job 为 `job_e5d245d206f6481fba7039a8c5b09e80`，总耗时 83.89 秒；30 分钟 job 为 `job_eae873f4b15945588e84ed92ec09d2ad`，总耗时 327.16 秒。两次均为单 chunk、`execution.mode=native_long_form`、`speaker_scope=global`，且没有在旧 120 秒边界出现 chunk speaker 命名空间重置。

60 分钟结果是在尾部保护加入前取得的反例：生成 token 只用了上限的 38%，HTTP/job 仍曾返回完成，但最后 555.64 秒没有任何 segment。这个反例证明只检测 `generated_tokens >= max_new_tokens` 不足以判断完整性。

90 分钟 explicit native job 为 `job_f234c2077b4d4553b7fcc8f0d810efef`。加入尾部保护后，同样的模型停止位置被识别为稳定缺尾边界，并返回：

```json
{
  "error": {
    "code": "incomplete_transcript",
    "details": {
      "audio_duration_seconds": 5406.069875,
      "last_segment_end_seconds": 3044.39,
      "uncovered_tail_seconds": 2361.679875,
      "allowed_tail_seconds": 108.1213975,
      "recommended_action": "retry with split_strategy=fixed and a validated chunk duration"
    }
  }
}
```

该错误是受控 422，MOSS worker 没有被当成崩溃进程杀掉，随后仍可接收自动降级任务。

## 自动降级验收

最终版本的 90 分钟 `split_strategy=auto` job：`job_6afb3b83d0624af1b42ecbca88225956`。

此前曾提交 `job_3d4a25c00649433d8d5b91b990d18531`，但它运行在 90 分钟 explicit native 失败后的旧 worker 上：CUDA allocator 保留约 15.9GB，第一块超过 11 分钟仍未完成。该污染样本被中止，促成了逐调用 CUDA cache cleanup；最终验收从重启后的干净 worker 重新执行。

最终结果：

- service timing 总耗时 1068.98 秒，其中模型 load 4.51 秒、解码 5.57 秒、推理 1058.67 秒。
- `execution.mode=chunked`、`automatic_chunk_fallback=true`、fallback reason 为 `duration_exceeds_validated_native_limit`。
- 4 个 fixed windows：前三个约 1800 秒，最后一个为含 overlap 的约 12 秒尾块；真实 job 进度依次为 25%、50%、75%、100%。
- 合计 prompt token 71,661、generated token 31,271；单次 `max_new_tokens=21600`，`truncated=false`。
- 峰值 allocated VRAM 5850.11 MiB；逐调用 CUDA cache cleanup 后，未复现旧 worker 长期保留约 15.9GB 的退化。
- 返回 36,818 字、543 个 segments。speaker 为 8 个明确的 chunk-scoped 标签：每个 chunk 各有 `S01/S02`，没有宣称是两位全局身份。
- 最后 segment 为 5405.88 秒，原始时长 5406.069875 秒，coverage ratio 0.9999648774。
- warnings 仅包含 `moss_native_long_form_fallback:duration_exceeds_validated_native_limit` 和 `moss_speaker_labels_are_chunk_local`。

这条 fallback 比 90 分钟 explicit native 少约 22.4 分钟墙钟时间，同时避免近满显存和 43.7% 尾部缺失；代价是 speaker identity 只能保证 chunk-local。

## 最终 LAN 与资源回收

Mac mini 使用 `curl --noproxy '*'` 访问 `http://YOUR_WINDOWS_LAN_IP:18080`：health 正常；13.06 秒短音频返回 native/global、非空文本、2 个 segments、56/2048 generated tokens、97.01% segment tail coverage，且 warnings 为空。

90 分钟 fallback 完成后，GPU 进程显存从推理期峰值回落到约 3.2GB；显式卸载 MOSS 后整卡 used memory 为约 1.0GB，服务仍为 `active`。本次复制到 `/path/to/user/asr-validation/moss-long-form` 的原始及裁剪音频已删除。

最终版本在 Mac 与 WSL 均为 139 tests passed、2 skipped；mypy 检查 47 个源码文件无错误。WSL CUDA 复验仍为 torch `2.11.0+cu128`、CUDA `12.8`、RTX 5070 Ti capability `(12, 0)`。

## 对抗性审查结论

1. 16GB 显存不是唯一边界。60 分钟在约 9GB allocated VRAM 下仍稳定缺尾，说明输出/上下文行为必须独立验收。
2. “未达到 `max_new_tokens`”不代表完整。必须同时检查可解析 segment 的尾部覆盖。
3. 单次 native 调用内部没有可信进度，job 只能报告真实阶段；自动降级后才能报告 chunk 完成数。
4. 30 分钟 native speaker 可以声明为本次音频全局范围；自动分块 speaker 只能声明 chunk-local。当前没有证据支持用重叠文本或声纹 embedding 自动拼接全局身份，因此未实现 stitching。
5. `/v1/models` 只声明实测边界：`validated_native_max_seconds=1801` 是对 1800.02 秒成功样本的一秒容器容差，不是 31 分钟能力声明。
6. 90 分钟挑战结束后，CUDA allocator 曾把进程显存保留在约 15.9GB，后续调用虽能复用但长期贴近上限。适配器现于每次调用提取完峰值指标和文本后执行 CUDA cache cleanup，模型本身继续热加载；成功、生成截断和推理异常路径都有测试覆盖。

## 客户端稳定契约

客户端先读取 `/v1/models`，再从每次结果消费以下字段：

- `execution.mode`
- `execution.speaker_scope`
- `execution.automatic_chunk_fallback`
- `execution.fallback_reason`
- `generation.generated_tokens`
- `generation.max_new_tokens`
- `generation.segment_coverage_ratio`
- `warnings`

只有 `speaker_scope=global` 时，客户端才可以把 `S01/S02` 当作整段一致身份。`speaker_scope=chunk` 时必须保留 `chunk-NNNN:` 前缀，并向用户展示这是分块说话人标签。
