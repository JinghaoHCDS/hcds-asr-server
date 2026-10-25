---
name: hcds-lan-asr
description: "Use an existing HCDS ASR Server to transcribe local audio or video, obtain real alignment or anonymous speaker segments, and run resumable client-side batches through its HTTP API."
metadata:
  version: "1.4.0"
  validated-client: "macOS with Python 3.12+; other client platforms require validation"
  hermes:
    tags: [asr, transcription, batch, local-network]
---

# HCDS ASR Server — Official Agent Skill

<!-- deployment-address -->

Use this skill to call an existing HCDS ASR Server. Inference runs on the server; the client needs no GPU packages or model downloads. This guide belongs to the server version that serves it. Runtime availability and capabilities always come from `GET /v1/models`.

## Find the server and its instructions

The server's ZIP automatically includes its deployment address in `server.json` beside SKILL.md. The bundled client's priority is `--base-url`, then `ASR_BASE_URL`, then that bundled `server.json`, then `~/.Agent/state/lan-asr/config.json`, then `http://127.0.0.1:18080`. A deployment's package takes precedence over an older machine-wide default; an explicit argument or environment variable can still override it. When installing or moving the Skill, keep `server.json` alongside SKILL.md and `scripts/`. Keep machine-specific addresses out of the repository template. Never substitute another person's IP or a public transcription service.

For a freshly downloaded package, no IP configuration is needed. Its address comes from the service's configured `ASR_PUBLIC_BASE_URL`, or from the actual download URL when no usable LAN address is configured. The repository copy and standalone script download have no bundled server.json; supply the server address through the available configuration methods. A loopback address refers to the client's own machine. If the service moves or its IP changes, download a fresh package or explicitly override its address.

Relative HTTP links in this guide use that server as their origin:

- `/skill`: this official Markdown guide.
- `/skill/hcds-lan-asr.zip`: installable folder containing this guide and `scripts/asr_batch.py`.
- `/skill/asr_batch.py`: the same dependency-free Python batch client as plain source.
- `/openapi.json`: request parameters, response schemas and error envelopes.
- `/docs`: interactive API documentation; `/docs/mcp`: client-side stdio MCP setup.

For an installed copy, read the target server's `/skill` when reachable before starting a new integration. If it returns 404 on an older deployment, use `/openapi.json`, `/docs/mcp` and `/v1/models`, and report the version difference. A network/authentication failure is not permission to switch services or change authentication. Inspect downloaded code before running it; reading Markdown does not install or execute the client.

All LAN calls must bypass environment proxies. For curl use `--noproxy '*'`; the bundled client connects directly. When `ASR_API_TOKEN` is enabled on the server, send `Authorization: Bearer <token>` to `/v1/**`. The client reads `ASR_TOKEN` from its environment. Never print it, put it in the batch database, or commit it. Documentation and `/health` are public; documentation access does not authorize GPU or administrative operations beyond the user's task.

## Choose capabilities before submitting

Read `/health` and `/v1/models`. Select a returned model ID that satisfies the actual task; an explicit model or capability request must not silently downgrade.

| Task | Required runtime declarations | Request |
| --- | --- | --- |
| Plain transcript | `capabilities.transcription=true` | `purpose=transcription` |
| Oral-video timing, word/character locations, pause or speaking-rate evidence | `forced_alignment=true` and the requested `timestamps` value | `purpose=oral_analysis`, `response_format=verbose_json`, `timestamps=auto` or a declared explicit granularity |
| Anonymous speaker segments | `diarization=true` and the requested `speaker_resolution_modes` | `purpose=podcast`, `response_format=verbose_json`, `speaker_resolution=auto` or `required` |

Qwen 1.7B is the usual default for transcription and alignment; Qwen 0.6B is the smaller alternative. MOSS (`moss-transcribe-diarize-0.9b`) is optional and currently provides speaker segments. These are examples, not a substitute for discovery. Check declared `languages`, `backends`, `pipelines`, `context_prompt` and token limits too. Current production backends declare `transformers`; parser acceptance of `vllm` is not support.

`quality_profile=legacy` remains the `/v1/**` default. For Qwen, request `quality_profile=quality_v1` only when its `pipelines` includes it and forced alignment is enabled. It preserves the source timeline and performs bounded review; `validated=false` means long-recording accuracy has not passed the quality gate. The usual automatic budget is 4096 for quality_v1; explicit `max_new_tokens` stays explicit, subject to the declared limit. MOSS uses legacy; high-speech-rate material may need an explicit higher budget. `generation_truncated` is a failure, not a complete transcript.

## Submit, poll and preserve results

Prefer asynchronous jobs for long material or batch work. One request uploads one local file, not a remote URL or a base64 payload. Keep the source available for recovery.

1. `POST /v1/audio/transcription-jobs` with multipart `file` and the selected request fields. Record the returned `id` and the request profile.
2. Poll `GET /v1/jobs/{id}?detail=compact` at a modest interval (for example 2–10 seconds). Progress is service phases and actual completed chunks; do not invent progress inside a model call.
3. On `completed`, fetch `GET /v1/jobs/{id}?detail=full` once for the final result. Preserve `result.text` and the complete result JSON, including warnings and provenance.
4. Mark the local item complete only after its output files have been successfully written. On failure preserve the source and error; report failed items as failures.
5. `DELETE /v1/jobs/{id}` requests cancellation. A running model call may finish before cancellation takes effect; a cancellation acknowledgement is not a completed transcript.

For a direct HTTP integration, after setting `ASR_BASE_URL` and choosing `MODEL` from discovery:

```bash
curl --noproxy '*' --connect-timeout 5 --max-time 30 --fail-with-body \
  "$ASR_BASE_URL/v1/models"

curl --noproxy '*' --connect-timeout 5 --max-time 300 --fail-with-body \
  "$ASR_BASE_URL/v1/audio/transcription-jobs" \
  -F 'file=@/absolute/path/to/audio.wav' -F "model=$MODEL" \
  -F 'purpose=transcription' -F 'language=auto' -F 'quality_profile=legacy'

# Replace JOB_ID with the creation response's id.
curl --noproxy '*' --connect-timeout 5 --max-time 30 --fail-with-body \
  "$ASR_BASE_URL/v1/jobs/JOB_ID?detail=compact"
curl --noproxy '*' --connect-timeout 5 --max-time 30 --fail-with-body \
  "$ASR_BASE_URL/v1/jobs/JOB_ID?detail=full"
```

When authentication is enabled, add `-H "Authorization: Bearer $ASR_TOKEN"` to each `/v1/**` request; keep shell tracing off. A short synchronous alternative is `POST /v1/audio/transcriptions`: a 200 result uses top-level `text`, while a 202 response is a job and follows the polling lifecycle above. Synchronous calls also enter the shared FIFO and can wait behind other work.

## Use the bundled batch client

The Skill is named `hcds-lan-asr`. For upgrade compatibility, local state and submission idempotency retain the legacy `lan-asr` namespace; changing the Skill folder does not move or reset existing batches.

Requires Python 3.12+ and its standard library. Run the `scripts/asr_batch.py` shipped alongside this SKILL.md from its actual install directory. For the existing shared installation, it is `~/.Agent/skills/hcds-lan-asr/scripts/asr_batch.py`; a repository checkout uses `skills/hcds-lan-asr/scripts/asr_batch.py`. No profile-specific installation is required.

Confirm the source and output directories when the user has not supplied them. The default output is `<source>/asr-transcripts`. Pass a directory, optionally restricted with repeatable `--include` patterns:

```bash
python3 /path/to/hcds-lan-asr/scripts/asr_batch.py run /absolute/path/to/media \
  --output-dir /absolute/path/to/transcripts \
  --purpose transcription --language auto --quality-profile legacy
```

The client discovers a compatible default model when `--model` is omitted; use `--model <returned-id>` to pin one. For aligned oral material use `--purpose oral_analysis --timestamps auto`; add `--quality-profile quality_v1` when supported. For speaker segments use `--purpose podcast` with a declared diarization model. `--response-format verbose_json`, `--speaker-resolution auto|required`, `--context`, `--hotwords`, and `--max-new-tokens` allow explicit choices. Context and vocabulary are references, not instructions to invent speech or identities.

`run` prints a batch ID. The local SQLite database defaults to `~/.Agent/state/lan-asr/batches.sqlite`; `--db` selects another local database. Existing batches remain readable. Request options, including the selected server, are frozen per batch and reused by `resume`; a later config change does not redirect an existing batch. Older batches have no original SHA256, so initial recovery can only compare their saved size/mtime before establishing a digest:

```bash
python3 /path/to/hcds-lan-asr/scripts/asr_batch.py status --batch BATCH_ID
python3 /path/to/hcds-lan-asr/scripts/asr_batch.py resume BATCH_ID
python3 /path/to/hcds-lan-asr/scripts/asr_batch.py cancel BATCH_ID
```

Outputs are `<original-filename>.txt` and `<original-filename>.asr.json` under the output directory, preserving subdirectories. This local batch state is separate from server job persistence. A lost `/v1` job may be resubmitted once with unchanged source and frozen options; recovery does not reuse server checkpoints. Report the batch ID, model, purpose/profile, completed/failed counts, output directory and unresolved errors.

## Interpret timing and speaker evidence

- Qwen fine timestamps come from the official ForcedAligner. Consume actual `granularity`, `source` and `status`; never distribute sentence or chunk duration evenly over text. Alignment does not prove the transcript is correct.
- MOSS provides real segment-level times, not word/character forced alignment. Speaker IDs are anonymous and task-local. Inspect `execution.speaker_scope`, `diarization.status` and per-segment scopes; only `global` supports whole-item consistent labels. Preserve `mixed`, `partial`, chunk namespaces and unresolved speakers. Anchor Replay supports at most four resolved identities, not verified real names.
- `timing.reading_segments` are reading groups, not verified semantic sentences. Acoustic pause candidates are not evidence of deliberate pauses.
- Preserve the original model output, warnings and correction provenance; do not fill uncertain speech from context. Six hours is an input/resource limit, not a six-hour accuracy guarantee.

## Persistence, limits and errors

All inference entry points share one bounded FIFO and one real model invocation at a time. More concurrent submissions do not create GPU batching. The default `/v1` upload cap is 512 MiB, the audio-duration cap six hours, and the memory-result TTL 3600 seconds; server configuration can change limits. Active and waiting operations share the queue budget. Temporary upload files are cleaned after the operation's safe terminal cleanup.

`/v1/**` jobs are in memory and disappear on process restart. The browser workbench `/api/**` separately uses individual HttpOnly sessions, SQLite tasks/checkpoints and seven-day terminal retention, with an 8 GiB default upload cap. A Bearer token does not grant access to a person's workbench tasks. Selecting quality_v1 on `/v1` does not make its jobs durable. Use workbench endpoints only for an explicitly requested, authenticated workbench integration, using the served OpenAPI contract.

Errors use `{error: {code, message, details}}`. Handle `401 unauthorized` by correcting credentials; `422 capability_not_supported` by explaining the unmet requirement; `generation_truncated` by reporting failure and reviewing the explicit budget; `413` by checking limits; `429 job_queue_full` with bounded backoff, never a synchronous bypass. On `404 job_not_found`, distinguish restart/expiry from success. Retry transient failures only within a finite budget and never silently change a pinned model, profile or required capability.

MCP is an alternative transport: `/docs/mcp` describes its client-side stdio bridge, local allowed roots and eight tools. Keep runtime capability discovery. Do not install GPU runtimes on a client, expose new server ports, invoke a WSL direct-GPU batch path, or claim streaming, vLLM, standalone `/v1/audio/alignments`, distributed inference or unvalidated hardware support.
