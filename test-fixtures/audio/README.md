# Synthetic audio test fixture

`test_short.wav` is synthetic English speech for engineering smoke tests. It replaces the former user-provided 13.06-second recording. The committed file passed a short Qwen workbench GPU smoke on 2026-10-28; this is not general recognition-quality acceptance.

| Property | Value |
| --- | --- |
| Format | WAV, PCM signed 16-bit, mono, 16,000 Hz |
| Duration | 10.105375 seconds |
| Frames | 161686 |
| SHA-256 | `56717d7e62e209e426d939d33334643cc682fac6b0bbd1f436cea090b37e5ad1` |
| Text | [Original reference](test_short.reference.txt) |
| Provenance | [Machine-readable source record](test_short.source.json) |
| Synthesis | Flite 2.2, stock `kal` voice |
| Licensing | Project material under [MIT](../../LICENSE), with [Flite and voice notices](../../THIRD_PARTY_NOTICES.md) retained |

No private recording or user voice clone is used. The original reference was authored for this repository. Flite and the voice retain their upstream terms; MIT does not relicense third-party components.

## Regeneration

The fixture is committed, so ordinary installation and CPU tests do not need a synthesizer. Regeneration is an optional local maintenance operation; no ASR model or CUDA dependency is installed.

Use the unmodified Flite source at `e9e2e37c329dbe98bfeb27a1828ef9a71fa84f88` (tag `v2.2`). The downloaded source archive SHA-256 was `1eebf476797cea10b4706cfbb9db9ba34f7b8ccb1fe7e0069b0b7528e0b0be46`. Build outside the tracked source tree:

```bash
git clone https://github.com/festvox/flite /path/to/flite
git -C /path/to/flite checkout e9e2e37c329dbe98bfeb27a1828ef9a71fa84f88
(cd /path/to/flite && ./configure --with-audio=none --with-vox=cmu_us_kal && make -j2)
uv run --frozen python -m scripts.generate_test_audio --flite /path/to/flite/bin/flite --replace
```

The generator loads only the reference text, asks the local `kal` voice to write a WAV, and converts 8 kHz PCM to 16 kHz using integer linear interpolation. It writes a standard WAV header and a content/source record without host paths or timestamps. The selected engine build produces repeatable output locally; different compilers/platforms still require comparing the resulting digest rather than assuming byte identity.

## Validation scope

The file format, source record, reference digest and local mock workflow are checked. A real HTTP workbench smoke with Qwen3-ASR 1.7B and the official ForcedAligner passed on WSL: 28 normalized reference words, zero word edits, aligned timestamps, all four exports, MP3 Range playback, and realignment after an edit. The temporary test task was deleted. See the [version-bound validation record](../../docs/validation/2026-10-28-repository-governance.md).

This short engineering result does not validate timestamp accuracy by human review, listening quality, MOSS, long-form quality or sustained operation. Historical timings, transcripts and quality measurements used the retired recording and cannot be reproduced by substituting this file. Regeneration writes an unvalidated source record for the newly generated output; retain acceptance claims only with matching audio and version evidence.

```bash
uv run --frozen python scripts/asr_client.py --base-url http://127.0.0.1:18080 transcribe test-fixtures/audio/test_short.wav --model qwen3-asr-1.7b --backend auto
```

Run the real-model example only against an explicitly authorized server. Long-media CPU regressions generate their own synthetic audio; real long-form quality work uses separately permitted material outside the repository.
