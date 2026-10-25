from __future__ import annotations

import argparse
import importlib
import json
import sys
from time import perf_counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qwen_asr_backend_smoke import (
    MODEL_REVISIONS,
    normalize_language,
    parsed_language,
    parsed_text,
    require_cuda_torch,
    resolve_torch_dtype,
)


from asr_server.registry import ALIGNER_SOURCE
from asr_server.adapters.qwen import _qwen_prefill_chat_template

ALIGNER_MODEL, ALIGNER_REVISION = ALIGNER_SOURCE


def run(args: argparse.Namespace) -> None:
    torch = require_cuda_torch()
    transformers = importlib.import_module("transformers")
    dtype = resolve_torch_dtype(torch, args.dtype)
    torch.cuda.reset_peak_memory_stats(args.device)
    started = perf_counter()
    asr_revision = args.asr_revision or MODEL_REVISIONS[args.asr_model]
    processor = transformers.AutoProcessor.from_pretrained(args.asr_model, revision=asr_revision)
    asr_model = (
        transformers.AutoModelForMultimodalLM.from_pretrained(
            args.asr_model,
            revision=asr_revision,
            dtype=dtype,
        )
        .to(args.device)
        .eval()
    )
    requested_language = normalize_language(args.language)
    request: dict[str, object] = {
        "audio": str(args.audio),
        "language": requested_language,
    }
    if args.prompt:
        request["prompt"] = args.prompt
    if requested_language is None:
        messages: list[dict[str, object]] = []
        if args.prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": args.prompt}]}
            )
        messages.append(
            {"role": "user", "content": [{"type": "audio", "audio": str(args.audio)}]}
        )
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        ).to(asr_model.device, asr_model.dtype)
    else:
        request["chat_template"] = _qwen_prefill_chat_template(processor)
        inputs = processor.apply_transcription_request(**request).to(asr_model.device, asr_model.dtype)
    with torch.inference_mode():
        output_ids = asr_model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    parsed = processor.decode(generated_ids, return_format="parsed")[0]
    transcript = parsed_text(parsed)
    language = parsed_language(parsed) or requested_language or ""
    if not transcript.strip():
        raise RuntimeError("ASR transcript is empty; forced alignment was not attempted")

    asr_seconds = perf_counter() - started
    align_started = perf_counter()
    aligner_processor = transformers.AutoProcessor.from_pretrained(
        ALIGNER_MODEL,
        revision=ALIGNER_REVISION,
    )
    aligner = (
        transformers.AutoModelForTokenClassification.from_pretrained(
            ALIGNER_MODEL,
            revision=ALIGNER_REVISION,
            dtype=dtype,
        )
        .to(args.device)
        .eval()
    )
    aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
        audio=str(args.audio),
        transcript=transcript,
        language=language,
    )
    aligner_inputs = aligner_inputs.to(aligner.device, aligner.dtype)
    with torch.inference_mode():
        outputs = aligner(**aligner_inputs)
    aligned = aligner_processor.decode_forced_alignment(
        logits=outputs.logits,
        input_ids=aligner_inputs["input_ids"],
        word_lists=word_lists,
        timestamp_token_id=aligner.config.timestamp_token_id,
    )[0]
    if not aligned:
        raise RuntimeError("forced aligner returned no items")
    if any(float(item["end_time"]) < float(item["start_time"]) for item in aligned):
        raise RuntimeError("forced aligner returned an invalid interval")
    alignment_seconds = perf_counter() - align_started

    print("asr_model:", args.asr_model)
    print("asr_revision:", asr_revision)
    print("aligner_model:", ALIGNER_MODEL)
    print("aligner_revision:", ALIGNER_REVISION)
    print("language:", language)
    print("transcript first 200:", transcript[:200])
    print("aligned item count:", len(aligned))
    print("aligned first 20:", json.dumps(aligned[:20], ensure_ascii=False))
    print("asr seconds:", round(asr_seconds, 3))
    print("alignment load+inference seconds:", round(alignment_seconds, 3))
    print("peak VRAM allocated MiB:", round(torch.cuda.max_memory_allocated(args.device) / 1024 / 1024, 2))
    print("peak VRAM reserved MiB:", round(torch.cuda.max_memory_reserved(args.device) / 1024 / 1024, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR followed by the official Qwen3 forced aligner.")
    parser.add_argument("--audio", type=Path, default=Path("test-fixtures/audio/test_short.wav"))
    parser.add_argument("--asr-model", default="Qwen/Qwen3-ASR-0.6B-hf", choices=sorted(MODEL_REVISIONS))
    parser.add_argument("--asr-revision")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cuda:0"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    run(args)


if __name__ == "__main__":
    main()
