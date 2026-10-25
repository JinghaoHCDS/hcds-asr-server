from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr_server.registry import MOSS_MODEL_ID, MODEL_SOURCES


DEFAULT_MODEL, DEFAULT_REVISION = MODEL_SOURCES[MOSS_MODEL_ID]
DEFAULT_AUDIO = "test-fixtures/audio/test_short.wav"


def require_cuda_torch() -> Any:
    torch = importlib.import_module("torch")
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.version.cuda is None:
        raise RuntimeError("当前环境是 CPU 版 torch，停止；请安装 PyTorch CUDA wheel 后重试")
    if not torch.cuda.is_available():
        raise RuntimeError("torch 看不到 CUDA，停止；请检查 WSL NVIDIA 驱动和 CUDA wheel")
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    return torch


def resolve_torch_dtype(torch: Any, dtype: str) -> Any:
    normalized = dtype.lower()
    if normalized == "auto" or normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    raise ValueError("dtype must be auto, bfloat16/bf16, or float16/fp16")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def segment_to_dict(segment: object) -> dict[str, object]:
    if isinstance(segment, dict):
        return {
            "start": segment.get("start"),
            "end": segment.get("end"),
            "speaker": segment.get("speaker"),
            "text": segment.get("text"),
        }
    return {
        "start": getattr(segment, "start", None),
        "end": getattr(segment, "end", None),
        "speaker": getattr(segment, "speaker", None),
        "text": getattr(segment, "text", None),
    }


def result_int(result: object, key: str) -> int | None:
    value = result.get(key) if isinstance(result, dict) else getattr(result, key, None)
    return int(value) if value is not None else None


def run_smoke(args: argparse.Namespace) -> None:
    torch = require_cuda_torch()
    transformers = importlib.import_module("transformers")
    moss_package = importlib.import_module("moss_transcribe_diarize")
    inference_utils = importlib.import_module("moss_transcribe_diarize.inference_utils")
    print("transformers:", getattr(transformers, "__version__", "unknown"))
    print("moss-transcribe-diarize:", package_version("moss-transcribe-diarize"))

    processor_cls = getattr(transformers, "AutoProcessor", None)
    model_cls = getattr(transformers, "AutoModelForCausalLM", None)
    if processor_cls is None or model_cls is None:
        raise RuntimeError("当前 transformers 不包含 AutoProcessor/AutoModelForCausalLM；请安装 MOSS 依赖")

    dtype = resolve_torch_dtype(torch, args.dtype)
    device = torch.device(args.device)
    model = (
        model_cls.from_pretrained(args.model, revision=args.revision, trust_remote_code=True, dtype="auto")
        .to(dtype=dtype)
        .to(device)
        .eval()
    )
    processor = processor_cls.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    language_instruction = {
        "zh": "请仅使用中文转写。",
        "en": "Transcribe in English only.",
    }.get(args.language, "")
    prompt = "\n".join(part for part in (args.prompt, language_instruction) if part)
    messages = inference_utils.build_transcription_messages(str(args.audio), prompt=prompt)
    torch.cuda.reset_peak_memory_stats(device)
    result = inference_utils.generate_transcription(
        model,
        processor,
        messages,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        device=device,
        dtype=dtype,
    )
    text = result.get("text", "") if isinstance(result, dict) else getattr(result, "text", "")
    text = text if isinstance(text, str) else str(text)
    segments = moss_package.parse_transcript(text)
    prompt_tokens = result_int(result, "prompt_len")
    generated_tokens = result_int(result, "generated_tokens")
    peak_vram_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    print("model:", args.model)
    print("revision:", args.revision)
    print("language request:", args.language)
    print("backend: transformers")
    print("loader: hf-remote-code")
    print("max_new_tokens:", args.max_new_tokens)
    print("prompt tokens:", prompt_tokens)
    print("generated tokens:", generated_tokens)
    print("peak VRAM allocated MiB:", round(peak_vram_allocated_mb, 2))
    print("text first 200:", text[:200])
    print("parsed segment count:", len(segments))
    if segments:
        print("first segment:", segment_to_dict(segments[0]))
        print("last segment:", segment_to_dict(segments[-1]))
    if not text.strip():
        raise RuntimeError("MOSS transformers 后端转录结果为空")
    if generated_tokens is not None and generated_tokens >= args.max_new_tokens:
        raise RuntimeError("MOSS generation reached max_new_tokens; transcript may be truncated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WSL 侧 MOSS-Transcribe-Diarize 最小后端验收脚本；先跑通它，再开启 ASR_ENABLE_MOSS。",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--audio", type=Path, default=Path(DEFAULT_AUDIO))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cuda:0"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--language", default="auto", choices=["auto", "zh", "en"])
    parser.add_argument(
        "--prompt",
        default=(
            "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
            "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.exists():
        raise FileNotFoundError(args.audio)
    run_smoke(args)


if __name__ == "__main__":
    main()
