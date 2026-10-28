from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr_server.adapters.qwen import _qwen_prefill_chat_template
from asr_server.registry import MODEL_SOURCES
from scripts.validation_output import add_report_arguments, emit_report, safe_language, text_metrics, validate_report_output


DEFAULT_MODEL = MODEL_SOURCES["qwen3-asr-0.6b"][0]
DEFAULT_AUDIO = "test-fixtures/audio/test_short.wav"
MODEL_REVISIONS = {repository: revision for model_id, (repository, revision) in MODEL_SOURCES.items()
                   if model_id in {"qwen3-asr-0.6b", "qwen3-asr-1.7b"}}


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


def normalize_language(language: str) -> str | None:
    return None if language == "auto" else language


def resolve_torch_dtype(torch: Any, dtype: str) -> Any:
    normalized = dtype.lower()
    if normalized == "auto" or normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    raise ValueError("dtype must be auto, bfloat16/bf16, or float16/fp16")


def parsed_text(result: object) -> str:
    if isinstance(result, dict):
        text = result.get("transcription", result.get("text", ""))
        return text if isinstance(text, str) else str(text)
    text = getattr(result, "text", "")
    if not isinstance(text, str):
        return str(text)
    return text


def parsed_language(result: object) -> str:
    if isinstance(result, dict):
        language = result.get("language", "")
        return language if isinstance(language, str) else "" if language is None else str(language)
    language = getattr(result, "language", "")
    if not isinstance(language, str):
        return str(language)
    return language


def print_result_summary(args: argparse.Namespace, *, language: str, text: str) -> None:
    if not text.strip():
        raise RuntimeError(f"hf-native {args.backend} 后端转录结果为空")
    emit_report({"model": args.model, "backend": args.backend, "loader": "hf-native",
                 "language": safe_language(language), **text_metrics(text)}, output=getattr(args, "output", None),
                content={"text": text}, include_content=getattr(args, "include_content", False))


def run_hf_native_transformers(args: argparse.Namespace) -> None:
    torch = require_cuda_torch()
    transformers = importlib.import_module("transformers")
    print("transformers:", getattr(transformers, "__version__", "unknown"))
    processor_cls = getattr(transformers, "AutoProcessor", None)
    model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
    if processor_cls is None or model_cls is None:
        raise RuntimeError("当前 transformers 不包含 AutoProcessor/AutoModelForMultimodalLM；请升级 release 或安装 transformers main")
    revision = args.revision or MODEL_REVISIONS.get(args.model)
    if revision is None:
        raise RuntimeError("unknown model requires an explicit --revision")
    processor = processor_cls.from_pretrained(args.model, revision=revision)
    model = model_cls.from_pretrained(args.model, revision=revision, dtype=resolve_torch_dtype(torch, args.dtype)).to(args.device).eval()
    apply_request = getattr(processor, "apply_transcription_request", None)
    if not callable(apply_request):
        raise RuntimeError("processor.apply_transcription_request 不可用；停止，避免临时字符串 prompt 绕过")
    request: dict[str, object] = {
        "audio": str(args.audio),
        "language": normalize_language(args.language),
    }
    if args.prompt:
        request["prompt"] = args.prompt
    if normalize_language(args.language) is None:
        messages: list[dict[str, object]] = []
        if args.prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": args.prompt}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "audio", "audio": str(args.audio)}],
            }
        )
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        ).to(model.device, model.dtype)
    else:
        request["chat_template"] = _qwen_prefill_chat_template(processor)
        inputs = apply_request(**request).to(model.device, model.dtype)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    parsed = processor.decode(generated_ids, return_format="parsed")[0]
    print_result_summary(
        args,
        language=parsed_language(parsed) or normalize_language(args.language) or "",
        text=parsed_text(parsed),
    )
    if args.reject_hallucinated_term and args.reject_hallucinated_term.casefold() in parsed_text(parsed).casefold():
        raise RuntimeError("negative Context check failed: absent keyword appeared in transcript")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WSL 侧 Qwen3-ASR 最小后端验收脚本；先跑通它，再开发服务端 adapter。",
    )
    parser.add_argument("--backend", required=True, choices=["transformers"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--audio", type=Path, default=Path(DEFAULT_AUDIO))
    parser.add_argument("--language", default="auto")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cuda:0"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--reject-hallucinated-term")
    add_report_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        validate_report_output(args.output, include_content=args.include_content)
        if not args.audio.is_file():
            raise ValueError("audio must be a file")
        run_hf_native_transformers(args)
    except Exception as exc:
        raise SystemExit(f"Qwen backend smoke failed ({type(exc).__name__})") from None


if __name__ == "__main__":
    main()
