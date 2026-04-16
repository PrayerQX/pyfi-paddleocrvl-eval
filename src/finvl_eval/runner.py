from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from .models import build_adapter, normalize_prediction
from .prompts import build_mcq_prompt
from .pyfi import iter_jsonl, iter_pyfi_csv
from .records import EvalRecord
from .scoring import aggregate
from .security import SecurityContext, create_security_context
from .veto import VetoConfig


def iter_records(dataset: str | Path, dataset_format: str) -> Iterable[EvalRecord]:
    if dataset_format == "pyfi-csv":
        return iter_pyfi_csv(dataset)
    if dataset_format == "jsonl":
        return iter_jsonl(dataset)
    raise ValueError(f"Unknown dataset format: {dataset_format}")


def run(args: argparse.Namespace) -> dict:
    # Build security context from CLI flags
    security_ctx = create_security_context(
        enable_pii_masking=args.pii_masking,
        enable_audit_log=args.audit_log is not None,
        enable_encryption=args.encrypt_artifacts,
        encryption_passphrase=getattr(args, "encryption_passphrase", None),
        audit_log_path=args.audit_log,
        strict_pii=args.pii_strict,
    )

    # Build veto config from CLI flags
    veto_config = VetoConfig(
        enabled=args.enable_veto,
        threshold=args.veto_threshold,
        enable_evidence_check=not args.veto_no_evidence_check,
        enable_contradiction_check=not args.veto_no_contradiction,
    )

    # Attach to args so build_adapter can pick them up
    args._security_ctx = security_ctx
    args._veto_config = veto_config

    adapter = build_adapter(args)
    records_iter = iter_records(args.dataset, args.format)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    processed = 0
    try:
        with output_path.open("w", encoding="utf-8") as f:
            for record in records_iter:
                if args.limit is not None and processed >= args.limit:
                    break
                image_path = record.resolved_image_path(args.images_root)
                if args.require_image and not image_path.exists():
                    continue

                prompt = build_mcq_prompt(record, context_mode=args.context_mode)
                try:
                    raw_prediction = adapter.predict(record, image_path, prompt)
                    error = None
                except Exception as exc:
                    raw_prediction = None
                    error = f"{type(exc).__name__}: {exc}"

                prediction = normalize_prediction(record, raw_prediction)
                item = {
                    "uid": record.uid,
                    "image_path": str(image_path),
                    "capability": record.capability,
                    "complexity": record.complexity,
                    "answer": record.answer,
                    "prediction": prediction,
                    "raw_prediction": raw_prediction,
                    "correct": prediction == record.answer if record.answer else False,
                    "error": error,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                results.append(item)
                processed += 1
                if args.progress_every and processed % args.progress_every == 0:
                    print(f"Processed {processed} records", file=sys.stderr)
    finally:
        # Close audit logger
        if security_ctx.audit_logger is not None:
            security_ctx.audit_logger.close()

    metrics = aggregate(results)
    metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM/document parsing models on PyFi-like MCQ data.")
    parser.add_argument("--dataset", required=True, help="CSV or JSONL dataset path")
    parser.add_argument("--format", choices=["pyfi-csv", "jsonl"], default="jsonl")
    parser.add_argument("--images-root", help="Root containing the images/ directory")
    parser.add_argument("--out", default="runs/eval.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-image", action="store_true")
    parser.add_argument("--context-mode", default="image_background_only")
    parser.add_argument("--progress-every", type=int, default=25)

    parser.add_argument(
        "--model",
        choices=[
            "first-option",
            "random-option",
            "openai-compatible-vlm",
            "paddleocr-text-docqa",
            "paddleocr-vl-docqa",
            "paddleocr-vl-hybrid-docqa",
            "paddleocr-vl-grounded-docqa",
            "paddleocr-vl-boosted-docqa",
        ],
        default="first-option",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--openai-model", default="gpt-4.1")
    parser.add_argument("--selector-model", help="Text-only selector model for PaddleOCR-VL parsed markdown")
    parser.add_argument("--selector-base-url", help="OpenAI-compatible selector base URL")
    parser.add_argument("--selector-max-tokens", type=int, help="Selector max_tokens or max_completion_tokens value")
    parser.add_argument(
        "--selector-use-max-completion-tokens",
        action="store_true",
        default=None,
        help="Send selector token limit as max_completion_tokens instead of max_tokens",
    )
    parser.add_argument(
        "--selector-stream",
        action="store_true",
        default=None,
        help="Use streaming chat completions for selector models",
    )
    parser.add_argument("--selector-extra-body-json", help="JSON object passed as extra_body to selector calls")
    parser.add_argument("--artifacts-dir", help="Directory for PaddleOCR-VL markdown/json artifacts")
    parser.add_argument("--ocr-artifacts-dir", help="Directory for traditional PaddleOCR text/json artifacts")
    parser.add_argument("--paddle-vl-backend", help="PaddleOCR VL recognition backend")
    parser.add_argument("--paddle-vl-server-url", help="PaddleOCR VL recognition server URL")
    parser.add_argument("--paddle-vl-model-dir", help="Local PaddleOCR-VL recognition model dir")
    parser.add_argument("--num-passes", type=int, default=3, help="Number of passes for self-consistency voting")

    # Security flags
    sec_group = parser.add_argument_group("data security")
    sec_group.add_argument("--pii-masking", action="store_true", help="Mask PII in prompts before sending to external APIs")
    sec_group.add_argument("--pii-strict", action="store_true", help="Apply stricter PII detection (e.g. financial ratios)")
    sec_group.add_argument("--audit-log", type=str, default=None, help="Path to write API audit JSONL log")
    sec_group.add_argument("--encrypt-artifacts", action="store_true", help="Encrypt cached artifacts on disk")
    sec_group.add_argument("--encryption-passphrase", type=str, default=None, help="Passphrase for artifact encryption (or set FINVL_ENCRYPTION_PASSPHRASE)")

    # Veto flags
    veto_group = parser.add_argument_group("one-vote veto")
    veto_group.add_argument("--enable-veto", action="store_true", help="Enable one-vote veto for low-confidence predictions")
    veto_group.add_argument("--veto-threshold", type=float, default=0.4, help="Confidence threshold below which veto fires (0.0-1.0)")
    veto_group.add_argument("--veto-no-evidence-check", action="store_true", help="Disable evidence support checking in veto scoring")
    veto_group.add_argument("--veto-no-contradiction", action="store_true", help="Disable evidence contradiction detection in veto scoring")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
