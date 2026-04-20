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


def iter_records(dataset: str | Path, dataset_format: str) -> Iterable[EvalRecord]:
    if dataset_format == "pyfi-csv":
        return iter_pyfi_csv(dataset)
    if dataset_format == "jsonl":
        return iter_jsonl(dataset)
    raise ValueError(f"Unknown dataset format: {dataset_format}")


def run(args: argparse.Namespace) -> dict:
    adapter = build_adapter(args)
    records_iter = iter_records(args.dataset, args.format)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    processed = 0
    existing_uids: set[str] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as existing:
            for line in existing:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    # Ignore a truncated tail line left behind by an interrupted run.
                    continue
                results.append(item)
                uid = item.get("uid")
                if isinstance(uid, str):
                    existing_uids.add(uid)
        processed = len(results)

    mode = "a" if existing_uids else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for record in records_iter:
            if args.limit is not None and processed >= args.limit:
                break
            if record.uid in existing_uids:
                continue
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
            routed_prompt_label = getattr(adapter, "last_remote_prompt_label", None)
            if isinstance(routed_prompt_label, str) and routed_prompt_label:
                item["remote_prompt_label"] = routed_prompt_label
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            results.append(item)
            processed += 1
            if args.progress_every and processed % args.progress_every == 0:
                print(f"Processed {processed} records", file=sys.stderr)

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
            "remote-paddleocr-vl-ernie-docqa",
            "paddleocr-vl-hybrid-docqa",
            "paddleocr-vl-grounded-docqa",
            "remote-paddleocr-vl-grounded-docqa",
            "remote-paddleocr-vl-table-spotting-router-docqa",
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
    parser.add_argument("--paddle-vl-api-url", help="Remote PaddleOCR-VL layout-parsing API URL")
    parser.add_argument(
        "--paddle-vl-prompt-label",
        choices=["chart", "seal", "spotting", "table", "formula", "ocr"],
        help="Remote PaddleOCR-VL promptLabel to use with the official parsing payload.",
    )
    local_ocr_group = parser.add_mutually_exclusive_group()
    local_ocr_group.add_argument(
        "--use-local-ocr-evidence",
        dest="use_local_ocr_evidence",
        action="store_true",
        default=None,
        help="Keep traditional local OCR as an extra evidence channel.",
    )
    local_ocr_group.add_argument(
        "--no-local-ocr-evidence",
        dest="use_local_ocr_evidence",
        action="store_false",
        help="Disable traditional local OCR and rely on PaddleOCR-VL markdown only.",
    )
    parser.add_argument("--num-passes", type=int, default=3, help="Number of passes for self-consistency voting")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
