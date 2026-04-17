from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from .models import _chat_completion_text, _safe_uid
from .pyfi import iter_jsonl
from .records import EvalRecord
from .scoring import aggregate, normalize_answer


OccurrenceKey = tuple[str, int]


def _occurrence_key(uid: str, seen: dict[str, int]) -> OccurrenceKey:
    occurrence = seen[uid]
    seen[uid] += 1
    return uid, occurrence


def _read_results(path: Path) -> dict[OccurrenceKey, dict[str, Any]]:
    rows: dict[OccurrenceKey, dict[str, Any]] = {}
    seen: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows[_occurrence_key(item["uid"], seen)] = item
    return rows


def _read_records(path: Path) -> list[EvalRecord]:
    return list(iter_jsonl(path))


def _read_artifact_text(directory: Path | None, uid: str, suffix: str) -> str:
    if directory is None:
        return ""
    path = directory / f"{_safe_uid(uid)}.{suffix}"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _trim(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit // 2]}\n\n...[omitted]...\n\n{text[-limit // 2:]}"


def _build_prompt(
    record: EvalRecord,
    primary: dict[str, Any],
    secondary: dict[str, Any],
    markdown: str,
    ocr_text: str,
) -> str:
    return "\n".join(
        [
            "You are adjudicating two PaddleOCR-based financial document QA runs.",
            "Both candidate answers were produced from PaddleOCR-VL/PaddleOCR evidence.",
            "Use the Paddle evidence and the question/options to choose the best valid option.",
            "You may choose either candidate or a third option if the evidence clearly supports it.",
            "Never output null or an explanation outside JSON.",
            'Return exactly JSON: {"answer":"A","reason":"short evidence-based reason"}',
            "",
            f"Capability: {record.capability}",
            f"Complexity: {record.complexity}",
            "",
            "Question:",
            record.question,
            "",
            "Options:",
            json.dumps(record.options, ensure_ascii=False),
            "",
            "Candidate from primary run:",
            json.dumps(
                {
                    "prediction": primary.get("prediction"),
                    "raw_prediction": str(primary.get("raw_prediction") or "")[:1200],
                },
                ensure_ascii=False,
            ),
            "",
            "Candidate from secondary run:",
            json.dumps(
                {
                    "prediction": secondary.get("prediction"),
                    "raw_prediction": str(secondary.get("raw_prediction") or "")[:1200],
                },
                ensure_ascii=False,
            ),
            "",
            "PaddleOCR-VL markdown:",
            _trim(markdown, 12000),
            "",
            "Traditional PaddleOCR text:",
            _trim(ocr_text, 6000),
            "",
            "Image background:",
            _trim(str(record.context.get("image_background") or ""), 2500),
        ]
    )


def adjudicate(args: argparse.Namespace) -> dict[str, Any]:
    records = _read_records(args.dataset)
    primary = _read_results(args.primary)
    secondary = _read_results(args.secondary)
    api_key = args.selector_api_key or os.getenv("FINVL_SELECTOR_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = args.selector_base_url or os.getenv("FINVL_SELECTOR_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise RuntimeError("FINVL_SELECTOR_API_KEY or OPENAI_API_KEY is required.")
    client = OpenAI(api_key=api_key, base_url=base_url or "https://api.openai.com/v1", timeout=120)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    adjudicated = 0
    with args.out.open("w", encoding="utf-8") as handle:
        seen_records: dict[str, int] = defaultdict(int)
        for record in records:
            key = _occurrence_key(record.uid, seen_records)
            if key not in primary or key not in secondary:
                continue
            item = dict(primary[key])
            pred_a = primary[key].get("prediction")
            pred_b = secondary[key].get("prediction")
            should_adjudicate = pred_a != pred_b
            if args.only_options:
                allowed = set(args.only_options.upper())
                should_adjudicate = should_adjudicate and (pred_a in allowed or pred_b in allowed)
            if should_adjudicate:
                markdown = _read_artifact_text(args.artifacts_dir, record.uid, "md")
                ocr_text = _read_artifact_text(args.ocr_artifacts_dir, record.uid, "txt")
                prompt = _build_prompt(record, primary[key], secondary[key], markdown, ocr_text)
                raw = _chat_completion_text(
                    client,
                    model=args.selector_model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.selector_max_tokens,
                )
                prediction = normalize_answer(raw, record.valid_options)
                if prediction is None:
                    prediction = pred_a or pred_b
                    raw = json.dumps({"answer": prediction, "fallback": "primary_or_secondary"}, ensure_ascii=False)
                item["prediction"] = prediction
                item["raw_prediction"] = json.dumps(
                    {
                        "answer": prediction,
                        "adjudication_raw": raw,
                        "primary_prediction": pred_a,
                        "secondary_prediction": pred_b,
                    },
                    ensure_ascii=False,
                )
                item["correct"] = prediction == record.answer if record.answer else False
                item["error"] = None
                adjudicated += 1
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            results.append(item)
            if args.progress_every and len(results) % args.progress_every == 0:
                print(f"Processed {len(results)} records; adjudicated {adjudicated}")

    metrics = aggregate(results)
    metrics["adjudicated"] = adjudicated
    metrics_path = args.out.with_suffix(args.out.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Adjudicate disagreements between two PaddleOCR result runs.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--secondary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--selector-model", default="ernie-4.5-turbo-128k-preview")
    parser.add_argument("--selector-base-url")
    parser.add_argument("--selector-api-key")
    parser.add_argument("--selector-max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--ocr-artifacts-dir", type=Path)
    parser.add_argument("--only-options", help="Only adjudicate disagreements involving these predicted option letters")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    adjudicate(args)


if __name__ == "__main__":
    main()
