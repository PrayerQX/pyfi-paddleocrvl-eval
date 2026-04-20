from __future__ import annotations

import argparse
import base64
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from finvl_eval.models import (
    _build_remote_markdown_selector_prompt,
    _chat_completion_text,
    _paddleocr_file_type,
    _safe_uid,
    _valid_or_fallback,
    normalize_prediction,
)
from finvl_eval.pyfi import iter_jsonl
from finvl_eval.scoring import aggregate


CONFIGS: dict[str, dict[str, Any]] = {
    "minimal_current": {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": True,
    },
    "official_full": {
        "markdownIgnoreLabels": [
            "header",
            "header_image",
            "footer",
            "footer_image",
            "number",
            "footnote",
            "aside_text",
        ],
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useLayoutDetection": False,
        "useChartRecognition": True,
        "useSealRecognition": True,
        "useOcrForImageBlock": True,
        "mergeTables": True,
        "relevelTitles": True,
        "layoutShapeMode": "auto",
        "promptLabel": "spotting",
        "repetitionPenalty": 1,
        "temperature": 0,
        "topP": 1,
        "minPixels": 147384,
        "maxPixels": 2822400,
        "layoutNms": True,
        "restructurePages": True,
    },
    "official_prompt_chart": {
        "markdownIgnoreLabels": [
            "header",
            "header_image",
            "footer",
            "footer_image",
            "number",
            "footnote",
            "aside_text",
        ],
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useLayoutDetection": False,
        "useChartRecognition": True,
        "useSealRecognition": True,
        "useOcrForImageBlock": True,
        "mergeTables": True,
        "relevelTitles": True,
        "layoutShapeMode": "auto",
        "promptLabel": "chart",
        "repetitionPenalty": 1,
        "temperature": 0,
        "topP": 1,
        "minPixels": 147384,
        "maxPixels": 2822400,
        "layoutNms": True,
        "restructurePages": True,
    },
    "official_prompt_seal": {
        "markdownIgnoreLabels": [
            "header",
            "header_image",
            "footer",
            "footer_image",
            "number",
            "footnote",
            "aside_text",
        ],
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useLayoutDetection": False,
        "useChartRecognition": True,
        "useSealRecognition": True,
        "useOcrForImageBlock": True,
        "mergeTables": True,
        "relevelTitles": True,
        "layoutShapeMode": "auto",
        "promptLabel": "seal",
        "repetitionPenalty": 1,
        "temperature": 0,
        "topP": 1,
        "minPixels": 147384,
        "maxPixels": 2822400,
        "layoutNms": True,
        "restructurePages": True,
    },
    "official_prompt_spotting": {
        "markdownIgnoreLabels": [
            "header",
            "header_image",
            "footer",
            "footer_image",
            "number",
            "footnote",
            "aside_text",
        ],
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useLayoutDetection": False,
        "useChartRecognition": True,
        "useSealRecognition": True,
        "useOcrForImageBlock": True,
        "mergeTables": True,
        "relevelTitles": True,
        "layoutShapeMode": "auto",
        "promptLabel": "spotting",
        "repetitionPenalty": 1,
        "temperature": 0,
        "topP": 1,
        "minPixels": 147384,
        "maxPixels": 2822400,
        "layoutNms": True,
        "restructurePages": True,
    },
    "official_prompt_table": {
        "markdownIgnoreLabels": [
            "header",
            "header_image",
            "footer",
            "footer_image",
            "number",
            "footnote",
            "aside_text",
        ],
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useLayoutDetection": False,
        "useChartRecognition": True,
        "useSealRecognition": True,
        "useOcrForImageBlock": True,
        "mergeTables": True,
        "relevelTitles": True,
        "layoutShapeMode": "auto",
        "promptLabel": "table",
        "repetitionPenalty": 1,
        "temperature": 0,
        "topP": 1,
        "minPixels": 147384,
        "maxPixels": 2822400,
        "layoutNms": True,
        "restructurePages": True,
    },
    "official_prompt_formula": {
        "markdownIgnoreLabels": [
            "header",
            "header_image",
            "footer",
            "footer_image",
            "number",
            "footnote",
            "aside_text",
        ],
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useLayoutDetection": False,
        "useChartRecognition": True,
        "useSealRecognition": True,
        "useOcrForImageBlock": True,
        "mergeTables": True,
        "relevelTitles": True,
        "layoutShapeMode": "auto",
        "promptLabel": "formula",
        "repetitionPenalty": 1,
        "temperature": 0,
        "topP": 1,
        "minPixels": 147384,
        "maxPixels": 2822400,
        "layoutNms": True,
        "restructurePages": True,
    },
    "official_prompt_ocr": {
        "markdownIgnoreLabels": [
            "header",
            "header_image",
            "footer",
            "footer_image",
            "number",
            "footnote",
            "aside_text",
        ],
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useLayoutDetection": False,
        "useChartRecognition": True,
        "useSealRecognition": True,
        "useOcrForImageBlock": True,
        "mergeTables": True,
        "relevelTitles": True,
        "layoutShapeMode": "auto",
        "promptLabel": "ocr",
        "repetitionPenalty": 1,
        "temperature": 0,
        "topP": 1,
        "minPixels": 147384,
        "maxPixels": 2822400,
        "layoutNms": True,
        "restructurePages": True,
    },
    "official_layout_on": {
        "markdownIgnoreLabels": [
            "header",
            "header_image",
            "footer",
            "footer_image",
            "number",
            "footnote",
            "aside_text",
        ],
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useLayoutDetection": True,
        "useChartRecognition": True,
        "useSealRecognition": True,
        "useOcrForImageBlock": True,
        "mergeTables": True,
        "relevelTitles": True,
        "layoutShapeMode": "auto",
        "promptLabel": "spotting",
        "repetitionPenalty": 1,
        "temperature": 0,
        "topP": 1,
        "minPixels": 147384,
        "maxPixels": 2822400,
        "layoutNms": True,
        "restructurePages": True,
    },
    "official_keep_labels": {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useLayoutDetection": False,
        "useChartRecognition": True,
        "useSealRecognition": True,
        "useOcrForImageBlock": True,
        "mergeTables": True,
        "relevelTitles": True,
        "layoutShapeMode": "auto",
        "promptLabel": "spotting",
        "repetitionPenalty": 1,
        "temperature": 0,
        "topP": 1,
        "minPixels": 147384,
        "maxPixels": 2822400,
        "layoutNms": True,
        "restructurePages": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep remote PaddleOCR-VL parsing parameters on a stratified PyFi subset.")
    parser.add_argument("--dataset", default="data/pyfi/pyfi_eval_301.jsonl")
    parser.add_argument("--images-root", default="data/pyfi")
    parser.add_argument("--subset-per-capability", type=int, default=4)
    parser.add_argument("--out-dir", default="runs/paddleocrvl_param_sweep")
    parser.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), help="Optional subset of config names to run")
    parser.add_argument("--api-url", default=os.getenv("PADDLEOCR_VL_API_URL"))
    parser.add_argument("--api-token", default=os.getenv("PADDLEOCR_VL_API_TOKEN"))
    parser.add_argument("--selector-model", default="ernie-4.5-turbo-128k-preview")
    parser.add_argument("--selector-base-url", default=os.getenv("FINVL_SELECTOR_BASE_URL", "https://aistudio.baidu.com/llm/lmapi/v3"))
    parser.add_argument("--selector-api-key", default=os.getenv("FINVL_SELECTOR_API_KEY"))
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def build_subset(dataset: str | Path, per_capability: int) -> list[Any]:
    buckets: dict[str, list[Any]] = defaultdict(list)
    seen_image_per_cap: dict[str, set[str]] = defaultdict(set)
    for record in iter_jsonl(dataset):
        cap = str(record.capability)
        if len(buckets[cap]) >= per_capability:
            continue
        if record.image_path in seen_image_per_cap[cap]:
            continue
        buckets[cap].append(record)
        seen_image_per_cap[cap].add(record.image_path)
    subset: list[Any] = []
    for key in sorted(buckets):
        subset.extend(buckets[key])
    return subset


def parse_remote(image_path: Path, api_url: str, api_token: str, payload_overrides: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    file_bytes = image_path.read_bytes()
    payload = {
        "file": base64.b64encode(file_bytes).decode("ascii"),
        "fileType": _paddleocr_file_type(image_path),
        **payload_overrides,
    }
    response = requests.post(
        api_url,
        json=payload,
        headers={
            "Authorization": f"token {api_token}",
            "Content-Type": "application/json",
        },
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    result = body["result"]
    markdown_parts: list[str] = []
    for item in result.get("layoutParsingResults", []):
        markdown = item.get("markdown")
        if isinstance(markdown, dict) and markdown.get("text"):
            markdown_parts.append(str(markdown["text"]))
    return "\n\n".join(part for part in markdown_parts if part).strip(), result


def main() -> None:
    args = parse_args()
    if not args.api_url or not args.api_token:
        raise RuntimeError("PADDLEOCR_VL_API_URL and PADDLEOCR_VL_API_TOKEN are required.")
    if not args.selector_api_key:
        raise RuntimeError("FINVL_SELECTOR_API_KEY is required.")

    subset = build_subset(args.dataset, args.subset_per_capability)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selector = OpenAI(api_key=args.selector_api_key, base_url=args.selector_base_url, timeout=120)

    subset_path = out_dir / "subset.jsonl"
    subset_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "uid": record.uid,
                    "image_path": record.image_path,
                    "question": record.question,
                    "options": record.options,
                    "answer": record.answer,
                    "capability": record.capability,
                    "complexity": record.complexity,
                    "context": record.context,
                    "metadata": record.metadata,
                },
                ensure_ascii=False,
            )
            for record in subset
        )
        + "\n",
        encoding="utf-8",
    )

    summary: dict[str, Any] = {}
    selected = args.configs or list(CONFIGS)
    for config_name in selected:
        payload_overrides = CONFIGS[config_name]
        config_dir = out_dir / config_name
        config_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        markdown_lengths: list[int] = []
        for index, record in enumerate(subset, start=1):
            image_path = record.resolved_image_path(args.images_root)
            cache_md = config_dir / f"{_safe_uid(record.uid)}.md"
            cache_json = config_dir / f"{_safe_uid(record.uid)}.json"
            if cache_md.exists() and cache_json.exists():
                parsed_markdown = cache_md.read_text(encoding="utf-8")
            else:
                parsed_markdown, raw_result = parse_remote(image_path, args.api_url, args.api_token, payload_overrides)
                cache_md.write_text(parsed_markdown, encoding="utf-8")
                cache_json.write_text(json.dumps(raw_result, ensure_ascii=False, indent=2), encoding="utf-8")
            markdown_lengths.append(len(parsed_markdown))

            prompt = _build_remote_markdown_selector_prompt(record, {"PaddleOCR-VL parsed markdown": parsed_markdown})
            try:
                raw_prediction = _chat_completion_text(
                    selector,
                    model=args.selector_model,
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=args.max_tokens,
                )
                error = None
            except Exception as exc:
                raw_prediction = None
                error = f"{type(exc).__name__}: {exc}"

            final_prediction = _valid_or_fallback(record, raw_prediction, parsed_markdown)
            prediction = normalize_prediction(record, final_prediction)
            item = {
                "uid": record.uid,
                "answer": record.answer,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "correct": prediction == record.answer if record.answer else False,
                "error": error,
                "capability": record.capability,
                "complexity": record.complexity,
                "markdown_chars": len(parsed_markdown),
            }
            results.append(item)
            print(f"[{config_name}] {index}/{len(subset)} {record.uid} -> {prediction} (ans={record.answer})")

        metrics = aggregate(results)
        metrics["avg_markdown_chars"] = sum(markdown_lengths) / len(markdown_lengths) if markdown_lengths else 0
        metrics["subset_size"] = len(subset)
        (config_dir / "results.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n",
            encoding="utf-8",
        )
        (config_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[config_name] = metrics

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
