from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings, load_settings, require_value
from .ernie_client import ErnieClient
from .io import collect_markdown
from .paddleocr_client import LayoutOptions, PaddleOCRRemoteClient


@dataclass(frozen=True, slots=True)
class PyFiRecord:
    uid: str
    image_path: str
    question: str
    options: dict[str, str]
    answer: str | None
    capability: str | None
    complexity: str | None
    context: dict[str, Any]

    @property
    def valid_options(self) -> set[str]:
        return {str(key).upper() for key in self.options}


RecordKey = tuple[str, int]


def iter_pyfi_jsonl(path: Path) -> list[PyFiRecord]:
    records: list[PyFiRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(
                PyFiRecord(
                    uid=str(item["uid"]),
                    image_path=str(item["image_path"]),
                    question=str(item["question"]),
                    options={str(k).upper(): str(v) for k, v in dict(item["options"]).items()},
                    answer=str(item["answer"]).upper() if item.get("answer") else None,
                    capability=item.get("capability"),
                    complexity=str(item.get("complexity")) if item.get("complexity") is not None else None,
                    context=dict(item.get("context") or {}),
                )
            )
    return records


def occurrence_keys(records: list[PyFiRecord]) -> list[RecordKey]:
    seen: dict[str, int] = defaultdict(int)
    keys: list[RecordKey] = []
    for record in records:
        occurrence = seen[record.uid]
        seen[record.uid] += 1
        keys.append((record.uid, occurrence))
    return keys


def read_existing_results(path: Path) -> dict[RecordKey, dict[str, Any]]:
    if not path.exists():
        return {}
    seen: dict[str, int] = defaultdict(int)
    rows: dict[RecordKey, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            uid = str(item["uid"])
            occurrence = seen[uid]
            seen[uid] += 1
            rows[(uid, occurrence)] = item
    return rows


def resolve_image_path(images_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    clean = image_path[2:] if image_path.startswith("./") else image_path
    return images_root / clean


def safe_name(uid: str) -> str:
    digest = hashlib.sha1(uid.encode("utf-8")).hexdigest()
    return digest


def read_or_parse(
    client: PaddleOCRRemoteClient,
    record: PyFiRecord,
    image_path: Path,
    artifacts_dir: Path,
    options: LayoutOptions,
) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_path = artifacts_dir / f"{safe_name(record.uid)}.layout.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    result = client.parse_file(image_path, file_type="image", options=options)
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = artifacts_dir / f"{safe_name(record.uid)}.md"
    md_path.write_text(collect_markdown(result), encoding="utf-8")
    return result


def build_choice_prompt(record: PyFiRecord, markdown: str) -> str:
    background = str(record.context.get("image_background") or "").strip()
    return "\n".join(
        [
            "You answer multiple-choice questions about financial charts/documents.",
            "Use the remote PaddleOCR Markdown as primary evidence.",
            "The OCR may contain garbled multilingual text; rely on numeric values, table structure, labels, and the question context.",
            "Choose exactly one valid option.",
            'Return exactly JSON such as {"answer":"A"}.',
            "Do not return explanations, markdown, or extra text.",
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
            "Image background:",
            background,
            "",
            "Remote PaddleOCR Markdown:",
            trim(markdown, 10000),
        ]
    )


def trim(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n\n...[omitted]...\n\n{text[-half:]}"


def normalize_answer(raw: Any, valid_options: set[str]) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        for key in ("answer", "option", "prediction"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip().upper() in valid_options:
                return value.strip().upper()
    text = str(raw).strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            parsed = None
        else:
            try:
                parsed = json.loads(match.group(0))
            except ValueError:
                parsed = None
    if isinstance(parsed, dict):
        return normalize_answer(parsed, valid_options)
    upper = text.upper()
    if upper in valid_options:
        return upper
    for pattern in [
        r"\bANSWER\s*(?:IS|:|=)?\s*([A-Z])\b",
        r"\bOPTION\s*([A-Z])\b",
        r"^\s*([A-Z])[\).:\s]",
        r'"ANSWER"\s*:\s*"([A-Z])"',
    ]:
        match = re.search(pattern, upper)
        if match and match.group(1) in valid_options:
            return match.group(1)
    if len(upper) <= 80:
        for match in re.findall(r"\b([A-Z])\b", upper):
            if match in valid_options:
                return match
    return None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    invalid = sum(1 for row in rows if row.get("prediction") is None)
    by_capability = bucket(rows, "capability")
    by_complexity = bucket(rows, "complexity")
    return {
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / total if total else 0.0,
        "invalid_rate": invalid / total if total else 0.0,
        "by_capability": by_capability,
        "by_complexity": by_complexity,
    }


def bucket(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "None")].append(row)
    result: dict[str, dict[str, float | int]] = {}
    for name, values in sorted(grouped.items()):
        total = len(values)
        correct = sum(1 for row in values if row.get("correct"))
        invalid = sum(1 for row in values if row.get("prediction") is None)
        result[name] = {
            "total": total,
            "correct": correct,
            "invalid": invalid,
            "accuracy": correct / total if total else 0.0,
            "invalid_rate": invalid / total if total else 0.0,
        }
    return result


def evaluate(args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    records = iter_pyfi_jsonl(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]
    keys = occurrence_keys(records)

    paddle = PaddleOCRRemoteClient(
        api_url=settings.paddleocr_api_url,
        api_token=require_value(settings.paddleocr_api_token, "PADDLEOCR_API_TOKEN"),
        timeout=settings.timeout,
    )
    ernie = ErnieClient(
        api_key=require_value(settings.ernie_api_key, "ERNIE_API_KEY"),
        base_url=settings.ernie_base_url,
        model=args.ernie_model or settings.ernie_model,
        timeout=settings.timeout,
    )
    layout_options = LayoutOptions(
        use_doc_orientation_classify=args.use_doc_orientation_classify,
        use_doc_unwarping=args.use_doc_unwarping,
        use_chart_recognition=args.use_chart_recognition,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing_results(args.out) if args.resume else {}
    results: list[dict[str, Any]] = []
    mode = "a" if args.resume and existing else "w"
    if existing:
        for key in keys:
            if key in existing:
                results.append(existing[key])
    with args.out.open(mode, encoding="utf-8") as handle:
        for index, (record, record_key) in enumerate(zip(records, keys, strict=True), start=1):
            if record_key in existing:
                if args.progress_every and index % args.progress_every == 0:
                    metrics = aggregate(results)
                    print(
                        f"Skipped {index}; correct={metrics['correct']}/{metrics['total']} "
                        f"accuracy={metrics['accuracy']:.4f} invalid={metrics['invalid']}"
                    )
                continue
            image_path = resolve_image_path(args.images_root, record.image_path)
            error = None
            raw_answer = None
            prediction = None
            try:
                parsed = with_retries(
                    lambda: read_or_parse(
                        paddle,
                        record,
                        image_path,
                        args.artifacts_dir,
                        layout_options,
                    ),
                    attempts=args.retry_attempts,
                    base_sleep=args.retry_base_sleep,
                )
                markdown = collect_markdown(parsed)
                prompt = build_choice_prompt(record, markdown)
                raw_answer = with_retries(
                    lambda: ernie.complete(
                        prompt,
                        web_search=not args.disable_web_search,
                        max_completion_tokens=args.max_completion_tokens,
                        stream=True,
                        include_reasoning=args.include_reasoning_fallback,
                    ),
                    attempts=args.retry_attempts,
                    base_sleep=args.retry_base_sleep,
                )
                prediction = normalize_answer(raw_answer, record.valid_options)
                if prediction is None and not args.include_reasoning_fallback:
                    raw_answer = with_retries(
                        lambda: ernie.complete(
                            prompt,
                            web_search=not args.disable_web_search,
                            max_completion_tokens=args.max_completion_tokens,
                            stream=True,
                            include_reasoning=True,
                        ),
                        attempts=args.retry_attempts,
                        base_sleep=args.retry_base_sleep,
                    )
                    prediction = normalize_answer(raw_answer, record.valid_options)
            except Exception as exc:  # pragma: no cover - integration path
                error = f"{type(exc).__name__}: {exc}"

            if prediction is None and args.fallback_first_option:
                prediction = sorted(record.valid_options)[0]
                if raw_answer:
                    raw_answer = f"{raw_answer}\n\n[fallback_first_option={prediction}]"
                else:
                    raw_answer = f"[fallback_first_option={prediction}]"

            item = {
                "uid": record.uid,
                "image_path": str(image_path),
                "capability": record.capability,
                "complexity": record.complexity,
                "answer": record.answer,
                "prediction": prediction,
                "raw_prediction": raw_answer,
                "correct": prediction == record.answer if record.answer else False,
                "error": error,
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            results.append(item)
            if args.sleep_between_records > 0:
                time.sleep(args.sleep_between_records)
            if args.progress_every and index % args.progress_every == 0:
                metrics = aggregate(results)
                print(
                    f"Processed {index}; correct={metrics['correct']}/{metrics['total']} "
                    f"accuracy={metrics['accuracy']:.4f} invalid={metrics['invalid']}"
                )

    metrics = aggregate(results)
    metrics["remote_paddleocr"] = {
        "use_chart_recognition": args.use_chart_recognition,
        "use_doc_orientation_classify": args.use_doc_orientation_classify,
        "use_doc_unwarping": args.use_doc_unwarping,
    }
    metrics["ernie_model"] = args.ernie_model or settings.ernie_model
    metrics["eval_runtime"] = {
        "max_completion_tokens": args.max_completion_tokens,
        "retry_attempts": args.retry_attempts,
        "retry_base_sleep": args.retry_base_sleep,
        "sleep_between_records": args.sleep_between_records,
        "include_reasoning_fallback": args.include_reasoning_fallback,
        "fallback_first_option": args.fallback_first_option,
        "resume": args.resume,
    }
    metrics_path = args.out.with_suffix(args.out.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def with_retries(action: Any, *, attempts: int, base_sleep: float) -> Any:
    last_error: Exception | None = None
    for attempt in range(max(attempts, 1)):
        try:
            return action()
        except Exception as exc:  # pragma: no cover - integration path
            last_error = exc
            if attempt >= attempts - 1:
                break
            time.sleep(base_sleep * (2**attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry action failed without an exception")


def add_eval_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("eval-pyfi", help="Evaluate remote PaddleOCR+ERNIE on a PyFi JSONL split.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ernie-model")
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--include-reasoning-fallback", action="store_true")
    parser.add_argument("--fallback-first-option", action="store_true")
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-base-sleep", type=float, default=3.0)
    parser.add_argument("--sleep-between-records", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-web-search", action="store_true")
    parser.add_argument("--use-doc-orientation-classify", action="store_true")
    parser.add_argument("--use-doc-unwarping", action="store_true")
    parser.add_argument("--use-chart-recognition", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate remote PaddleOCR+ERNIE on PyFi.")
    parser.add_argument("--env-file", type=Path)
    add_eval_parser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args()
    settings = load_settings(args.env_file)
    evaluate(args, settings)


if __name__ == "__main__":
    main()
