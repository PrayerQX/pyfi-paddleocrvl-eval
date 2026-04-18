from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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


def complete_with_cache(
    ernie: ErnieClient,
    args: argparse.Namespace,
    prompt: str,
) -> tuple[str, bool]:
    cache_dir: Path | None = args.selector_cache_dir
    web_search = not args.disable_web_search
    if cache_dir is None:
        return (
            ernie.complete(
                prompt,
                web_search=web_search,
                max_completion_tokens=args.max_completion_tokens,
                stream=True,
                include_reasoning=False,
                temperature=args.temperature,
            ),
            False,
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{selector_cache_key(args, prompt)}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            raw = cached.get("raw")
            if isinstance(raw, str):
                return raw, True
        except (OSError, json.JSONDecodeError):
            pass

    raw = ernie.complete(
        prompt,
        web_search=web_search,
        max_completion_tokens=args.max_completion_tokens,
        stream=True,
        include_reasoning=False,
        temperature=args.temperature,
    )
    payload = {
        "model": getattr(args, "_selector_model_name", None) or args.ernie_model,
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": args.temperature,
        "web_search": web_search,
        "raw": raw,
    }
    tmp_path = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(cache_path)
    return raw, False


def selector_cache_key(args: argparse.Namespace, prompt: str) -> str:
    payload = {
        "model": getattr(args, "_selector_model_name", None) or args.ernie_model,
        "prompt": prompt,
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": args.temperature,
        "web_search": not args.disable_web_search,
        "stream": True,
        "include_reasoning": False,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_choice_prompt(record: PyFiRecord, markdown: str, layout_blocks: str = "") -> str:
    background = str(record.context.get("image_background") or "").strip()
    lines = [
        "You answer multiple-choice questions about financial charts/documents.",
        "Use the remote PaddleOCR Markdown, the question, the options, and the image background together.",
        "The OCR may contain garbled multilingual text; rely on numeric values, table structure, labels, and the image background.",
        "Do not prefer an option just because it appears earlier or because the OCR text is noisy.",
        "Choose exactly one valid option.",
        'Return exactly JSON such as {"answer":"A"}.',
        "Do not return explanations, markdown, or extra text.",
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
    if layout_blocks.strip():
        lines.extend(
            [
                "",
                "Remote PaddleOCR layout blocks:",
                trim(layout_blocks, 6000),
            ]
        )
    return "\n".join(lines)


def build_structured_choice_prompt(record: PyFiRecord, structured_evidence: str) -> str:
    background = str(record.context.get("image_background") or "").strip()
    return "\n".join(
        [
            "You answer multiple-choice questions about financial charts/documents.",
            "All evidence below comes from the remote PaddleOCR-VL layout parsing API.",
            "Use the structured intermediate evidence, the question, the options, and the image background together.",
            "If evidence is incomplete, choose the option best supported by remote PaddleOCR-VL evidence.",
            "Do not use option-letter priors and do not prefer an option because it appears earlier.",
            "Choose exactly one valid option.",
            'Return exactly JSON such as {"answer":"A"}.',
            "Do not return explanations, markdown, or extra text.",
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
            "Remote PaddleOCR-VL structured intermediate:",
            trim(structured_evidence, 18000),
        ]
    )


def collect_structured_intermediate(record: PyFiRecord, result: dict[str, Any], markdown: str) -> str:
    query_text = " ".join([record.question, *record.options.values()])
    query_tokens = keywords(query_text)
    query_numbers = set(number_strings(query_text))
    lines = collect_structured_lines(result, markdown)
    scored = [
        {
            "source": source,
            "score": score_text(text, query_tokens, query_numbers),
            "text": text,
        }
        for source, text in lines
    ]
    relevant = [item for item in sorted(scored, key=lambda row: row["score"], reverse=True) if item["score"] > 0]
    if len(relevant) < 16:
        relevant = sorted(scored, key=lambda row: row["score"], reverse=True)
    evidence_text = "\n".join(text for _, text in lines)
    payload = {
        "question_focus": {
            "keywords": sorted(query_tokens)[:40],
            "numbers": sorted(query_numbers),
        },
        "option_evidence": structured_option_evidence(record, evidence_text),
        "numeric_candidates": numeric_candidates(record, relevant[:40]),
        "relevant_remote_paddleocr_rows": relevant[:32],
        "remote_paddleocr_markdown_excerpt": trim(markdown, 5000),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def collect_structured_lines(result: dict[str, Any], markdown: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for document_index, res in enumerate(result.get("layoutParsingResults", [])):
        pruned = res.get("prunedResult") or {}
        blocks = pruned.get("parsing_res_list") or []
        if isinstance(blocks, list):
            for block in sorted(blocks, key=layout_block_sort_key):
                if not isinstance(block, dict):
                    continue
                label = str(block.get("block_label") or "unknown")
                bbox = block.get("block_bbox") or block.get("bbox") or []
                content = " ".join(str(block.get("block_content") or "").split())
                if not content:
                    continue
                rows.append((f"layout_block_{document_index}_{label}", f"{label} bbox={bbox}: {content}"))
    for line in markdown.splitlines():
        text = " ".join(line.split())
        if text and text not in {"|", "-"}:
            rows.append(("markdown", text))
    return rows[:240]


def collect_layout_blocks(result: dict[str, Any]) -> str:
    lines: list[str] = []
    for document_index, res in enumerate(result.get("layoutParsingResults", [])):
        pruned = res.get("prunedResult") or {}
        blocks = pruned.get("parsing_res_list") or []
        if not isinstance(blocks, list):
            continue
        lines.append(f"# Document {document_index} layout blocks")
        for block in sorted(blocks, key=layout_block_sort_key):
            if not isinstance(block, dict):
                continue
            label = str(block.get("block_label") or "unknown")
            bbox = block.get("block_bbox") or block.get("bbox") or []
            content = " ".join(str(block.get("block_content") or "").split())
            if not content and label not in {"table", "chart", "image"}:
                continue
            if content:
                lines.append(f"- {label} bbox={bbox}: {trim(content, 240)}")
            else:
                lines.append(f"- {label} bbox={bbox}")
    return "\n".join(lines)


def layout_block_sort_key(block: dict[str, Any]) -> tuple[int, int, int]:
    order = block.get("block_order")
    if isinstance(order, int):
        return (0, order, 0)
    bbox = block.get("block_bbox") or []
    if isinstance(bbox, list) and len(bbox) >= 2:
        return (1, int(bbox[1]), int(bbox[0]))
    return (2, 0, 0)


def structured_option_evidence(record: PyFiRecord, evidence_text: str) -> list[dict[str, Any]]:
    evidence_lower = evidence_text.lower()
    rows: list[dict[str, Any]] = []
    for letter, option in record.options.items():
        option_text = str(option)
        option_tokens = keywords(option_text)
        option_numbers = number_strings(option_text)
        rows.append(
            {
                "option": letter,
                "text": option_text,
                "exact_text_in_remote_paddleocr": option_text.strip().lower() in evidence_lower
                if option_text.strip()
                else False,
                "keyword_hits": sorted(token for token in option_tokens if token in evidence_lower),
                "number_hits": [number for number in option_numbers if number in evidence_text],
            }
        )
    return rows


def numeric_candidates(record: PyFiRecord, relevant: list[dict[str, Any]]) -> list[str]:
    question = record.question.lower()
    option_values = {
        letter: [parse_number(num) for num in number_strings(text)]
        for letter, text in record.options.items()
    }
    option_values = {letter: [num for num in values if num is not None] for letter, values in option_values.items()}
    evidence_values: list[tuple[float, str]] = []
    for row in relevant:
        text = str(row.get("text") or "")
        for raw in number_strings(text):
            value = parse_number(raw)
            if value is not None:
                evidence_values.append((value, text))
    evidence_values = evidence_values[:80]

    candidates: list[tuple[float, str]] = []
    if any(word in question for word in ["percentage decrease", "percent decrease", "decrease"]):
        for old, old_line in evidence_values:
            if old == 0:
                continue
            for new, new_line in evidence_values:
                pct = ((old - new) / abs(old)) * 100
                append_option_matches(candidates, option_values, pct, f"percentage_decrease {old:g} -> {new:g} = {pct:.2f}% | {old_line} / {new_line}")
    if any(word in question for word in ["percentage increase", "percent increase", "increase", "growth"]):
        for old, old_line in evidence_values:
            if old == 0:
                continue
            for new, new_line in evidence_values:
                pct = ((new - old) / abs(old)) * 100
                append_option_matches(candidates, option_values, pct, f"percentage_increase {old:g} -> {new:g} = {pct:.2f}% | {old_line} / {new_line}")
    if any(word in question for word in ["difference", "change"]):
        for left, left_line in evidence_values:
            for right, right_line in evidence_values:
                diff = left - right
                append_option_matches(candidates, option_values, diff, f"difference {left:g} - {right:g} = {diff:.2f} | {left_line} / {right_line}")

    deduped: list[str] = []
    seen: set[str] = set()
    for _, text in sorted(candidates, key=lambda item: item[0])[:12]:
        if text not in seen:
            deduped.append(text)
            seen.add(text)
    return deduped


def append_option_matches(
    candidates: list[tuple[float, str]],
    option_values: dict[str, list[float]],
    computed: float,
    description: str,
) -> None:
    for letter, values in option_values.items():
        for option_value in values:
            tolerance = max(0.75, abs(option_value) * 0.025)
            delta = abs(computed - option_value)
            if delta <= tolerance:
                candidates.append((delta, f"matches option {letter}: {description}"))


def score_text(text: str, query_tokens: set[str], query_numbers: set[str]) -> int:
    lowered = text.lower()
    line_tokens = keywords(text)
    number_hits = sum(4 for number in query_numbers if number and number in text)
    token_hits = sum(1 for token in query_tokens if token in line_tokens or token in lowered)
    unit_bonus = 2 if re.search(r"[%$€£¥]|percent|percentage|ratio|rate|gdp|revenue|profit|debt", lowered) else 0
    return token_hits + number_hits + unit_bonus


def keywords(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "from",
        "with",
        "which",
        "what",
        "this",
        "that",
        "shown",
        "based",
        "option",
        "into",
        "without",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*|[\u4e00-\u9fff]{2,}", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in stop}


def number_strings(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])-?\d[\d,]*(?:\.\d+)?%?", text)


def parse_number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").rstrip("%"))
    except ValueError:
        return None


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


def process_record(
    args: argparse.Namespace,
    paddle: PaddleOCRRemoteClient,
    ernie: ErnieClient,
    layout_options: LayoutOptions,
    index: int,
    record: PyFiRecord,
) -> dict[str, Any]:
    image_path = resolve_image_path(args.images_root, record.image_path)
    error = None
    raw_answer = None
    prediction = None
    selector_cache_hit = False
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
        if args.structured_intermediate:
            structured_evidence = collect_structured_intermediate(record, parsed, markdown)
            prompt = build_structured_choice_prompt(record, structured_evidence)
        else:
            layout_blocks = collect_layout_blocks(parsed) if args.include_layout_blocks else ""
            prompt = build_choice_prompt(record, markdown, layout_blocks)
        raw_answer, selector_cache_hit = with_retries(
            lambda: complete_with_cache(ernie, args, prompt),
            attempts=args.retry_attempts,
            base_sleep=args.retry_base_sleep,
        )
        prediction = normalize_answer(raw_answer, record.valid_options)
    except Exception as exc:  # pragma: no cover - integration path
        error = f"{type(exc).__name__}: {exc}"

    return {
        "index": index,
        "uid": record.uid,
        "image_path": str(image_path),
        "capability": record.capability,
        "complexity": record.complexity,
        "answer": record.answer,
        "prediction": prediction,
        "raw_prediction": raw_answer,
        "selector_cache_hit": selector_cache_hit,
        "correct": prediction == record.answer if record.answer else False,
        "error": error,
    }


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
    args._selector_model_name = args.ernie_model or settings.ernie_model
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
        pending: list[tuple[int, PyFiRecord, RecordKey]] = []
        for index, (record, record_key) in enumerate(zip(records, keys, strict=True), start=1):
            if record_key in existing:
                if args.progress_every and index % args.progress_every == 0:
                    metrics = aggregate(results)
                    print(
                        f"Skipped {index}; correct={metrics['correct']}/{metrics['total']} "
                        f"accuracy={metrics['accuracy']:.4f} invalid={metrics['invalid']}"
                    )
                continue
            pending.append((index, record, record_key))

        def write_item(item: dict[str, Any]) -> None:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            results.append(item)
            if args.sleep_between_records > 0 and args.selector_concurrency <= 1:
                time.sleep(args.sleep_between_records)
            index = int(item["index"])
            if args.progress_every and index % args.progress_every == 0:
                metrics = aggregate(results)
                cache_hits = sum(1 for row in results if row.get("selector_cache_hit"))
                print(
                    f"Processed {index}; correct={metrics['correct']}/{metrics['total']} "
                    f"accuracy={metrics['accuracy']:.4f} invalid={metrics['invalid']} "
                    f"selector_cache_hits={cache_hits}"
                )

        if args.selector_concurrency <= 1:
            for index, record, _record_key in pending:
                write_item(process_record(args, paddle, ernie, layout_options, index, record))
        else:
            with ThreadPoolExecutor(max_workers=args.selector_concurrency) as executor:
                mapped = executor.map(
                    lambda item: process_record(args, paddle, ernie, layout_options, item[0], item[1]),
                    pending,
                )
                for item in mapped:
                    write_item(item)

    metrics = aggregate(results)
    metrics["selector_cache"] = {
        "dir": str(args.selector_cache_dir) if args.selector_cache_dir else None,
        "hits": sum(1 for row in results if row.get("selector_cache_hit")),
        "total": len(results),
    }
    metrics["remote_paddleocr"] = {
        "use_chart_recognition": args.use_chart_recognition,
        "use_doc_orientation_classify": args.use_doc_orientation_classify,
        "use_doc_unwarping": args.use_doc_unwarping,
    }
    metrics["ernie_model"] = args.ernie_model or settings.ernie_model
    metrics["eval_runtime"] = {
        "structured_intermediate": args.structured_intermediate,
        "include_layout_blocks": args.include_layout_blocks,
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": args.temperature,
        "selector_cache_dir": str(args.selector_cache_dir) if args.selector_cache_dir else None,
        "selector_concurrency": args.selector_concurrency,
        "retry_attempts": args.retry_attempts,
        "retry_base_sleep": args.retry_base_sleep,
        "sleep_between_records": args.sleep_between_records,
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
    parser.add_argument("--selector-cache-dir", type=Path)
    parser.add_argument("--selector-concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ernie-model")
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-base-sleep", type=float, default=3.0)
    parser.add_argument("--sleep-between-records", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-web-search", action="store_true")
    parser.add_argument("--use-doc-orientation-classify", action="store_true")
    parser.add_argument("--use-doc-unwarping", action="store_true")
    parser.add_argument("--use-chart-recognition", action="store_true")
    parser.add_argument("--include-layout-blocks", action="store_true")
    parser.add_argument("--structured-intermediate", action="store_true")
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
