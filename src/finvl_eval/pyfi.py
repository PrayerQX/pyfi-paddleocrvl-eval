from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .records import EvalRecord


PYFI_FIELDNAMES = {
    "question_node_no",
    "options",
    "complexity",
    "visit_count",
    "fq_no",
    "capability",
    "victory_count",
    "image_background",
    "parent_node_no",
    "actions",
    "question",
    "image_path",
}


def _parse_structured_value(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return fallback


def extract_gold_answer(actions: Any) -> str | None:
    parsed = _parse_structured_value(actions, [])
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        return None

    candidates: list[dict[str, Any]] = [item for item in parsed if isinstance(item, dict)]
    if not candidates:
        return None

    def sort_key(item: dict[str, Any]) -> tuple[int, int]:
        return (
            int(item.get("victory_count") or 0),
            int(item.get("visit_count") or 0),
        )

    best = max(candidates, key=sort_key)
    answer = best.get("answer")
    if answer is None:
        return None
    answer = str(answer).strip().upper()
    return answer or None


def record_from_pyfi_row(row: dict[str, Any]) -> EvalRecord:
    options = _parse_structured_value(row.get("options"), {})
    if not isinstance(options, dict):
        options = {}
    options = {str(key).strip().upper(): str(value) for key, value in options.items()}

    question_node_no = str(row.get("question_node_no", "")).strip()
    fq_no = str(row.get("fq_no", "")).strip()
    image_path = str(row.get("image_path", "")).strip()
    uid = f"{image_path}::fq{fq_no}::node{question_node_no}"

    return EvalRecord(
        uid=uid,
        image_path=image_path,
        question=str(row.get("question", "")).strip(),
        options=options,
        answer=extract_gold_answer(row.get("actions")),
        capability=str(row.get("capability", "") or "").strip() or None,
        complexity=str(row.get("complexity", "") or "").strip() or None,
        context={
            "image_background": str(row.get("image_background", "") or "").strip(),
        },
        metadata={
            "fq_no": fq_no,
            "question_node_no": question_node_no,
            "parent_node_no": str(row.get("parent_node_no", "") or "").strip(),
            "visit_count": str(row.get("visit_count", "") or "").strip(),
            "victory_count": str(row.get("victory_count", "") or "").strip(),
            "actions": _parse_structured_value(row.get("actions"), []),
        },
    )


def iter_pyfi_csv(path: str | Path, encoding: str = "utf-8-sig") -> Iterator[EvalRecord]:
    with Path(path).open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        missing = PYFI_FIELDNAMES.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing PyFi CSV columns: {sorted(missing)}")
        for row in reader:
            record = record_from_pyfi_row(row)
            if record.question and record.options and record.image_path:
                yield record


def iter_jsonl(path: str | Path, encoding: str = "utf-8") -> Iterator[EvalRecord]:
    with Path(path).open("r", encoding=encoding) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield EvalRecord(
                uid=str(data.get("uid") or f"{path}:{line_no}"),
                image_path=str(data["image_path"]),
                question=str(data["question"]),
                options={str(k).upper(): str(v) for k, v in data["options"].items()},
                answer=(str(data["answer"]).upper() if data.get("answer") else None),
                capability=data.get("capability"),
                complexity=str(data["complexity"]) if data.get("complexity") is not None else None,
                context=dict(data.get("context") or {}),
                metadata=dict(data.get("metadata") or {}),
            )


def write_jsonl(records: Iterable[EvalRecord], path: str | Path) -> int:
    count = 0
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
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
                + "\n"
            )
            count += 1
    return count
