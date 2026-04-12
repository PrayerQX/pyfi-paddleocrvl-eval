from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable


def normalize_answer(raw: Any, valid_options: set[str]) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().upper()
    if text in valid_options:
        return text
    if not text:
        return None
    matches = re.findall(r"\b([A-Z])\b", text)
    for match in matches:
        if match in valid_options:
            return match
    if len(text) == 1 and text in valid_options:
        return text
    return None


def aggregate(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    correct = 0
    invalid = 0
    by_capability: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "invalid": 0}
    )
    by_complexity: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "invalid": 0}
    )

    for item in results:
        total += 1
        is_correct = bool(item.get("correct"))
        is_invalid = item.get("prediction") is None
        correct += int(is_correct)
        invalid += int(is_invalid)

        capability = item.get("capability") or "unknown"
        complexity = str(item.get("complexity") or "unknown")
        for bucket, key in [(by_capability, capability), (by_complexity, complexity)]:
            bucket[key]["total"] += 1
            bucket[key]["correct"] += int(is_correct)
            bucket[key]["invalid"] += int(is_invalid)

    def finalize(bucket: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
        finalized: dict[str, dict[str, float | int]] = {}
        for key, value in sorted(bucket.items()):
            n = value["total"]
            finalized[key] = {
                **value,
                "accuracy": (value["correct"] / n if n else 0.0),
                "invalid_rate": (value["invalid"] / n if n else 0.0),
            }
        return finalized

    return {
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / total if total else 0.0,
        "invalid_rate": invalid / total if total else 0.0,
        "by_capability": finalize(by_capability),
        "by_complexity": finalize(by_complexity),
    }
