from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pyfi import iter_jsonl
from .records import EvalRecord


OccurrenceKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class RunInput:
    name: str
    path: Path


def _occurrence_key(uid: str, seen: dict[str, int]) -> OccurrenceKey:
    occurrence = seen[uid]
    seen[uid] += 1
    return uid, occurrence


def _parse_run(value: str) -> RunInput:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "run must be NAME=PATH, for example adjudicated=runs/out.jsonl"
        ) from exc
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return RunInput(name=name, path=Path(path))


def _parse_min_weight(value: str) -> tuple[str, int]:
    try:
        name, weight = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("min weight must be NAME=WEIGHT") from exc
    try:
        parsed = int(weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid min weight: {weight}") from exc
    return name, parsed


def _read_records(path: Path) -> list[EvalRecord]:
    return list(iter_jsonl(path))


def _read_results(path: Path) -> dict[OccurrenceKey, dict[str, Any]]:
    rows: dict[OccurrenceKey, dict[str, Any]] = {}
    seen: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows[_occurrence_key(item["uid"], seen)] = item
    return rows


def _bucket(record: EvalRecord, field: str) -> str:
    if field == "capability":
        return str(record.capability or "None")
    if field == "complexity":
        return str(record.complexity or "None")
    raise RuntimeError("field must be capability or complexity")


def _predict(
    key: OccurrenceKey,
    valid_options: set[str],
    names: list[str],
    runs: dict[str, dict[OccurrenceKey, dict[str, Any]]],
    weights: dict[str, int],
) -> str | None:
    scores: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for index, name in enumerate(names):
        item = runs[name].get(key)
        prediction = item.get("prediction") if item else None
        if prediction not in valid_options:
            continue
        weight = weights[name]
        if weight <= 0:
            continue
        scores[prediction] += weight
        first_seen.setdefault(prediction, index)
    if not scores:
        return None
    best_score = max(scores.values())
    tied = [prediction for prediction, score in scores.items() if score == best_score]
    tied.sort(key=lambda prediction: first_seen[prediction])
    return tied[0]


def _score(
    rows: list[tuple[OccurrenceKey, EvalRecord]],
    names: list[str],
    runs: dict[str, dict[OccurrenceKey, dict[str, Any]]],
    weights: dict[str, int],
) -> int:
    correct = 0
    for key, record in rows:
        prediction = _predict(key, record.valid_options, names, runs, weights)
        correct += int(prediction == record.answer)
    return correct


def tune(args: argparse.Namespace) -> dict[str, Any]:
    records = _read_records(args.dataset)
    seen_records: dict[str, int] = defaultdict(int)
    keyed_records = []
    for record in records:
        keyed_records.append((_occurrence_key(record.uid, seen_records), record))

    run_inputs: list[RunInput] = args.run
    names = [item.name for item in run_inputs]
    runs = {item.name: _read_results(item.path) for item in run_inputs}
    min_weights = dict(args.min_weight or [])

    profiles: dict[str, dict[str, int]] = {}
    report: dict[str, Any] = {}
    for bucket in sorted({_bucket(record, args.field) for _, record in keyed_records}):
        rows = [(key, record) for key, record in keyed_records if _bucket(record, args.field) == bucket]
        best_score = -1
        best_weights: dict[str, int] | None = None
        for values in itertools.product(range(args.min_search_weight, args.max_search_weight + 1), repeat=len(names)):
            weights = dict(zip(names, values))
            if any(weights.get(name, 0) < minimum for name, minimum in min_weights.items()):
                continue
            score = _score(rows, names, runs, weights)
            if score > best_score:
                best_score = score
                best_weights = weights
        if best_weights is None:
            raise RuntimeError(f"No valid weights found for bucket {bucket}.")
        profiles[bucket] = best_weights
        report[bucket] = {
            "correct": best_score,
            "total": len(rows),
            "accuracy": best_score / len(rows) if rows else 0.0,
            "weights": best_weights,
        }

    output = {
        "field": args.field,
        "profiles": profiles,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "report": report}, ensure_ascii=False, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune stack-run weights on a labeled calibration split."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--field", choices=["capability", "complexity"], default="capability")
    parser.add_argument("--run", required=True, action="append", type=_parse_run)
    parser.add_argument("--min-search-weight", type=int, default=1)
    parser.add_argument("--max-search-weight", type=int, default=5)
    parser.add_argument(
        "--min-weight",
        action="append",
        type=_parse_min_weight,
        help="Optional minimum for a run weight, for example direct=2.",
    )
    args = parser.parse_args()
    tune(args)


if __name__ == "__main__":
    main()
