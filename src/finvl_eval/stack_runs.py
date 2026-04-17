from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pyfi import iter_jsonl
from .records import EvalRecord
from .scoring import aggregate


OccurrenceKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class RunSpec:
    name: str
    path: Path
    weight: float


@dataclass(frozen=True, slots=True)
class WeightProfile:
    field: str
    profiles: dict[str, dict[str, float]]


def _occurrence_key(uid: str, seen: dict[str, int]) -> OccurrenceKey:
    occurrence = seen[uid]
    seen[uid] += 1
    return uid, occurrence


def _read_records(path: Path) -> list[EvalRecord]:
    return list(iter_jsonl(path))


def _read_results(path: Path) -> dict[OccurrenceKey, dict[str, Any]]:
    rows: dict[OccurrenceKey, dict[str, Any]] = {}
    seen: dict[str, int] = defaultdict(int)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows[_occurrence_key(item["uid"], seen)] = item
    return rows


def _parse_run_spec(value: str) -> RunSpec:
    try:
        name, path, weight = value.split("=", 2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "run spec must be NAME=PATH=WEIGHT, for example adjudicated=runs/out.jsonl=4"
        ) from exc
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    try:
        parsed_weight = float(weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid run weight: {weight}") from exc
    if parsed_weight < 0:
        raise argparse.ArgumentTypeError("run weight must be non-negative")
    return RunSpec(name=name, path=Path(path), weight=parsed_weight)


def _read_weight_profile(path: Path | None) -> WeightProfile | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    field = str(data.get("field") or "").strip()
    if field not in {"capability", "complexity"}:
        raise RuntimeError("profile field must be either 'capability' or 'complexity'.")
    raw_profiles = data.get("profiles")
    if not isinstance(raw_profiles, dict):
        raise RuntimeError("profile weights file must contain a 'profiles' object.")

    profiles: dict[str, dict[str, float]] = {}
    for bucket, weights in raw_profiles.items():
        if not isinstance(weights, dict):
            raise RuntimeError(f"profile '{bucket}' must be an object of run weights.")
        profiles[str(bucket)] = {str(name): float(weight) for name, weight in weights.items()}
    return WeightProfile(field=field, profiles=profiles)


def _record_bucket(record: EvalRecord, profile: WeightProfile) -> str:
    if profile.field == "capability":
        return str(record.capability or "None")
    return str(record.complexity or "None")


def _weights_for_record(
    record: EvalRecord,
    specs: list[RunSpec],
    profile: WeightProfile | None,
) -> tuple[dict[str, float], str | None]:
    base_weights = {spec.name: spec.weight for spec in specs}
    if profile is None:
        return base_weights, None

    bucket = _record_bucket(record, profile)
    overrides = profile.profiles.get(bucket)
    if not overrides:
        return base_weights, bucket
    weights = dict(base_weights)
    weights.update(overrides)
    return weights, bucket


def _weighted_prediction(
    key: OccurrenceKey,
    valid_options: set[str],
    specs: list[RunSpec],
    runs: dict[str, dict[OccurrenceKey, dict[str, Any]]],
    weights: dict[str, float],
) -> tuple[str | None, dict[str, Any]]:
    scores: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    source_predictions: dict[str, str | None] = {}

    for index, spec in enumerate(specs):
        item = runs[spec.name].get(key)
        prediction = item.get("prediction") if item else None
        if prediction not in valid_options:
            prediction = None
        source_predictions[spec.name] = prediction
        weight = weights.get(spec.name, spec.weight)
        if prediction is None or weight == 0:
            continue
        scores[prediction] += weight
        first_seen.setdefault(prediction, index)

    if not scores:
        return None, {"source_predictions": source_predictions, "weighted_votes": {}}

    best_score = max(scores.values())
    tied = [prediction for prediction, score in scores.items() if score == best_score]
    tied.sort(key=lambda prediction: first_seen[prediction])
    prediction = tied[0]
    return prediction, {
        "source_predictions": source_predictions,
        "weighted_votes": dict(scores),
        "run_weights": {spec.name: weights.get(spec.name, spec.weight) for spec in specs},
        "selected_weight": best_score,
        "tie_break_order": [spec.name for spec in specs],
    }


def stack_runs(args: argparse.Namespace) -> dict[str, Any]:
    specs: list[RunSpec] = args.run
    if not specs:
        raise RuntimeError("At least one --run is required.")

    records = _read_records(args.dataset)
    runs = {spec.name: _read_results(spec.path) for spec in specs}
    profile = _read_weight_profile(args.profile_weights)

    results: list[dict[str, Any]] = []
    seen_records: dict[str, int] = defaultdict(int)
    for record in records:
        key = _occurrence_key(record.uid, seen_records)
        weights, profile_bucket = _weights_for_record(record, specs, profile)
        prediction, decision = _weighted_prediction(key, record.valid_options, specs, runs, weights)
        if profile_bucket is not None:
            decision["profile_field"] = profile.field if profile else None
            decision["profile_bucket"] = profile_bucket
        primary_item = next((runs[spec.name].get(key) for spec in specs if runs[spec.name].get(key)), None)
        item = dict(primary_item or {})
        item.update(
            {
                "uid": record.uid,
                "prediction": prediction,
                "answer": record.answer,
                "correct": prediction == record.answer if record.answer else False,
                "capability": record.capability,
                "complexity": record.complexity,
                "error": None if prediction else "no_weighted_prediction",
                "raw_prediction": json.dumps(
                    {
                        "answer": prediction,
                        "stacking": decision,
                    },
                    ensure_ascii=False,
                ),
            }
        )
        results.append(item)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics = aggregate(results)
    metrics["stacking_runs"] = [
        {"name": spec.name, "path": str(spec.path), "weight": spec.weight} for spec in specs
    ]
    if profile is not None:
        metrics["weight_profile"] = {
            "field": profile.field,
            "profiles": profile.profiles,
        }
    metrics_path = args.out.with_suffix(args.out.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Weighted-stack existing JSONL prediction runs.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--run",
        required=True,
        action="append",
        type=_parse_run_spec,
        help="Weighted run spec: NAME=PATH=WEIGHT. Ties are broken by --run order.",
    )
    parser.add_argument(
        "--profile-weights",
        type=Path,
        help="Optional JSON file with per-capability or per-complexity run weight overrides.",
    )
    args = parser.parse_args()
    stack_runs(args)


if __name__ == "__main__":
    main()
