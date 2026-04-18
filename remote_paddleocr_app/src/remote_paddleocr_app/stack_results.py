from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .eval_pyfi import aggregate, iter_pyfi_jsonl, occurrence_keys


ResultKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class RunSpec:
    name: str
    path: Path


def parse_run(value: str) -> RunSpec:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("run must be NAME=PATH") from exc
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return RunSpec(name=name, path=Path(path))


def parse_preference(value: str) -> tuple[str, str]:
    try:
        bucket, run_name = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("preference must be BUCKET=RUN_NAME") from exc
    return bucket.strip(), run_name.strip()


def read_results(path: Path) -> dict[ResultKey, dict[str, Any]]:
    seen: dict[str, int] = defaultdict(int)
    rows: dict[ResultKey, dict[str, Any]] = {}
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


def stack_results(args: argparse.Namespace) -> dict[str, Any]:
    records = iter_pyfi_jsonl(args.dataset)
    keys = occurrence_keys(records)
    specs: list[RunSpec] = args.run
    if not specs:
        raise RuntimeError("At least one --run is required.")
    run_names = {spec.name for spec in specs}
    if args.default_run not in run_names:
        raise RuntimeError(f"--default-run must be one of: {sorted(run_names)}")
    preferences = dict(args.prefer_capability or [])
    unknown = set(preferences.values()) - run_names
    if unknown:
        raise RuntimeError(f"Unknown preferred run(s): {sorted(unknown)}")

    runs = {spec.name: read_results(spec.path) for spec in specs}
    output_rows: list[dict[str, Any]] = []
    for record, key in zip(records, keys, strict=True):
        run_name = preferences.get(record.capability or "None", args.default_run)
        item = dict(runs[run_name][key])
        item["stack_source_run"] = run_name
        item["stack_policy"] = {
            "default_run": args.default_run,
            "prefer_capability": preferences,
        }
        output_rows.append(item)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for item in output_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics = aggregate(output_rows)
    metrics["stack_policy"] = {
        "default_run": args.default_run,
        "prefer_capability": preferences,
        "runs": [{"name": spec.name, "path": str(spec.path)} for spec in specs],
    }
    metrics_path = args.out.with_suffix(args.out.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def add_stack_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("stack-results", help="Stack remote evaluation JSONL result files.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run", required=True, action="append", type=parse_run)
    parser.add_argument("--default-run", required=True)
    parser.add_argument("--prefer-capability", action="append", type=parse_preference, default=[])


def main() -> None:
    parser = argparse.ArgumentParser(description="Stack remote evaluation result files.")
    add_stack_parser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args()
    stack_results(args)


if __name__ == "__main__":
    main()
