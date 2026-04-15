from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .scoring import aggregate


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_run(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Run must use NAME=PATH format: {raw}")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Run name is empty: {raw}")
    return name, Path(path)


def build_report(dataset: Path, runs: list[tuple[str, Path]], title: str) -> str:
    dataset_rows = _read_jsonl(dataset)
    rows = []
    for name, path in runs:
        results = _read_jsonl(path)
        metrics = aggregate(results)
        rows.append(
            {
                "name": name,
                "path": path,
                "total": metrics["total"],
                "correct": metrics["correct"],
                "invalid": metrics["invalid"],
                "accuracy": metrics["accuracy"],
                "invalid_rate": metrics["invalid_rate"],
                "metrics": metrics,
            }
        )

    ranked = sorted(rows, key=lambda item: (item["accuracy"], -item["invalid_rate"]), reverse=True)
    lines = [
        f"# {title}",
        "",
        "## Split",
        "",
        f"- Dataset: `{dataset}`",
        f"- Records: {len(dataset_rows)}",
        f"- SHA256: `{_sha256(dataset)}`",
    ]
    if dataset_rows:
        lines.extend(
            [
                f"- First UID: `{dataset_rows[0].get('uid')}`",
                f"- Last UID: `{dataset_rows[-1].get('uid')}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| Rank | Method | Total | Correct | Accuracy | Invalid | Invalid Rate | Result File |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for idx, item in enumerate(ranked, start=1):
        lines.append(
            "| "
            f"{idx} | {item['name']} | {item['total']} | {item['correct']} | "
            f"{item['accuracy']:.2%} | {item['invalid']} | {item['invalid_rate']:.2%} | "
            f"`{item['path']}` |"
        )

    lines.extend(["", "## Capability Accuracy", ""])
    for item in ranked:
        by_capability = item["metrics"].get("by_capability", {})
        parts = [
            f"{capability}: {value.get('accuracy', 0.0):.2%}"
            for capability, value in by_capability.items()
        ]
        lines.append(f"- **{item['name']}**: " + "; ".join(parts))

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "Use the same dataset path and SHA256 above. Each result file is plain JSONL and can be rescored with:",
            "",
            "```powershell",
            "python -m finvl_eval.compare_runs `",
            f"  --dataset {dataset} `",
            "  --out docs/pyfi_local301_reusable_leaderboard.md `",
        ]
    )
    for name, path in runs:
        lines.append(f"  --run \"{name}={path}\" `")
    if lines[-1].endswith(" `"):
        lines[-1] = lines[-1][:-2]
    lines.extend(["```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reusable local leaderboard from JSONL result files.")
    parser.add_argument("--dataset", required=True, type=Path, help="Fixed evaluation JSONL split")
    parser.add_argument("--run", action="append", required=True, help="Run in NAME=PATH format")
    parser.add_argument("--out", required=True, type=Path, help="Markdown report path")
    parser.add_argument("--title", default="PyFi Local Reusable Leaderboard")
    args = parser.parse_args()

    report = build_report(args.dataset, [_parse_run(raw) for raw in args.run], args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
