from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def safe_uid(uid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", uid)[:180]


def classify_invalid(raw_prediction: Any, error: str | None, evidence: str) -> str:
    if error:
        return "runtime_error"
    raw = "" if raw_prediction is None else str(raw_prediction).strip()
    lower = raw.lower()
    if raw == "" or lower in {"none", "null", "n/a"}:
        if not evidence.strip():
            return "empty_evidence_and_null"
        return "selector_null"
    if "insufficient" in lower or "cannot" in lower or "not include" in lower:
        return "selector_declared_insufficient_evidence"
    if not evidence.strip():
        return "empty_evidence"
    return "unparseable_option_output"


def read_artifact_text(artifacts_dir: Path | None, uid: str) -> str:
    if artifacts_dir is None:
        return ""
    base = safe_uid(uid)
    for suffix in [".md", ".txt"]:
        path = artifacts_dir / f"{base}{suffix}"
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Markdown audit for invalid JSONL predictions.")
    parser.add_argument("--results", required=True, help="Prediction JSONL")
    parser.add_argument("--artifacts-dir", help="Directory containing .md/.txt evidence artifacts")
    parser.add_argument("--out", required=True, help="Markdown report path")
    parser.add_argument("--max-examples", type=int, default=30)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else None
    records = [
        json.loads(line)
        for line in Path(args.results).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    invalid = [item for item in records if item.get("prediction") is None]

    by_reason: Counter[str] = Counter()
    by_capability: Counter[str] = Counter()
    by_complexity: Counter[str] = Counter()
    examples: list[tuple[dict[str, Any], str, str]] = []

    for item in invalid:
        evidence = read_artifact_text(artifacts_dir, item["uid"])
        reason = classify_invalid(item.get("raw_prediction"), item.get("error"), evidence)
        by_reason[reason] += 1
        by_capability[item.get("capability") or "unknown"] += 1
        by_complexity[str(item.get("complexity") or "unknown")] += 1
        if len(examples) < args.max_examples:
            examples.append((item, reason, evidence))

    lines = [
        "# Invalid Sample Audit",
        "",
        f"- Results: `{args.results}`",
        f"- Total records: {len(records)}",
        f"- Invalid records: {len(invalid)}",
        "",
        "## By Reason",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in by_reason.most_common())

    lines.extend(["", "## By Capability", "", "| Capability | Count |", "|---|---:|"])
    lines.extend(f"| {key} | {value} |" for key, value in sorted(by_capability.items()))

    lines.extend(["", "## By Complexity", "", "| Complexity | Count |", "|---|---:|"])
    lines.extend(f"| {key} | {value} |" for key, value in sorted(by_complexity.items()))

    lines.extend(["", "## Examples", ""])
    for item, reason, evidence in examples:
        snippet = evidence.strip().replace("\r\n", "\n")[:800]
        raw = "" if item.get("raw_prediction") is None else str(item.get("raw_prediction"))
        lines.extend(
            [
                f"### {item['uid']}",
                "",
                f"- Reason: `{reason}`",
                f"- Capability: `{item.get('capability')}`",
                f"- Complexity: `{item.get('complexity')}`",
                f"- Gold answer: `{item.get('answer')}`",
                f"- Raw prediction: `{raw[:500]}`",
                "",
                "Evidence snippet:",
                "",
                "```text",
                snippet,
                "```",
                "",
            ]
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote audit report to {out_path}")


if __name__ == "__main__":
    main()
