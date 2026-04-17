from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .models import _safe_uid
from .pyfi import iter_jsonl
from .records import EvalRecord
from .scoring import aggregate
from .veto import VetoConfig, check_veto


def _read_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_records(path: Path) -> dict[str, EvalRecord]:
    return {record.uid: record for record in iter_jsonl(path)}


def _artifact_text(directory: Path | None, uid: str, suffix: str) -> str:
    if directory is None:
        return ""
    path = directory / f"{_safe_uid(uid)}.{suffix}"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _votes_from_raw(raw_prediction: Any) -> dict[str, int] | None:
    if not raw_prediction:
        return None
    try:
        parsed = json.loads(str(raw_prediction))
    except ValueError:
        return None
    votes = parsed.get("votes") if isinstance(parsed, dict) else None
    if not isinstance(votes, dict):
        return None
    return {str(key): int(value) for key, value in votes.items()}


def apply_veto(args: argparse.Namespace) -> dict[str, Any]:
    records = _read_records(args.dataset)
    rows = _read_results(args.results)
    cfg = VetoConfig(
        enabled=True,
        threshold=args.veto_threshold,
        enable_evidence_check=not args.veto_no_evidence_check,
        enable_contradiction_check=not args.veto_no_contradiction,
        safe_mode=not args.veto_unsafe_mode,
        min_fallback_support=args.veto_min_fallback_support,
        min_support_margin=args.veto_min_support_margin,
        max_original_support=args.veto_max_original_support,
        max_veto_vote_confidence=args.veto_max_vote_confidence,
    )

    out_rows: list[dict[str, Any]] = []
    vetoes = 0
    fixed = 0
    broken = 0
    for item in rows:
        record = records.get(item["uid"])
        if record is None:
            continue
        evidence = "\n\n".join(
            [
                _artifact_text(args.artifacts_dir, item["uid"], "md"),
                _artifact_text(args.ocr_artifacts_dir, item["uid"], "txt"),
            ]
        )
        result = check_veto(
            prediction=item.get("prediction"),
            record=record,
            evidence_text=evidence,
            raw_prediction=item.get("raw_prediction"),
            votes=_votes_from_raw(item.get("raw_prediction")),
            config=cfg,
        )
        new_item = dict(item)
        if result.vetoed and result.fallback_prediction:
            vetoes += 1
            was_correct = bool(item.get("correct"))
            new_item["prediction"] = result.fallback_prediction
            new_item["raw_prediction"] = json.dumps(
                {
                    "answer": result.fallback_prediction,
                    "veto": result.to_dict(),
                    "original_raw_prediction": item.get("raw_prediction"),
                },
                ensure_ascii=False,
            )
            new_item["correct"] = result.fallback_prediction == record.answer if record.answer else False
            if not was_correct and new_item["correct"]:
                fixed += 1
            if was_correct and not new_item["correct"]:
                broken += 1
        out_rows.append(new_item)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for item in out_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics = aggregate(out_rows)
    metrics["vetoes"] = vetoes
    metrics["fixed"] = fixed
    metrics["broken"] = broken
    args.out.with_suffix(args.out.suffix + ".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply veto rules offline to an existing JSONL result file.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--ocr-artifacts-dir", type=Path)
    parser.add_argument("--veto-threshold", type=float, default=0.4)
    parser.add_argument("--veto-no-evidence-check", action="store_true")
    parser.add_argument("--veto-no-contradiction", action="store_true")
    parser.add_argument("--veto-unsafe-mode", action="store_true")
    parser.add_argument("--veto-min-fallback-support", type=float, default=0.9)
    parser.add_argument("--veto-min-support-margin", type=float, default=0.45)
    parser.add_argument("--veto-max-original-support", type=float, default=0.25)
    parser.add_argument("--veto-max-vote-confidence", type=float, default=2 / 3)
    args = parser.parse_args()
    apply_veto(args)


if __name__ == "__main__":
    main()
