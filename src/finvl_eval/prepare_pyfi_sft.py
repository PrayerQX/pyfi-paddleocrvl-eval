from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .prompts import build_mcq_prompt
from .pyfi import iter_jsonl, iter_pyfi_csv
from .records import EvalRecord


COT_KEYS = ("question_chain", "chain", "cot", "reasoning_steps", "reasoning")


def iter_records(dataset: Path, dataset_format: str) -> Iterable[EvalRecord]:
    if dataset_format == "pyfi-csv":
        return iter_pyfi_csv(dataset)
    if dataset_format == "jsonl":
        return iter_jsonl(dataset)
    raise ValueError(f"Unknown dataset format: {dataset_format}")


def build_training_example(record: EvalRecord, *, mode: str, prompt_style: str, images_root: Path | None) -> dict | None:
    if not record.answer:
        return None
    image_path = record.resolved_image_path(images_root)
    prompt = build_mcq_prompt(record, context_mode="image_background_only", prompt_style=prompt_style)
    assistant = record.answer
    if mode == "cot-from-metadata":
        cot = _metadata_cot(record)
        if not cot:
            return None
        assistant = f"{cot}\nFinal answer: {record.answer}"

    return {
        "uid": record.uid,
        "image": str(image_path),
        "mode": mode,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": assistant,
            },
        ],
        "answer": record.answer,
        "capability": record.capability,
        "complexity": record.complexity,
    }


def _metadata_cot(record: EvalRecord) -> str | None:
    for key in COT_KEYS:
        value = record.metadata.get(key)
        if value:
            if isinstance(value, list):
                return "\n".join(str(item) for item in value if str(item).strip())
            return str(value).strip()
    return None


def looks_like_eval_split(path: Path) -> bool:
    name = path.name.lower()
    return "eval" in name or "301" in name


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PyFi records for PaddleOCR-VL LoRA/SFT experiments.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--format", choices=["pyfi-csv", "jsonl"], default="jsonl")
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=["final-only", "cot-from-metadata"], default="final-only")
    parser.add_argument("--prompt-style", choices=["direct", "pyramid"], default="pyramid")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--allow-eval-split",
        action="store_true",
        help="Allow preparing SFT data from an eval-looking split. Use only for smoke tests.",
    )
    args = parser.parse_args()

    if looks_like_eval_split(args.dataset) and not args.allow_eval_split:
        raise SystemExit(
            "Refusing to prepare training data from an eval-looking split. "
            "Use a training split, or pass --allow-eval-split for a smoke test only."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for record in iter_records(args.dataset, args.format):
            if args.limit is not None and written >= args.limit:
                break
            example = build_training_example(
                record,
                mode=args.mode,
                prompt_style=args.prompt_style,
                images_root=args.images_root,
            )
            if example is None:
                skipped += 1
                continue
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1

    summary = {"written": written, "skipped": skipped, "mode": args.mode, "out": str(args.out)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
