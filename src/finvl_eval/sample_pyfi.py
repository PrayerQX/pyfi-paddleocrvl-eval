from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from .pyfi import iter_pyfi_csv, write_jsonl
from .records import EvalRecord


def sample_records(
    csv_path: str | Path,
    limit: int,
    seed: int = 20260412,
    stratify: str = "capability",
    images_root: str | Path | None = None,
    require_image: bool = False,
) -> list[EvalRecord]:
    rng = random.Random(seed)
    if stratify not in {"none", "capability", "complexity"}:
        raise ValueError("stratify must be one of: none, capability, complexity")

    if stratify == "none":
        reservoir: list[EvalRecord] = []
        seen = 0
        for record in iter_pyfi_csv(csv_path):
            if require_image and not record.resolved_image_path(images_root).exists():
                continue
            seen += 1
            if len(reservoir) < limit:
                reservoir.append(record)
            else:
                j = rng.randrange(seen)
                if j < limit:
                    reservoir[j] = record
        return reservoir

    buckets: dict[str, list[EvalRecord]] = defaultdict(list)
    seen_by_bucket: dict[str, int] = defaultdict(int)
    for record in iter_pyfi_csv(csv_path):
        if require_image and not record.resolved_image_path(images_root).exists():
            continue
        key = record.capability if stratify == "capability" else record.complexity
        key = key or "unknown"
        seen_by_bucket[key] += 1
        bucket = buckets[key]
        per_bucket_limit = max(limit, 1)
        if len(bucket) < per_bucket_limit:
            bucket.append(record)
        else:
            j = rng.randrange(seen_by_bucket[key])
            if j < per_bucket_limit:
                bucket[j] = record

    ordered_keys = sorted(buckets)
    selected: list[EvalRecord] = []
    while len(selected) < limit and any(buckets.values()):
        for key in ordered_keys:
            if buckets[key] and len(selected) < limit:
                idx = rng.randrange(len(buckets[key]))
                selected.append(buckets[key].pop(idx))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small JSONL evaluation split from PyFi CSV.")
    parser.add_argument("--csv", required=True, help="Path to PyFi-600K-dataset.csv")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=301, help="Number of samples")
    parser.add_argument("--seed", type=int, default=20260412)
    parser.add_argument("--stratify", choices=["none", "capability", "complexity"], default="capability")
    parser.add_argument("--images-root", help="Root containing the images/ directory")
    parser.add_argument("--require-image", action="store_true", help="Drop rows whose image is missing")
    args = parser.parse_args()

    records = sample_records(
        args.csv,
        limit=args.limit,
        seed=args.seed,
        stratify=args.stratify,
        images_root=args.images_root,
        require_image=args.require_image,
    )
    count = write_jsonl(records, args.out)
    print(f"Wrote {count} records to {args.out}")


if __name__ == "__main__":
    main()
