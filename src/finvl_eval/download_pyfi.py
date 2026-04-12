from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


HF_BASE = "https://huggingface.co/datasets/AgenticFinLab/PyFi-600K/resolve/main"
FILES = {
    "csv": "PyFi-600K-dataset.csv",
    "json": "PyFi-600K-dataset.json",
    "chain": "PyFi-600K-chain-dataset.json",
    "chain_cot": "PyFi-600K-chain-CoT-dataset.json",
    "images": "images.zip",
    "readme": "README.md",
}


def download(url: str, target: Path, chunk_size: int = 1024 * 1024) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as f:
        total = response.headers.get("Content-Length")
        total_int = int(total) if total and total.isdigit() else None
        done = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total_int:
                print(f"{target.name}: {done / total_int:.1%}", end="\r")
    tmp.replace(target)
    print(f"{target.name}: done{' ' * 20}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PyFi-600K files from Hugging Face.")
    parser.add_argument("--out", default="data/pyfi", help="Output directory")
    parser.add_argument(
        "--files",
        nargs="+",
        choices=sorted(FILES),
        default=["readme", "csv"],
        help="Files to download. Use 'images' for images.zip.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    for key in args.files:
        filename = FILES[key]
        download(f"{HF_BASE}/{filename}", out_dir / filename)


if __name__ == "__main__":
    main()
