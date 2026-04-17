from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


def save_layout_result(result: dict[str, Any], output_dir: str | Path, timeout: float = 180) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    result_path = out / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(result_path)

    for index, res in enumerate(result.get("layoutParsingResults", [])):
        markdown = res.get("markdown") or {}
        markdown_text = markdown.get("text") or ""
        md_path = out / f"doc_{index}.md"
        md_path.write_text(markdown_text, encoding="utf-8")
        written.append(md_path)

        markdown_images = markdown.get("images") or {}
        for image_path, image_url in markdown_images.items():
            full_path = out / "markdown_images" / image_path
            _download_image(str(image_url), full_path, timeout)
            written.append(full_path)

        output_images = res.get("outputImages") or {}
        for image_name, image_url in output_images.items():
            full_path = out / "output_images" / f"{image_name}_{index}.jpg"
            _download_image(str(image_url), full_path, timeout)
            written.append(full_path)

    return written


def collect_markdown(result: dict[str, Any]) -> str:
    documents: list[str] = []
    for index, res in enumerate(result.get("layoutParsingResults", [])):
        markdown = res.get("markdown") or {}
        text = str(markdown.get("text") or "").strip()
        if text:
            documents.append(f"# Document {index}\n\n{text}")
    return "\n\n---\n\n".join(documents)


def _download_image(url: str, path: Path, timeout: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download image {url}: HTTP {response.status_code}")
    path.write_bytes(response.content)
