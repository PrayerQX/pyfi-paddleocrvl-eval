from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import requests

FileType = Literal["pdf", "image"]


@dataclass(frozen=True, slots=True)
class LayoutOptions:
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_chart_recognition: bool = False


class PaddleOCRRemoteClient:
    def __init__(self, api_url: str, api_token: str, timeout: float = 180) -> None:
        self.api_url = api_url
        self.api_token = api_token
        self.timeout = timeout
        self.session = requests.Session()

    def parse_file(
        self,
        file_path: str | Path,
        file_type: FileType | None = None,
        options: LayoutOptions | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)
        resolved_type = file_type or infer_file_type(path)
        options = options or LayoutOptions()

        file_data = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "file": file_data,
            "fileType": file_type_to_api_value(resolved_type),
            "useDocOrientationClassify": options.use_doc_orientation_classify,
            "useDocUnwarping": options.use_doc_unwarping,
            "useChartRecognition": options.use_chart_recognition,
        }
        headers = {
            "Authorization": f"token {self.api_token}",
            "Content-Type": "application/json",
        }
        response = self.session.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"PaddleOCR API failed with HTTP {response.status_code}: {response.text[:1000]}"
            )
        body = response.json()
        if "result" not in body:
            raise RuntimeError(f"PaddleOCR API response missing result: {json.dumps(body)[:1000]}")
        return body["result"]


def infer_file_type(file_path: Path) -> FileType:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        return "image"
    raise ValueError(f"Cannot infer file type from extension: {file_path.suffix}")


def file_type_to_api_value(file_type: FileType) -> int:
    if file_type == "pdf":
        return 0
    if file_type == "image":
        return 1
    raise ValueError(f"Unsupported file type: {file_type}")
