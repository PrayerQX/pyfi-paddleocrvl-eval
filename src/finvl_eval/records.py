from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EvalRecord:
    """Normalized multiple-choice VQA/document-QA record."""

    uid: str
    image_path: str
    question: str
    options: dict[str, str]
    answer: str | None
    capability: str | None = None
    complexity: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolved_image_path(self, images_root: str | Path | None = None) -> Path:
        raw_path = self.image_path.replace("\\", "/")
        while raw_path.startswith("./"):
            raw_path = raw_path[2:]
        path = Path(raw_path)
        if path.is_absolute() or images_root is None:
            return path
        return Path(images_root) / path

    @property
    def valid_options(self) -> set[str]:
        return {key.upper() for key in self.options}
