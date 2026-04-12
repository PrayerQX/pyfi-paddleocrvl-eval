from __future__ import annotations

import base64
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Protocol

from .records import EvalRecord
from .scoring import normalize_answer


class ModelAdapter(Protocol):
    name: str

    def predict(self, record: EvalRecord, image_path: Path, prompt: str) -> str | None:
        ...


class FirstOptionAdapter:
    name = "first_option"

    def predict(self, record: EvalRecord, image_path: Path, prompt: str) -> str | None:
        return next(iter(record.options), None)


class RandomOptionAdapter:
    name = "random_option"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def predict(self, record: EvalRecord, image_path: Path, prompt: str) -> str | None:
        if not record.options:
            return None
        return self._rng.choice(list(record.options))


class OpenAICompatibleVLMAdapter:
    """Direct VLM adapter for OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.name = model
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.timeout = timeout

    def predict(self, record: EvalRecord, image_path: Path, prompt: str) -> str | None:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for openai-compatible-vlm.")

        from openai import OpenAI

        mime = _mime_type(image_path)
        image_payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            max_tokens=128,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_payload}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content


class PaddleOCRTextDocQAAdapter:
    """
    Traditional OCR + selector baseline.

    This baseline intentionally uses text OCR only. It does not use
    PaddleOCR-VL chart/table parsing, so it gives a lower-complexity comparison
    point for PyFi-style financial image QA.
    """

    name = "paddleocr-text-docqa"

    def __init__(
        self,
        selector_model: str | None = None,
        selector_base_url: str | None = None,
        selector_api_key: str | None = None,
        artifacts_dir: str | Path | None = None,
    ) -> None:
        self.selector_model = selector_model or os.getenv("FINVL_SELECTOR_MODEL")
        self.selector_base_url = selector_base_url or os.getenv("OPENAI_BASE_URL")
        self.selector_api_key = selector_api_key or os.getenv("OPENAI_API_KEY")
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self._ocr = None

    def predict(self, record: EvalRecord, image_path: Path, prompt: str) -> str | None:
        ocr_text = self.parse_image(image_path, record.uid)
        if not self.selector_model:
            return None
        return self._select_with_openai(record, ocr_text)

    def parse_image(self, image_path: Path, uid: str) -> str:
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

        results = self._ocr.predict(str(image_path))
        text_parts: list[str] = []
        json_parts: list[dict[str, Any]] = []
        for result in results:
            try:
                res_json = result.json["res"]
                json_parts.append(res_json)
                text_parts.extend(str(text) for text in res_json.get("rec_texts", []))
            except Exception:
                json_parts.append({"raw": str(result)})
                text_parts.append(str(result))

        parsed_text = "\n".join(text_parts).strip()
        if self.artifacts_dir:
            safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", uid)[:180]
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            (self.artifacts_dir / f"{safe_uid}.txt").write_text(parsed_text, encoding="utf-8")
            (self.artifacts_dir / f"{safe_uid}.json").write_text(
                json.dumps(json_parts, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return parsed_text

    def _select_with_openai(self, record: EvalRecord, ocr_text: str) -> str | None:
        if not self.selector_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when FINVL_SELECTOR_MODEL is set.")
        from openai import OpenAI

        client = OpenAI(
            api_key=self.selector_api_key,
            base_url=(self.selector_base_url or "https://api.openai.com/v1"),
            timeout=120,
        )
        prompt = (
            "You are selecting an answer to a multiple-choice question using only "
            "traditional OCR text from a financial image.\n\n"
            f"OCR text:\n{ocr_text[:12000]}\n\n"
            f"Question:\n{record.question}\n\n"
            f"Options:\n{json.dumps(record.options, ensure_ascii=False)}\n\n"
            "Output only the single option letter. If the OCR text is insufficient, output null."
        )
        response = client.chat.completions.create(
            model=self.selector_model,
            temperature=0.1,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class PaddleOCRVLDocQAAdapter:
    """
    PaddleOCR-VL 1.5 parser + answer selector.

    PaddleOCR-VL is a document parsing pipeline. For PyFi's multiple-choice
    QA format this adapter first parses the financial image to markdown, then
    selects an option from that markdown. Use selector_model for real runs.
    """

    name = "paddleocr-vl-1.5-docqa"

    def __init__(
        self,
        selector_model: str | None = None,
        selector_base_url: str | None = None,
        selector_api_key: str | None = None,
        vl_rec_backend: str | None = None,
        vl_rec_server_url: str | None = None,
        vl_rec_model_dir: str | None = None,
        artifacts_dir: str | Path | None = None,
        use_chart_recognition: bool = True,
    ) -> None:
        self.selector_model = selector_model or os.getenv("FINVL_SELECTOR_MODEL")
        self.selector_base_url = selector_base_url or os.getenv("OPENAI_BASE_URL")
        self.selector_api_key = selector_api_key or os.getenv("OPENAI_API_KEY")
        self.vl_rec_backend = vl_rec_backend or os.getenv("PADDLEOCR_VL_BACKEND")
        self.vl_rec_server_url = vl_rec_server_url or os.getenv("PADDLEOCR_VL_SERVER_URL")
        self.vl_rec_model_dir = vl_rec_model_dir or os.getenv("PADDLEOCR_VL_MODEL_DIR")
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self.use_chart_recognition = use_chart_recognition
        self._pipeline = None

    def predict(self, record: EvalRecord, image_path: Path, prompt: str) -> str | None:
        parsed_markdown = self.parse_image(image_path, record.uid)
        if self.selector_model:
            return self._select_with_openai(record, parsed_markdown)
        return self._select_with_lexical_heuristic(record, parsed_markdown)

    def parse_image(self, image_path: Path, uid: str) -> str:
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL

            kwargs: dict[str, Any] = {
                "pipeline_version": "v1.5",
                "use_chart_recognition": self.use_chart_recognition,
                "format_block_content": True,
                "merge_layout_blocks": True,
            }
            if self.vl_rec_backend:
                kwargs["vl_rec_backend"] = self.vl_rec_backend
            if self.vl_rec_server_url:
                kwargs["vl_rec_server_url"] = self.vl_rec_server_url
            if self.vl_rec_model_dir:
                kwargs["vl_rec_model_dir"] = self.vl_rec_model_dir
            self._pipeline = PaddleOCRVL(**kwargs)

        results = self._pipeline.predict(str(image_path))
        markdown_parts: list[str] = []
        json_parts: list[dict[str, Any]] = []
        for result in results:
            markdown = getattr(result, "markdown", None)
            if isinstance(markdown, dict):
                markdown_parts.append(str(markdown.get("markdown_texts") or ""))
            else:
                try:
                    markdown_parts.append(str(result.markdown.get("markdown_texts") or ""))
                except Exception:
                    markdown_parts.append(str(result))

            try:
                json_parts.append(result.json["res"])
            except Exception:
                json_parts.append({"raw": str(result)})

        parsed_text = "\n\n".join(part for part in markdown_parts if part).strip()
        if self.artifacts_dir:
            safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", uid)[:180]
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            (self.artifacts_dir / f"{safe_uid}.md").write_text(parsed_text, encoding="utf-8")
            (self.artifacts_dir / f"{safe_uid}.json").write_text(
                json.dumps(json_parts, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return parsed_text

    def _select_with_openai(self, record: EvalRecord, parsed_markdown: str) -> str | None:
        if not self.selector_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when FINVL_SELECTOR_MODEL is set.")
        from openai import OpenAI

        client = OpenAI(
            api_key=self.selector_api_key,
            base_url=(self.selector_base_url or "https://api.openai.com/v1"),
            timeout=120,
        )
        prompt = (
            "You are selecting an answer to a multiple-choice question using only "
            "PaddleOCR-VL parsed markdown from a financial image.\n\n"
            f"Parsed markdown:\n{parsed_markdown[:12000]}\n\n"
            f"Question:\n{record.question}\n\n"
            f"Options:\n{json.dumps(record.options, ensure_ascii=False)}\n\n"
            "Output only the single option letter. If the parsed markdown is insufficient, output null."
        )
        response = client.chat.completions.create(
            model=self.selector_model,
            temperature=0.1,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def _select_with_lexical_heuristic(self, record: EvalRecord, parsed_markdown: str) -> str | None:
        text = parsed_markdown.lower()
        hits: list[tuple[int, str]] = []
        for letter, option in record.options.items():
            option_text = str(option).strip().lower()
            if not option_text:
                continue
            count = text.count(option_text)
            if count:
                hits.append((count, letter))
        if not hits:
            return None
        hits.sort(reverse=True)
        return hits[0][1]


def build_adapter(args: Any) -> ModelAdapter:
    if args.model == "first-option":
        return FirstOptionAdapter()
    if args.model == "random-option":
        return RandomOptionAdapter(seed=args.seed)
    if args.model == "openai-compatible-vlm":
        return OpenAICompatibleVLMAdapter(model=args.openai_model)
    if args.model == "paddleocr-text-docqa":
        return PaddleOCRTextDocQAAdapter(
            selector_model=args.selector_model,
            artifacts_dir=args.artifacts_dir,
        )
    if args.model == "paddleocr-vl-docqa":
        return PaddleOCRVLDocQAAdapter(
            selector_model=args.selector_model,
            artifacts_dir=args.artifacts_dir,
            vl_rec_backend=args.paddle_vl_backend,
            vl_rec_server_url=args.paddle_vl_server_url,
            vl_rec_model_dir=args.paddle_vl_model_dir,
        )
    raise ValueError(f"Unknown model adapter: {args.model}")


def _mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def normalize_prediction(record: EvalRecord, raw_prediction: str | None) -> str | None:
    return normalize_answer(raw_prediction, record.valid_options)
