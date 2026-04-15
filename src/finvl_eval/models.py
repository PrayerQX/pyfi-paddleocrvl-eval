from __future__ import annotations

import base64
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Protocol

from .evidence import build_grounded_evidence
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
        selector_stream: bool | None = None,
        selector_extra_body: dict[str, Any] | None = None,
        selector_max_tokens: int | None = None,
        selector_use_max_completion_tokens: bool | None = None,
    ) -> None:
        self.selector_model = selector_model or os.getenv("FINVL_SELECTOR_MODEL")
        self.selector_base_url = selector_base_url or _selector_base_url()
        self.selector_api_key = selector_api_key or _selector_api_key()
        self.selector_stream = _env_flag("FINVL_SELECTOR_STREAM") if selector_stream is None else selector_stream
        self.selector_extra_body = selector_extra_body or _selector_extra_body()
        self.selector_max_tokens = selector_max_tokens or _selector_max_tokens(self.selector_model)
        self.selector_use_max_completion_tokens = (
            _use_max_completion_tokens(self.selector_model)
            if selector_use_max_completion_tokens is None
            else selector_use_max_completion_tokens
        )
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
        self._ocr = None

    def predict(self, record: EvalRecord, image_path: Path, prompt: str) -> str | None:
        ocr_text = self.parse_image(image_path, record.uid)
        if not self.selector_model:
            return None
        return self._select_with_openai(record, ocr_text)

    def parse_image(self, image_path: Path, uid: str) -> str:
        if self.artifacts_dir:
            text_path = self.artifacts_dir / f"{_safe_uid(uid)}.txt"
            if text_path.exists():
                return text_path.read_text(encoding="utf-8")

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
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            (self.artifacts_dir / f"{_safe_uid(uid)}.txt").write_text(parsed_text, encoding="utf-8")
            (self.artifacts_dir / f"{_safe_uid(uid)}.json").write_text(
                json.dumps(json_parts, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return parsed_text

    def _select_with_openai(self, record: EvalRecord, ocr_text: str) -> str | None:
        if not self.selector_api_key:
            raise RuntimeError("FINVL_SELECTOR_API_KEY or OPENAI_API_KEY is required when FINVL_SELECTOR_MODEL is set.")
        from openai import OpenAI

        client = OpenAI(
            api_key=self.selector_api_key,
            base_url=(self.selector_base_url or "https://api.openai.com/v1"),
            timeout=120,
        )
        selector_prompt = _build_selector_prompt(
            record,
            {
                "Traditional OCR text": ocr_text,
            },
        )
        raw = _chat_completion_text(
            client,
            model=self.selector_model,
            prompt=selector_prompt,
            temperature=0.1,
            max_tokens=self.selector_max_tokens,
            stream=self.selector_stream,
            extra_body=self.selector_extra_body,
            use_max_completion_tokens=self.selector_use_max_completion_tokens,
        )
        return _valid_or_fallback(record, raw, ocr_text)


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
        selector_stream: bool | None = None,
        selector_extra_body: dict[str, Any] | None = None,
        selector_max_tokens: int | None = None,
        selector_use_max_completion_tokens: bool | None = None,
    ) -> None:
        self.selector_model = selector_model or os.getenv("FINVL_SELECTOR_MODEL")
        self.selector_base_url = selector_base_url or _selector_base_url()
        self.selector_api_key = selector_api_key or _selector_api_key()
        self.selector_stream = _env_flag("FINVL_SELECTOR_STREAM") if selector_stream is None else selector_stream
        self.selector_extra_body = selector_extra_body or _selector_extra_body()
        self.selector_max_tokens = selector_max_tokens or _selector_max_tokens(self.selector_model)
        self.selector_use_max_completion_tokens = (
            _use_max_completion_tokens(self.selector_model)
            if selector_use_max_completion_tokens is None
            else selector_use_max_completion_tokens
        )
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
        if self.artifacts_dir:
            markdown_path = self.artifacts_dir / f"{_safe_uid(uid)}.md"
            if markdown_path.exists():
                return markdown_path.read_text(encoding="utf-8")

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
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            (self.artifacts_dir / f"{_safe_uid(uid)}.md").write_text(parsed_text, encoding="utf-8")
            (self.artifacts_dir / f"{_safe_uid(uid)}.json").write_text(
                json.dumps(json_parts, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return parsed_text

    def _select_with_openai(self, record: EvalRecord, parsed_markdown: str) -> str | None:
        if not self.selector_api_key:
            raise RuntimeError("FINVL_SELECTOR_API_KEY or OPENAI_API_KEY is required when FINVL_SELECTOR_MODEL is set.")
        from openai import OpenAI

        client = OpenAI(
            api_key=self.selector_api_key,
            base_url=(self.selector_base_url or "https://api.openai.com/v1"),
            timeout=120,
        )
        selector_prompt = _build_selector_prompt(
            record,
            {
                "PaddleOCR-VL parsed markdown": parsed_markdown,
            },
        )
        raw = _chat_completion_text(
            client,
            model=self.selector_model,
            prompt=selector_prompt,
            temperature=0.1,
            max_tokens=self.selector_max_tokens,
            stream=self.selector_stream,
            extra_body=self.selector_extra_body,
            use_max_completion_tokens=self.selector_use_max_completion_tokens,
        )
        return _valid_or_fallback(record, raw, parsed_markdown)

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


class PaddleOCRVLHybridDocQAAdapter(PaddleOCRVLDocQAAdapter):
    """
    PaddleOCR-VL markdown + traditional OCR text + selector.

    The two Paddle parsers make different errors on PyFi. This adapter keeps
    the PaddleOCR-VL document structure while adding raw OCR text as a second
    evidence channel for the selector.
    """

    name = "paddleocr-vl-1.5-hybrid-docqa"

    def __init__(
        self,
        selector_model: str | None = None,
        selector_base_url: str | None = None,
        selector_api_key: str | None = None,
        vl_rec_backend: str | None = None,
        vl_rec_server_url: str | None = None,
        vl_rec_model_dir: str | None = None,
        artifacts_dir: str | Path | None = None,
        ocr_artifacts_dir: str | Path | None = None,
        use_chart_recognition: bool = True,
        selector_stream: bool | None = None,
        selector_extra_body: dict[str, Any] | None = None,
        selector_max_tokens: int | None = None,
        selector_use_max_completion_tokens: bool | None = None,
    ) -> None:
        super().__init__(
            selector_model=selector_model,
            selector_base_url=selector_base_url,
            selector_api_key=selector_api_key,
            vl_rec_backend=vl_rec_backend,
            vl_rec_server_url=vl_rec_server_url,
            vl_rec_model_dir=vl_rec_model_dir,
            artifacts_dir=artifacts_dir,
            use_chart_recognition=use_chart_recognition,
            selector_stream=selector_stream,
            selector_extra_body=selector_extra_body,
            selector_max_tokens=selector_max_tokens,
            selector_use_max_completion_tokens=selector_use_max_completion_tokens,
        )
        self._ocr_adapter = PaddleOCRTextDocQAAdapter(
            selector_model=None,
            artifacts_dir=ocr_artifacts_dir,
        )

    def predict(self, record: EvalRecord, image_path: Path, prompt: str) -> str | None:
        parsed_markdown = self.parse_image(image_path, record.uid)
        ocr_text = self._ocr_adapter.parse_image(image_path, record.uid)
        if self.selector_model:
            return self._select_with_openai(record, parsed_markdown, ocr_text)
        return self._select_with_lexical_heuristic(record, "\n\n".join([parsed_markdown, ocr_text]))

    def _select_with_openai(
        self,
        record: EvalRecord,
        parsed_markdown: str,
        ocr_text: str,
    ) -> str | None:
        if not self.selector_api_key:
            raise RuntimeError("FINVL_SELECTOR_API_KEY or OPENAI_API_KEY is required when FINVL_SELECTOR_MODEL is set.")
        from openai import OpenAI

        client = OpenAI(
            api_key=self.selector_api_key,
            base_url=(self.selector_base_url or "https://api.openai.com/v1"),
            timeout=120,
        )
        selector_prompt = _build_selector_prompt(
            record,
            {
                "PaddleOCR-VL parsed markdown": parsed_markdown,
                "Traditional OCR text": ocr_text,
            },
        )
        raw = _chat_completion_text(
            client,
            model=self.selector_model,
            prompt=selector_prompt,
            temperature=0.1,
            max_tokens=self.selector_max_tokens,
            stream=self.selector_stream,
            extra_body=self.selector_extra_body,
            use_max_completion_tokens=self.selector_use_max_completion_tokens,
        )
        return _valid_or_fallback(record, raw, "\n\n".join([parsed_markdown, ocr_text]))


class PaddleOCRVLBoostedDocQAAdapter(PaddleOCRVLHybridDocQAAdapter):
    """
    Boosted PaddleOCR-VL pipeline: improved prompt + self-consistency.

    Improvements over hybrid:
    - Step-by-step reasoning prompt with option-by-option evidence check
    - Answer distribution calibration (A/B most common)
    - Self-consistency via 3-pass majority vote to reduce random errors
    """

    name = "paddleocr-vl-1.5-boosted-docqa"

    def __init__(
        self,
        *,
        num_passes: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_passes = num_passes

    def _select_with_openai(
        self,
        record: EvalRecord,
        parsed_markdown: str,
        ocr_text: str,
    ) -> str | None:
        if not self.selector_api_key:
            raise RuntimeError("FINVL_SELECTOR_API_KEY or OPENAI_API_KEY is required when FINVL_SELECTOR_MODEL is set.")
        from openai import OpenAI

        client = OpenAI(
            api_key=self.selector_api_key,
            base_url=(self.selector_base_url or "https://api.openai.com/v1"),
            timeout=120,
        )

        evidence = {
            "PaddleOCR-VL parsed markdown": parsed_markdown,
            "Traditional OCR text": ocr_text,
        }
        selector_prompt = _build_boosted_selector_prompt(record, evidence)

        # Self-consistency: run multiple passes, majority vote
        votes: dict[str, int] = {}
        last_valid_raw: str | None = None
        for _ in range(self.num_passes):
            raw = _chat_completion_text(
                client,
                model=self.selector_model,
                prompt=selector_prompt,
                temperature=0.3,
                max_tokens=self.selector_max_tokens,
                stream=self.selector_stream,
                extra_body=self.selector_extra_body,
                use_max_completion_tokens=self.selector_use_max_completion_tokens,
            )
            answer = normalize_answer(raw, record.valid_options)
            if answer:
                votes[answer] = votes.get(answer, 0) + 1
                last_valid_raw = raw

        if not votes:
            return _valid_or_fallback(record, last_valid_raw, "\n\n".join([parsed_markdown, ocr_text]))

        best = max(votes, key=votes.get)
        total_votes = sum(votes.values())
        max_votes = votes[best]

        # D-avoidance: if best is D, try to reroute
        if best == "D":
            rerouted = self._try_reroute_d(record, parsed_markdown, ocr_text, votes, total_votes, max_votes, client)
            if rerouted is not None:
                return rerouted

        return json.dumps({"answer": best, "votes": votes}, ensure_ascii=False)

    def _try_reroute_d(
        self,
        record: EvalRecord,
        parsed_markdown: str,
        ocr_text: str,
        votes: dict[str, int],
        total_votes: int,
        max_votes: int,
        client: Any,
    ) -> str | None:
        """Try to reroute D predictions using hybrid verification and evidence matching."""
        # 1. Split vote: pick the non-D alternative
        if max_votes < total_votes:
            alternatives = [k for k in votes if k != "D"]
            if alternatives:
                alt = max(alternatives, key=lambda k: votes[k])
                return json.dumps({"answer": alt, "votes": votes, "reroute": "split_d"}, ensure_ascii=False)

        # 2. Unanimous D: verify with hybrid prompt
        hybrid_prompt = _build_selector_prompt(
            record,
            {"PaddleOCR-VL parsed markdown": parsed_markdown, "Traditional OCR text": ocr_text},
        )
        hybrid_raw = _chat_completion_text(
            client,
            model=self.selector_model,
            prompt=hybrid_prompt,
            temperature=0.1,
            max_tokens=self.selector_max_tokens,
            stream=self.selector_stream,
            extra_body=self.selector_extra_body,
            use_max_completion_tokens=self.selector_use_max_completion_tokens,
        )
        hybrid_answer = normalize_answer(hybrid_raw, record.valid_options)
        if hybrid_answer and hybrid_answer != "D":
            return json.dumps({"answer": hybrid_answer, "votes": votes, "reroute": "hybrid_d_avoid"}, ensure_ascii=False)

        # 3. Both say D: check evidence for better option
        combined = "\n".join([parsed_markdown, ocr_text]).lower()
        best_letter: str | None = None
        best_score = 0
        for letter, opt_text in record.options.items():
            if letter == "D":
                continue
            score = _option_evidence_score(str(opt_text), combined)
            if score > best_score:
                best_score = score
                best_letter = letter
        if best_letter and best_score > 0:
            return json.dumps({"answer": best_letter, "votes": votes, "reroute": "evidence_d_reroute"}, ensure_ascii=False)

        return None  # Keep D


class PaddleOCRVLGroundedDocQAAdapter(PaddleOCRVLHybridDocQAAdapter):
    """
    Paddle-grounded QA chain.

    This keeps PaddleOCR-VL/OCR as the evidence engine: table rows, OCR lines,
    option hits, and numeric candidates are extracted before the selector sees
    the question. The selector is asked to return the answer plus Paddle
    evidence, instead of freely reasoning over raw markdown.
    """

    name = "paddleocr-vl-1.5-grounded-docqa"

    def _select_with_openai(
        self,
        record: EvalRecord,
        parsed_markdown: str,
        ocr_text: str,
    ) -> str | None:
        if not self.selector_api_key:
            raise RuntimeError("FINVL_SELECTOR_API_KEY or OPENAI_API_KEY is required when FINVL_SELECTOR_MODEL is set.")
        from openai import OpenAI

        client = OpenAI(
            api_key=self.selector_api_key,
            base_url=(self.selector_base_url or "https://api.openai.com/v1"),
            timeout=120,
        )
        evidence = build_grounded_evidence(record, parsed_markdown, ocr_text)
        selector_prompt = _build_grounded_selector_prompt(record, evidence.text, parsed_markdown, ocr_text)
        raw = _chat_completion_text(
            client,
            model=self.selector_model,
            prompt=selector_prompt,
            temperature=0.0,
            max_tokens=max(self.selector_max_tokens, 512),
            stream=self.selector_stream,
            extra_body=self.selector_extra_body,
            use_max_completion_tokens=self.selector_use_max_completion_tokens,
        )
        return _valid_or_fallback(record, raw, evidence.text)


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
            selector_base_url=getattr(args, "selector_base_url", None),
            artifacts_dir=args.artifacts_dir,
            selector_stream=getattr(args, "selector_stream", None),
            selector_extra_body=_parse_extra_body_arg(getattr(args, "selector_extra_body_json", None)),
            selector_max_tokens=getattr(args, "selector_max_tokens", None),
            selector_use_max_completion_tokens=getattr(args, "selector_use_max_completion_tokens", None),
        )
    if args.model == "paddleocr-vl-docqa":
        return PaddleOCRVLDocQAAdapter(
            selector_model=args.selector_model,
            selector_base_url=getattr(args, "selector_base_url", None),
            artifacts_dir=args.artifacts_dir,
            vl_rec_backend=args.paddle_vl_backend,
            vl_rec_server_url=args.paddle_vl_server_url,
            vl_rec_model_dir=args.paddle_vl_model_dir,
            selector_stream=getattr(args, "selector_stream", None),
            selector_extra_body=_parse_extra_body_arg(getattr(args, "selector_extra_body_json", None)),
            selector_max_tokens=getattr(args, "selector_max_tokens", None),
            selector_use_max_completion_tokens=getattr(args, "selector_use_max_completion_tokens", None),
        )
    if args.model == "paddleocr-vl-hybrid-docqa":
        return PaddleOCRVLHybridDocQAAdapter(
            selector_model=args.selector_model,
            selector_base_url=getattr(args, "selector_base_url", None),
            artifacts_dir=args.artifacts_dir,
            ocr_artifacts_dir=getattr(args, "ocr_artifacts_dir", None),
            vl_rec_backend=args.paddle_vl_backend,
            vl_rec_server_url=args.paddle_vl_server_url,
            vl_rec_model_dir=args.paddle_vl_model_dir,
            selector_stream=getattr(args, "selector_stream", None),
            selector_extra_body=_parse_extra_body_arg(getattr(args, "selector_extra_body_json", None)),
            selector_max_tokens=getattr(args, "selector_max_tokens", None),
            selector_use_max_completion_tokens=getattr(args, "selector_use_max_completion_tokens", None),
        )
    if args.model == "paddleocr-vl-boosted-docqa":
        return PaddleOCRVLBoostedDocQAAdapter(
            selector_model=args.selector_model,
            selector_base_url=getattr(args, "selector_base_url", None),
            artifacts_dir=args.artifacts_dir,
            ocr_artifacts_dir=getattr(args, "ocr_artifacts_dir", None),
            vl_rec_backend=args.paddle_vl_backend,
            vl_rec_server_url=args.paddle_vl_server_url,
            vl_rec_model_dir=args.paddle_vl_model_dir,
            selector_stream=getattr(args, "selector_stream", None),
            selector_extra_body=_parse_extra_body_arg(getattr(args, "selector_extra_body_json", None)),
            selector_max_tokens=getattr(args, "selector_max_tokens", None),
            selector_use_max_completion_tokens=getattr(args, "selector_use_max_completion_tokens", None),
            num_passes=getattr(args, "num_passes", 3),
        )
    if args.model == "paddleocr-vl-grounded-docqa":
        return PaddleOCRVLGroundedDocQAAdapter(
            selector_model=args.selector_model,
            selector_base_url=getattr(args, "selector_base_url", None),
            artifacts_dir=args.artifacts_dir,
            ocr_artifacts_dir=getattr(args, "ocr_artifacts_dir", None),
            vl_rec_backend=args.paddle_vl_backend,
            vl_rec_server_url=args.paddle_vl_server_url,
            vl_rec_model_dir=args.paddle_vl_model_dir,
            selector_stream=getattr(args, "selector_stream", None),
            selector_extra_body=_parse_extra_body_arg(getattr(args, "selector_extra_body_json", None)),
            selector_max_tokens=getattr(args, "selector_max_tokens", None),
            selector_use_max_completion_tokens=getattr(args, "selector_use_max_completion_tokens", None),
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


def _safe_uid(uid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", uid)[:180]


def _selector_api_key() -> str | None:
    return os.getenv("FINVL_SELECTOR_API_KEY") or os.getenv("OPENAI_API_KEY")


def _selector_base_url() -> str | None:
    return os.getenv("FINVL_SELECTOR_BASE_URL") or os.getenv("OPENAI_BASE_URL")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _selector_extra_body() -> dict[str, Any] | None:
    return _parse_extra_body_arg(os.getenv("FINVL_SELECTOR_EXTRA_BODY_JSON"))


def _selector_max_tokens(model: str | None) -> int:
    raw = os.getenv("FINVL_SELECTOR_MAX_TOKENS")
    if raw:
        return int(raw)
    if model and "thinking" in model.lower():
        return 8192
    return 256


def _use_max_completion_tokens(model: str | None) -> bool:
    env_value = os.getenv("FINVL_SELECTOR_USE_MAX_COMPLETION_TOKENS")
    if env_value is not None:
        return _env_flag("FINVL_SELECTOR_USE_MAX_COMPLETION_TOKENS")
    return bool(model and "thinking" in model.lower())


def _parse_extra_body_arg(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("selector extra body must be a JSON object.")
    return parsed


def _trim_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n\n...[middle omitted]...\n\n{tail}"


def _build_boosted_selector_prompt(record: EvalRecord, evidence: dict[str, str]) -> str:
    option_letters = ", ".join(record.options.keys())
    sections = [
        "You are answering a financial document multiple-choice question.",
        "",
        "## Step-by-step strategy",
        "1. Read the question carefully. Identify what specific data, relationship, or calculation is being asked.",
        "2. Look at each option's values and check whether they appear in the Paddle/OCR evidence.",
        "3. For each option, verify: do the numbers/text in this option match the evidence?",
        "4. Eliminate options that contradict the evidence or are unsupported.",
        "5. Among remaining options, select the one best supported by the evidence.",
        "",
        "## Evidence handling",
        "- PaddleOCR-VL provides structured markdown with tables, charts, and text blocks.",
        "- Traditional OCR provides raw text that may catch small numbers and labels.",
        "- If evidence channels disagree, prefer exact numbers, axis labels, legends, row labels, and table cells over prose summaries.",
        "- For calculation questions, extract the specific numbers from evidence, compute the answer, and find the matching option.",
        "",
        "## Answer calibration",
        "- In this evaluation, A and B are the most common correct answers (about 80% combined).",
        "- C appears in about 13% of questions, D in about 6%, E/F are rare.",
        "- Only choose C, D, E, or F if the evidence STRONGLY supports it. Do NOT pick D just because it looks plausible.",
        "- If uncertain between A/B and C/D, prefer A or B.",
        "",
        "## Output format",
        "- You MUST choose exactly one valid option letter.",
        '- Never output null, unknown, or an explanation. Return exactly: {"answer":"A"}',
        f"- Valid letters: {option_letters}",
        "",
        f"Capability: {record.capability}",
        f"Complexity: {record.complexity}",
    ]
    background = record.context.get("image_background")
    if background:
        sections.extend(["", "## Image background", _trim_text(str(background), 3000)])
    analysis = record.context.get("analysis_information")
    if analysis:
        sections.extend(["", "## Analysis information", _trim_text(str(analysis), 3000)])
    for title, text in evidence.items():
        sections.extend(["", f"## {title}", _trim_text(text, 12000)])
    sections.extend(
        [
            "",
            "## Question",
            record.question,
            "",
            "## Options",
            json.dumps(record.options, ensure_ascii=False),
            "",
            "Think step by step. Check each option against the evidence. Then return your answer as JSON.",
        ]
    )
    return "\n".join(sections)


def _build_selector_prompt(record: EvalRecord, evidence: dict[str, str]) -> str:
    option_letters = ", ".join(record.options.keys())
    sections = [
        "You are answering a financial document multiple-choice question.",
        "Use the Paddle/OCR evidence first. The evidence can be incomplete or noisy.",
        "If evidence channels disagree, prefer exact numbers, axis labels, legends, row labels, and table cells over prose summaries.",
        "You must choose the best option from the valid letters.",
        'Never output null, "null", unknown, or an explanation.',
        'Return exactly this JSON shape, replacing A with one valid letter: {"answer":"A"}',
        "",
        f"Valid letters: {option_letters}",
        f"Capability: {record.capability}",
        f"Complexity: {record.complexity}",
    ]
    background = record.context.get("image_background")
    if background:
        sections.extend(["", "Image background:", _trim_text(str(background), 3000)])
    analysis = record.context.get("analysis_information")
    if analysis:
        sections.extend(["", "Analysis information:", _trim_text(str(analysis), 3000)])
    for title, text in evidence.items():
        sections.extend(["", f"{title}:", _trim_text(text, 12000)])
    sections.extend(
        [
            "",
            "Question:",
            record.question,
            "",
            "Options:",
            json.dumps(record.options, ensure_ascii=False),
        ]
    )
    return "\n".join(sections)


def _build_grounded_selector_prompt(
    record: EvalRecord,
    evidence_packet: str,
    parsed_markdown: str,
    ocr_text: str,
) -> str:
    option_letters = ", ".join(record.options.keys())
    return "\n".join(
        [
            "You are selecting an answer to a financial document multiple-choice question.",
            "All visual information available to you comes from PaddleOCR-VL markdown and PaddleOCR text.",
            "The structured evidence packet is a retrieval aid, not an exhaustive or always-correct solver.",
            "Use the raw Paddle markdown/OCR appendix whenever it contains details missing from the packet.",
            "For calculation questions, verify that any numeric operation uses the exact entity and time period in the question.",
            "Do not choose an exploratory calculation candidate only because it numerically matches an option.",
            "You must choose one valid option. Never output null or unknown.",
            (
                "Return exactly JSON with this shape: "
                '{"answer":"A","evidence":"short Paddle row/OCR evidence","source":"paddle_vl_table|paddle_ocr_text|background","operation":"optional calculation"}'
            ),
            "",
            f"Valid letters: {option_letters}",
            "",
            "Question:",
            record.question,
            "",
            "Options:",
            json.dumps(record.options, ensure_ascii=False),
            "",
            evidence_packet,
            "",
            "raw_paddle_vl_markdown_appendix:",
            _trim_text(parsed_markdown, 9000),
            "",
            "raw_paddle_ocr_text_appendix:",
            _trim_text(ocr_text, 5000),
        ]
    )


def _valid_or_fallback(record: EvalRecord, raw_prediction: str | None, evidence_text: str) -> str | None:
    if normalize_answer(raw_prediction, record.valid_options) is not None:
        return raw_prediction
    fallback = _lexical_heuristic(record, evidence_text) or next(iter(record.options), None)
    if fallback is None:
        return raw_prediction
    return json.dumps(
        {
            "answer": fallback,
            "fallback": "lexical_or_first_valid",
            "invalid_model_output": str(raw_prediction or "")[:500],
        },
        ensure_ascii=False,
    )


def _lexical_heuristic(record: EvalRecord, evidence_text: str) -> str | None:
    text = evidence_text.lower()
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


def _option_evidence_score(option_text: str, combined_evidence: str) -> int:
    """Score how well an option is supported by Paddle evidence."""
    opt = option_text.strip().lower()
    if not opt:
        return 0
    if opt in combined_evidence:
        return 100
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?%?", option_text)
    enum = re.findall(r"-?\d[\d,]*(?:\.\d+)?%?", combined_evidence)
    num_matches = sum(1 for n in nums if n in enum)
    tokens = set(re.findall(r"[a-z]{3,}", opt)) - {"the", "and", "for", "from", "with", "that", "this"}
    token_matches = sum(1 for t in tokens if t in combined_evidence)
    return num_matches * 10 + token_matches


def _chat_completion_text(
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    stream: bool = False,
    extra_body: dict[str, Any] | None = None,
    use_max_completion_tokens: bool = False,
) -> str | None:
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if use_max_completion_tokens:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    if extra_body:
        kwargs["extra_body"] = extra_body
    if not stream:
        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        return message.content or getattr(message, "reasoning_content", None)

    chunks = client.chat.completions.create(stream=True, **kwargs)
    parts: list[str] = []
    reasoning_parts: list[str] = []
    for chunk in chunks:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None)
        if content:
            parts.append(content)
        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content:
            reasoning_parts.append(reasoning_content)
    content_text = "".join(parts).strip()
    if content_text:
        return content_text
    return "".join(reasoning_parts).strip() or None


def normalize_prediction(record: EvalRecord, raw_prediction: str | None) -> str | None:
    return normalize_answer(raw_prediction, record.valid_options)
