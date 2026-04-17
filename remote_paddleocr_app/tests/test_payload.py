from __future__ import annotations

from pathlib import Path

from remote_paddleocr_app.paddleocr_client import file_type_to_api_value, infer_file_type
from remote_paddleocr_app.pipeline import build_document_qa_prompt


def test_file_type_values() -> None:
    assert file_type_to_api_value("pdf") == 0
    assert file_type_to_api_value("image") == 1


def test_infer_file_type() -> None:
    assert infer_file_type(Path("report.pdf")) == "pdf"
    assert infer_file_type(Path("chart.jpg")) == "image"
    assert infer_file_type(Path("chart.png")) == "image"


def test_build_document_qa_prompt_contains_choices() -> None:
    prompt = build_document_qa_prompt(
        "# Table\nvalue",
        "Which option is correct?",
        {"B": "Second", "A": "First"},
    )
    assert "PaddleOCR Markdown" in prompt
    assert "A. First" in prompt
    assert "B. Second" in prompt
