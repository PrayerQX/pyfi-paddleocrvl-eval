from __future__ import annotations

from pathlib import Path

from remote_paddleocr_app.paddleocr_client import file_type_to_api_value, infer_file_type
from remote_paddleocr_app.eval_pyfi import (
    PyFiRecord,
    build_choice_prompt,
    normalize_answer,
    occurrence_keys,
)
from remote_paddleocr_app.pipeline import build_document_qa_prompt
from remote_paddleocr_app.stack_results import parse_preference, parse_run


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


def test_normalize_answer_from_json() -> None:
    assert normalize_answer('{"answer":"B","reason":"x"}', {"A", "B"}) == "B"


def test_normalize_answer_from_text() -> None:
    assert normalize_answer("Answer: C", {"A", "B", "C"}) == "C"


def test_occurrence_keys_preserve_duplicate_uids() -> None:
    records = [
        PyFiRecord("x", "a.jpg", "q", {"A": "a"}, "A", None, None, {}),
        PyFiRecord("x", "a.jpg", "q", {"A": "a"}, "A", None, None, {}),
        PyFiRecord("y", "b.jpg", "q", {"A": "a"}, "A", None, None, {}),
    ]
    assert occurrence_keys(records) == [("x", 0), ("x", 1), ("y", 0)]


def test_build_choice_prompt_uses_perception_guidance() -> None:
    record = PyFiRecord(
        uid="x",
        image_path="a.jpg",
        question="Which country is represented by the green dashed line?",
        options={"A": "A", "B": "B"},
        answer="A",
        capability="Perception",
        complexity="1",
        context={"image_background": "A chart with colored lines."},
    )
    prompt = build_choice_prompt(record, "table")
    assert "visual perception" in prompt
    assert "line style" in prompt


def test_stack_arg_parsers() -> None:
    run = parse_run("round2=out.jsonl")
    assert run.name == "round2"
    assert str(run.path) == "out.jsonl"
    assert parse_preference("Perception=round3") == ("Perception", "round3")
