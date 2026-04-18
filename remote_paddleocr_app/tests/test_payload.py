from __future__ import annotations

from pathlib import Path

from remote_paddleocr_app.paddleocr_client import file_type_to_api_value, infer_file_type
from remote_paddleocr_app.eval_pyfi import (
    PyFiRecord,
    build_choice_prompt,
    build_structured_choice_prompt,
    collect_layout_blocks,
    collect_structured_intermediate,
    normalize_answer,
    occurrence_keys,
    selector_cache_key,
)
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


def test_normalize_answer_from_json() -> None:
    assert normalize_answer('{"answer":"B","reason":"x"}', {"A", "B"}) == "B"


def test_normalize_answer_from_text() -> None:
    assert normalize_answer("Answer: C", {"A", "B", "C"}) == "C"


def test_build_choice_prompt_omits_task_metadata() -> None:
    record = PyFiRecord(
        uid="x",
        image_path="a.jpg",
        question="Which option has the largest increase?",
        options={"A": "First", "B": "Second"},
        answer="A",
        capability="Calculation_analysis",
        complexity="5",
        context={"image_background": "A line chart."},
    )
    prompt = build_choice_prompt(record, "table evidence")
    assert "remote PaddleOCR Markdown" in prompt
    assert "Capability:" not in prompt
    assert "Complexity:" not in prompt


def test_collect_layout_blocks_includes_text_and_bbox() -> None:
    result = {
        "layoutParsingResults": [
            {
                "prunedResult": {
                    "parsing_res_list": [
                        {
                            "block_label": "text",
                            "block_content": "Revenue 2024",
                            "block_bbox": [10, 20, 80, 40],
                            "block_order": 2,
                        },
                        {
                            "block_label": "image",
                            "block_content": "",
                            "block_bbox": [0, 0, 100, 100],
                            "block_order": None,
                        },
                    ]
                }
            }
        ]
    }
    blocks = collect_layout_blocks(result)
    assert "text bbox=[10, 20, 80, 40]: Revenue 2024" in blocks
    assert "image bbox=[0, 0, 100, 100]" in blocks


def test_collect_structured_intermediate_uses_remote_layout_result() -> None:
    record = PyFiRecord(
        uid="x",
        image_path="a.jpg",
        question="What is the difference in revenue between 2020 and 2021?",
        options={"A": "10", "B": "20"},
        answer="A",
        capability="Calculation_analysis",
        complexity="2",
        context={"image_background": "A line chart."},
    )
    result = {
        "layoutParsingResults": [
            {
                "prunedResult": {
                    "parsing_res_list": [
                        {
                            "block_label": "text",
                            "block_content": "2020 revenue 50; 2021 revenue 60",
                            "block_bbox": [10, 20, 80, 40],
                            "block_order": 1,
                        }
                    ]
                }
            }
        ]
    }
    evidence = collect_structured_intermediate(record, result, "2020 revenue 50\n2021 revenue 60")
    assert "option_evidence" in evidence
    assert "numeric_candidates" in evidence
    assert "2020 revenue 50" in evidence


def test_build_structured_choice_prompt_omits_task_metadata() -> None:
    record = PyFiRecord(
        uid="x",
        image_path="a.jpg",
        question="Which option is correct?",
        options={"A": "First", "B": "Second"},
        answer="A",
        capability="Perception",
        complexity="1",
        context={"image_background": "A chart."},
    )
    prompt = build_structured_choice_prompt(record, '{"evidence": "x"}')
    assert "remote PaddleOCR-VL" in prompt
    assert "Capability:" not in prompt
    assert "Complexity:" not in prompt


def test_selector_cache_key_ignores_credentials() -> None:
    class Args:
        ernie_model = "ernie-4.5"
        max_completion_tokens = 256
        temperature = None
        disable_web_search = True

    first = selector_cache_key(Args(), "prompt")
    second = selector_cache_key(Args(), "prompt")
    assert first == second
    assert selector_cache_key(Args(), "different prompt") != first


def test_occurrence_keys_preserve_duplicate_uids() -> None:
    records = [
        PyFiRecord("x", "a.jpg", "q", {"A": "a"}, "A", None, None, {}),
        PyFiRecord("x", "a.jpg", "q", {"A": "a"}, "A", None, None, {}),
        PyFiRecord("y", "b.jpg", "q", {"A": "a"}, "A", None, None, {}),
    ]
    assert occurrence_keys(records) == [("x", 0), ("x", 1), ("y", 0)]
