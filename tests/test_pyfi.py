from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from finvl_eval.models import (
    PaddleOCRVLHybridDocQAAdapter,
    PaddleOCRVLDocQAAdapter,
    REMOTE_PADDLEOCR_VL_DEFAULTS,
    RemotePaddleOCRVLErnieAdapter,
    _paddleocr_file_type,
    route_table_spotting_prompt_label,
)
from finvl_eval.prompts import build_mcq_prompt
from finvl_eval.pyfi import extract_gold_answer, iter_pyfi_csv, write_jsonl
from finvl_eval.records import EvalRecord
from finvl_eval.runner import run
from finvl_eval.scoring import aggregate, normalize_answer


class PyFiParsingTests(unittest.TestCase):
    def test_extract_gold_answer_selects_best_action(self) -> None:
        actions = [
            {"answer": "A", "visit_count": 1, "victory_count": 0},
            {"answer": "C", "visit_count": 2, "victory_count": 1},
        ]
        self.assertEqual(extract_gold_answer(actions), "C")
        self.assertEqual(extract_gold_answer(str(actions)), "C")

    def test_iter_pyfi_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "pyfi.csv"
            csv_path.write_text(
                "question_node_no,options,complexity,visit_count,fq_no,capability,"
                "victory_count,image_background,parent_node_no,actions,question,image_path\n"
                "1,\"{'A': 'Light green', 'B': 'Dark blue'}\",1,3,1,Perception,0,"
                "\"A chart background\",0,\"[{'answer_no': 0, 'answer': 'A', 'visit_count': 1, 'victory_count': 0}]\","
                "\"Which color?\",./images/000001/000010.jpg\n",
                encoding="utf-8",
            )
            records = list(iter_pyfi_csv(csv_path))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].answer, "A")
        self.assertEqual(records[0].options["B"], "Dark blue")


class PromptAndScoringTests(unittest.TestCase):
    def test_prompt_and_scoring(self) -> None:
        record = EvalRecord(
            uid="1",
            image_path="./images/a.jpg",
            question="Pick one.",
            options={"A": "Alpha", "B": "Beta"},
            answer="B",
            capability="Perception",
            complexity="1",
            context={"image_background": "Background"},
        )
        prompt = build_mcq_prompt(record)
        self.assertIn("Background", prompt)
        self.assertEqual(normalize_answer("The answer is B.", record.valid_options), "B")
        self.assertEqual(normalize_answer('{"answer":"B"}', record.valid_options), "B")
        self.assertEqual(normalize_answer('```json\n{"answer":"B"}\n```', record.valid_options), "B")
        self.assertIsNone(
            normalize_answer(
                "A long analysis without a final answer, continuing with more reasoning text.",
                record.valid_options,
            )
        )
        metrics = aggregate(
            [
                {"correct": True, "prediction": "B", "capability": "Perception", "complexity": "1"},
                {"correct": False, "prediction": None, "capability": "Perception", "complexity": "1"},
            ]
        )
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["invalid_rate"], 0.5)

    def test_write_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sample.jsonl"
            count = write_jsonl(
                [
                    EvalRecord(
                        uid="x",
                        image_path="./images/x.jpg",
                        question="Q",
                        options={"A": "a"},
                        answer="A",
                    )
                ],
                out,
            )
            self.assertEqual(count, 1)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["answer"], "A")


class PaddleOCRRemoteTests(unittest.TestCase):
    def test_paddleocr_file_type(self) -> None:
        self.assertEqual(_paddleocr_file_type(Path("sample.pdf")), 0)
        self.assertEqual(_paddleocr_file_type(Path("sample.png")), 1)

    def test_remote_paddleocr_disables_local_ocr_by_default(self) -> None:
        adapter = PaddleOCRVLHybridDocQAAdapter(
            selector_model="ernie-4.5-turbo-128k-preview",
            api_url="https://example.com/layout-parsing",
            api_token="token",
        )
        self.assertFalse(adapter.use_local_ocr_evidence)
        self.assertIsNone(adapter._ocr_adapter)

    def test_local_mode_keeps_local_ocr_by_default(self) -> None:
        adapter = PaddleOCRVLHybridDocQAAdapter(
            selector_model="ernie-4.5-turbo-128k-preview",
        )
        self.assertTrue(adapter.use_local_ocr_evidence)

    def test_remote_adapter_never_uses_local_ocr(self) -> None:
        adapter = RemotePaddleOCRVLErnieAdapter(
            selector_model="ernie-4.5-turbo-128k-preview",
            api_url="https://example.com/layout-parsing",
            api_token="token",
        )
        self.assertFalse(hasattr(adapter, "_ocr_adapter"))

    def test_remote_paddleocr_uses_table_focused_payload_defaults(self) -> None:
        adapter = PaddleOCRVLDocQAAdapter(
            selector_model="ernie-4.5-turbo-128k-preview",
            api_url="https://example.com/layout-parsing",
            api_token="token",
        )
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {
                "layoutParsingResults": [
                    {
                        "markdown": {
                            "text": "| metric | value |\n| --- | --- |\n| revenue | 10 |"
                        }
                    }
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            image_path.write_bytes(b"fake-image")
            with patch("requests.post", return_value=mock_response) as mock_post:
                parsed_text, _ = adapter._parse_image_remote(image_path)

        self.assertIn("revenue", parsed_text)
        sent_payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(sent_payload["promptLabel"], "table")
        self.assertEqual(sent_payload["markdownIgnoreLabels"], REMOTE_PADDLEOCR_VL_DEFAULTS["markdownIgnoreLabels"])
        self.assertTrue(sent_payload["mergeTables"])
        self.assertTrue(sent_payload["restructurePages"])
        self.assertFalse(sent_payload["useLayoutDetection"])
        self.assertTrue(sent_payload["useChartRecognition"])

    def test_remote_paddleocr_prompt_label_can_be_overridden(self) -> None:
        adapter = PaddleOCRVLDocQAAdapter(
            selector_model="ernie-4.5-turbo-128k-preview",
            api_url="https://example.com/layout-parsing",
            api_token="token",
            remote_prompt_label="seal",
        )
        mock_response = Mock()
        mock_response.json.return_value = {
            "result": {"layoutParsingResults": [{"markdown": {"text": "seal text"}}]}
        }
        mock_response.raise_for_status.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.png"
            image_path.write_bytes(b"fake-image")
            with patch("requests.post", return_value=mock_response) as mock_post:
                adapter._parse_image_remote(image_path)

        sent_payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(sent_payload["promptLabel"], "seal")


class PromptRoutingTests(unittest.TestCase):
    def test_table_spotting_router_prefers_table_for_numeric_computation(self) -> None:
        record = EvalRecord(
            uid="calc",
            image_path="./images/calc.jpg",
            question="What is the percentage decrease in revenue from 2019 to 2021?",
            options={"A": "10%", "B": "25%", "C": "40%", "D": "55%"},
            answer="B",
        )
        self.assertEqual(route_table_spotting_prompt_label(record), "table")

    def test_table_spotting_router_prefers_spotting_for_visual_trend_questions(self) -> None:
        record = EvalRecord(
            uid="trend",
            image_path="./images/trend.jpg",
            question="Which color corresponds to the diversified portfolio and what trend is observed?",
            options={"A": "Green", "B": "Blue", "C": "Cyan", "D": "Dark Blue"},
            answer="C",
        )
        self.assertEqual(route_table_spotting_prompt_label(record), "spotting")

    def test_table_spotting_router_uses_image_background_signals(self) -> None:
        record = EvalRecord(
            uid="diagram",
            image_path="./images/diagram.jpg",
            question="After consent is provided, what is the next step?",
            options={"A": "Bank selection", "B": "Authentication", "C": "Risk score", "D": "Revocation"},
            answer="B",
            context={"image_background": "The figure is a consent management process flow diagram."},
        )
        self.assertEqual(route_table_spotting_prompt_label(record), "spotting")


class RunnerResumeTests(unittest.TestCase):
    def test_runner_resumes_from_existing_output(self) -> None:
        records = [
            EvalRecord(uid="1", image_path="./images/1.jpg", question="Q1", options={"A": "a"}, answer="A"),
            EvalRecord(uid="2", image_path="./images/2.jpg", question="Q2", options={"A": "a"}, answer="A"),
        ]
        adapter = Mock()
        adapter.predict.return_value = "A"

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "resume.jsonl"
            out_path.write_text(
                json.dumps(
                    {
                        "uid": "1",
                        "image_path": "data\\pyfi\\images\\1.jpg",
                        "capability": None,
                        "complexity": None,
                        "answer": "A",
                        "prediction": "A",
                        "raw_prediction": "A",
                        "correct": True,
                        "error": None,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                model="remote-paddleocr-vl-ernie-docqa",
                dataset="unused.jsonl",
                format="jsonl",
                out=str(out_path),
                limit=None,
                require_image=False,
                images_root="data/pyfi",
                context_mode="image_background_only",
                progress_every=1000,
            )
            with patch("finvl_eval.runner.build_adapter", return_value=adapter), patch(
                "finvl_eval.runner.iter_records", return_value=iter(records)
            ):
                metrics = run(args)

        self.assertEqual(adapter.predict.call_count, 1)
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["correct"], 2)


if __name__ == "__main__":
    unittest.main()
