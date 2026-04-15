from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from finvl_eval.prompts import build_mcq_prompt
from finvl_eval.pyfi import extract_gold_answer, iter_pyfi_csv, write_jsonl
from finvl_eval.records import EvalRecord
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


if __name__ == "__main__":
    unittest.main()
