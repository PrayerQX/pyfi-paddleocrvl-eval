"""Tests for the one-vote veto module."""
import unittest

from finvl_eval.records import EvalRecord
from finvl_eval.veto import (
    VetoConfig,
    ConfidenceScore,
    VetoResult,
    check_veto,
    compute_confidence,
    score_evidence_contradiction,
    score_evidence_support,
    score_model_certainty,
    score_vote_confidence,
)


def _record(
    options: dict[str, str] | None = None,
    answer: str = "A",
) -> EvalRecord:
    return EvalRecord(
        uid="test-1",
        image_path="./images/test.jpg",
        question="What is the revenue growth?",
        options=options or {"A": "5.3%", "B": "3.2%", "C": "7.1%", "D": "2.8%"},
        answer=answer,
        capability="Data_extraction",
        complexity="1",
    )


EVIDENCE_WITH_A = "Revenue growth was 5.3% in Q4, driven by strong sales."
EVIDENCE_WITH_B = "The growth rate reached 3.2% according to the chart."
EVIDENCE_WITH_NONE = "The chart shows various colors and legend items."


class VoteConfidenceTests(unittest.TestCase):
    def test_unanimous(self) -> None:
        self.assertAlmostEqual(score_vote_confidence({"A": 3}), 1.0)

    def test_split_vote(self) -> None:
        self.assertAlmostEqual(score_vote_confidence({"A": 2, "B": 1}), 2 / 3)

    def test_even_split(self) -> None:
        self.assertAlmostEqual(score_vote_confidence({"A": 1, "B": 1}), 0.5)

    def test_empty_votes(self) -> None:
        # No voting context (single pass) -> full confidence
        self.assertAlmostEqual(score_vote_confidence(None), 1.0)
        self.assertAlmostEqual(score_vote_confidence({}), 1.0)

    def test_three_way_split(self) -> None:
        self.assertAlmostEqual(score_vote_confidence({"A": 1, "B": 1, "C": 1}), 1 / 3)


class EvidenceSupportTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        rec = _record()
        score = score_evidence_support("A", rec, EVIDENCE_WITH_A)
        self.assertEqual(score, 1.0)

    def test_no_match(self) -> None:
        rec = _record()
        score = score_evidence_support("A", rec, EVIDENCE_WITH_NONE)
        self.assertEqual(score, 0.0)

    def test_numeric_match(self) -> None:
        rec = _record()
        score = score_evidence_support("B", rec, EVIDENCE_WITH_B)
        self.assertGreater(score, 0.5)

    def test_none_prediction(self) -> None:
        rec = _record()
        score = score_evidence_support(None, rec, EVIDENCE_WITH_A)
        self.assertEqual(score, 0.0)

    def test_invalid_prediction(self) -> None:
        rec = _record()
        score = score_evidence_support("Z", rec, EVIDENCE_WITH_A)
        self.assertEqual(score, 0.0)


class EvidenceContradictionTests(unittest.TestCase):
    def test_strong_alternative(self) -> None:
        rec = _record()
        # Predict A, but evidence clearly supports B
        score = score_evidence_contradiction("D", rec, EVIDENCE_WITH_A)
        self.assertGreater(score, 0.5)  # A is well-supported as alternative

    def test_no_alternative(self) -> None:
        rec = _record()
        score = score_evidence_contradiction("A", rec, EVIDENCE_WITH_A)
        # B, C, D are not in evidence, so contradiction should be low
        self.assertLess(score, 0.5)

    def test_none_prediction(self) -> None:
        rec = _record()
        score = score_evidence_contradiction(None, rec, EVIDENCE_WITH_A)
        self.assertEqual(score, 0.0)


class ModelCertaintyTests(unittest.TestCase):
    def test_clean_json(self) -> None:
        self.assertAlmostEqual(score_model_certainty('{"answer":"B"}'), 0.9)

    def test_explicit_pattern(self) -> None:
        self.assertAlmostEqual(score_model_certainty("The ANSWER IS B"), 0.5)

    def test_null(self) -> None:
        self.assertAlmostEqual(score_model_certainty(None), 0.0)

    def test_empty(self) -> None:
        self.assertAlmostEqual(score_model_certainty(""), 0.0)

    def test_plain_text(self) -> None:
        # Plain text without clear pattern -> default 0.3
        self.assertAlmostEqual(score_model_certainty("Some reasoning text"), 0.3)


class ComputeConfidenceTests(unittest.TestCase):
    def test_high_confidence(self) -> None:
        rec = _record()
        conf = compute_confidence(
            predicted_option="A",
            record=rec,
            evidence_text=EVIDENCE_WITH_A,
            raw_prediction='{"answer":"A"}',
            votes={"A": 3},
        )
        self.assertGreater(conf.composite, 0.7)
        self.assertEqual(conf.predicted_option, "A")

    def test_low_confidence(self) -> None:
        rec = _record()
        conf = compute_confidence(
            predicted_option="D",
            record=rec,
            evidence_text=EVIDENCE_WITH_A,  # supports A, not D
            raw_prediction=None,
            votes={"D": 1, "A": 2},
        )
        self.assertLess(conf.composite, 0.5)

    def test_factors_populated(self) -> None:
        rec = _record()
        conf = compute_confidence("A", rec, EVIDENCE_WITH_A, '{"answer":"A"}')
        self.assertIn("vote_weight", conf.factors)
        self.assertIn("evidence_weight", conf.factors)

    def test_custom_weights(self) -> None:
        rec = _record()
        cfg = VetoConfig(evidence_weight=1.0, vote_weight=0.0, contradiction_weight=0.0, certainty_weight=0.0)
        conf = compute_confidence("A", rec, EVIDENCE_WITH_A, '{"answer":"A"}', config=cfg)
        # With evidence_weight=1.0 and exact match, composite should be ~1.0
        self.assertAlmostEqual(conf.composite, 1.0, places=1)


class CheckVetoTests(unittest.TestCase):
    def test_veto_below_threshold(self) -> None:
        rec = _record()
        cfg = VetoConfig(enabled=True, threshold=0.6)
        result = check_veto(
            prediction="D",
            record=rec,
            evidence_text=EVIDENCE_WITH_A,  # supports A, not D
            raw_prediction=None,
            config=cfg,
        )
        self.assertTrue(result.vetoed)
        self.assertEqual(result.original_prediction, "D")
        # Fallback should be A (well-supported in evidence)
        self.assertIsNotNone(result.fallback_prediction)

    def test_no_veto_above_threshold(self) -> None:
        rec = _record()
        cfg = VetoConfig(enabled=True, threshold=0.3)
        result = check_veto(
            prediction="A",
            record=rec,
            evidence_text=EVIDENCE_WITH_A,
            raw_prediction='{"answer":"A"}',
            votes={"A": 3},
            config=cfg,
        )
        self.assertFalse(result.vetoed)

    def test_veto_disabled(self) -> None:
        rec = _record()
        cfg = VetoConfig(enabled=False)
        result = check_veto(
            prediction="D",
            record=rec,
            evidence_text=EVIDENCE_WITH_A,
            raw_prediction=None,
            config=cfg,
        )
        self.assertFalse(result.vetoed)

    def test_veto_none_prediction(self) -> None:
        rec = _record()
        cfg = VetoConfig(enabled=True, threshold=0.9)
        result = check_veto(
            prediction=None,
            record=rec,
            evidence_text=EVIDENCE_WITH_A,
            raw_prediction=None,
            config=cfg,
        )
        self.assertFalse(result.vetoed)

    def test_veto_uses_evidence_fallback(self) -> None:
        rec = _record()
        cfg = VetoConfig(enabled=True, threshold=0.8)
        result = check_veto(
            prediction="C",
            record=rec,
            evidence_text=EVIDENCE_WITH_A,
            raw_prediction="maybe C",
            config=cfg,
        )
        self.assertTrue(result.vetoed)
        # Fallback should be A (best evidence match)
        self.assertEqual(result.fallback_prediction, "A")

    def test_veto_result_to_dict(self) -> None:
        rec = _record()
        cfg = VetoConfig(enabled=True, threshold=0.9)
        result = check_veto(
            prediction="D",
            record=rec,
            evidence_text=EVIDENCE_WITH_A,
            raw_prediction=None,
            config=cfg,
        )
        d = result.to_dict()
        self.assertIn("vetoed", d)
        self.assertIn("confidence", d)
        self.assertIn("original_prediction", d)

    def test_veto_no_evidence_check(self) -> None:
        rec = _record()
        cfg = VetoConfig(enabled=True, threshold=0.2, enable_evidence_check=False)
        result = check_veto(
            prediction="A",
            record=rec,
            evidence_text=EVIDENCE_WITH_A,
            raw_prediction='{"answer":"A"}',
            votes={"A": 3},
            config=cfg,
        )
        # Without evidence check, confidence relies on vote and certainty
        # vote=1.0, certainty=0.9 -> composite should be > 0.2
        self.assertFalse(result.vetoed)

    def test_veto_with_votes(self) -> None:
        rec = _record()
        cfg = VetoConfig(enabled=True, threshold=0.5)
        # Split votes + wrong evidence
        result = check_veto(
            prediction="D",
            record=rec,
            evidence_text=EVIDENCE_WITH_A,
            raw_prediction="D",
            votes={"D": 1, "A": 2},
            config=cfg,
        )
        self.assertTrue(result.vetoed)
        self.assertEqual(result.fallback_prediction, "A")


if __name__ == "__main__":
    unittest.main()
