"""One-vote veto mechanism: confidence scoring and evidence-based rejection."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .records import EvalRecord


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VetoConfig:
    """Configuration for the one-vote veto mechanism."""

    enabled: bool = False
    threshold: float = 0.4
    vote_weight: float = 0.30
    evidence_weight: float = 0.40
    contradiction_weight: float = 0.20
    certainty_weight: float = 0.10
    enable_evidence_check: bool = True
    enable_contradiction_check: bool = True
    safe_mode: bool = True
    min_fallback_support: float = 0.9
    min_support_margin: float = 0.45
    max_original_support: float = 0.25
    max_veto_vote_confidence: float = 2 / 3


@dataclass(slots=True)
class ConfidenceScore:
    """Multi-factor confidence assessment for a single prediction."""

    predicted_option: str
    vote_confidence: float
    evidence_support: float
    evidence_contradiction: float
    model_certainty: float
    composite: float
    factors: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_option": self.predicted_option,
            "vote_confidence": self.vote_confidence,
            "evidence_support": self.evidence_support,
            "evidence_contradiction": self.evidence_contradiction,
            "model_certainty": self.model_certainty,
            "composite": self.composite,
            "factors": self.factors,
        }


@dataclass(slots=True)
class VetoResult:
    """Outcome of a veto check."""

    vetoed: bool
    original_prediction: str | None
    fallback_prediction: str | None
    confidence: ConfidenceScore
    veto_reason: str | None = None
    fallback_support: float = 0.0
    support_margin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "vetoed": self.vetoed,
            "original_prediction": self.original_prediction,
            "fallback_prediction": self.fallback_prediction,
            "confidence": self.confidence.to_dict(),
            "veto_reason": self.veto_reason,
            "fallback_support": self.fallback_support,
            "support_margin": self.support_margin,
        }


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def score_vote_confidence(votes: dict[str, int] | None) -> float:
    """Compute vote confidence from a self-consistency voting result.

    Formula: ``max_votes / total_votes``.
    Returns 1.0 if there is only one pass (no voting context).
    Returns 0.0 if *votes* is ``None`` or empty.
    """
    if not votes:
        return 1.0  # single-pass: no voting context, assume full confidence
    total = sum(votes.values())
    if total == 0:
        return 0.0
    return max(votes.values()) / total


def score_evidence_support(
    predicted_option: str | None,
    record: EvalRecord,
    evidence_text: str,
) -> float:
    """Score how well the predicted option is supported by PaddleOCR evidence.

    Returns a value in ``[0.0, 1.0]``:
    - 1.0: option value appears verbatim in evidence
    - 0.5-0.9: numeric values from option found in evidence
    - 0.1-0.4: keyword overlap only
    - 0.0: no support found
    """
    if not predicted_option or predicted_option not in record.options:
        return 0.0

    option_text = str(record.options[predicted_option]).strip()
    if not option_text:
        return 0.0

    evidence_lower = evidence_text.lower()
    opt_lower = option_text.lower()

    # Exact text match
    if opt_lower in evidence_lower:
        return 1.0

    # Numeric match: option numbers found in evidence
    opt_numbers = _number_strings(option_text)
    ev_numbers = _number_strings(evidence_text)
    if opt_numbers:
        num_matches = sum(1 for n in opt_numbers if n in ev_numbers)
        ratio = num_matches / len(opt_numbers)
        if ratio == 1.0:
            return 0.9
        if ratio > 0:
            return 0.5 + 0.4 * ratio

    # Keyword overlap
    opt_tokens = _keywords(option_text)
    if opt_tokens:
        ev_tokens = _keywords(evidence_text)
        overlap = len(opt_tokens & ev_tokens)
        if overlap > 0:
            return 0.1 + 0.3 * min(overlap / len(opt_tokens), 1.0)

    return 0.0


def score_evidence_contradiction(
    predicted_option: str | None,
    record: EvalRecord,
    evidence_text: str,
) -> float:
    """Detect if another option is BETTER supported than the predicted one.

    Returns the maximum evidence support score among all alternatives.
    A high value means a strong competing option exists.
    """
    if not predicted_option:
        return 0.0

    max_alt = 0.0
    for letter in record.options:
        if letter == predicted_option:
            continue
        support = score_evidence_support(letter, record, evidence_text)
        if support > max_alt:
            max_alt = support

    return max_alt


def score_model_certainty(raw_prediction: str | None) -> float:
    """Heuristic certainty from the raw model output format.

    - Clean JSON with answer key: 0.9
    - Free text with explicit answer pattern: 0.5
    - None / empty / unparseable: 0.0
    """
    if raw_prediction is None:
        return 0.0

    text = str(raw_prediction).strip()
    if not text:
        return 0.0

    # Clean JSON: {"answer": "X"} or {"option": "X"}
    if text.startswith("{"):
        try:
            import json

            parsed = json.loads(text)
            if isinstance(parsed, dict) and any(k in parsed for k in ("answer", "option", "prediction")):
                return 0.9
        except (json.JSONDecodeError, ValueError):
            pass

    # Explicit answer pattern
    patterns = [
        r"\bANSWER\s*(?:IS|:|=)\s*[A-Z]\b",
        r"\bOPTION\s*[A-Z]\b",
        r"答案\s*(?:是|:|：)?\s*[A-Z]\b",
        r"选项\s*[A-Z]\b",
    ]
    for pattern in patterns:
        if re.search(pattern, text.upper()):
            return 0.5

    return 0.3


# ---------------------------------------------------------------------------
# Composite confidence
# ---------------------------------------------------------------------------


def compute_confidence(
    predicted_option: str | None,
    record: EvalRecord,
    evidence_text: str,
    raw_prediction: str | None,
    votes: dict[str, int] | None = None,
    config: VetoConfig | None = None,
) -> ConfidenceScore:
    """Compute the full composite confidence score for a prediction."""
    cfg = config or VetoConfig()

    vote_c = score_vote_confidence(votes)
    evidence_s = score_evidence_support(predicted_option, record, evidence_text) if cfg.enable_evidence_check else 0.0
    contradiction = score_evidence_contradiction(predicted_option, record, evidence_text) if cfg.enable_contradiction_check else 0.0
    certainty = score_model_certainty(raw_prediction)

    # Normalize weights so they sum to 1.0
    total_w = cfg.vote_weight + cfg.evidence_weight + cfg.contradiction_weight + cfg.certainty_weight
    if total_w == 0:
        total_w = 1.0

    composite = (
        cfg.vote_weight * vote_c
        + cfg.evidence_weight * evidence_s
        - cfg.contradiction_weight * contradiction
        + cfg.certainty_weight * certainty
    ) / total_w

    # Clamp to [0, 1]
    composite = max(0.0, min(1.0, composite))

    return ConfidenceScore(
        predicted_option=predicted_option or "",
        vote_confidence=vote_c,
        evidence_support=evidence_s,
        evidence_contradiction=contradiction,
        model_certainty=certainty,
        composite=composite,
        factors={
            "vote_weight": cfg.vote_weight,
            "evidence_weight": cfg.evidence_weight,
            "contradiction_weight": cfg.contradiction_weight,
            "certainty_weight": cfg.certainty_weight,
        },
    )


# ---------------------------------------------------------------------------
# Veto check
# ---------------------------------------------------------------------------


def check_veto(
    prediction: str | None,
    record: EvalRecord,
    evidence_text: str,
    raw_prediction: str | None,
    votes: dict[str, int] | None = None,
    config: VetoConfig | None = None,
) -> VetoResult:
    """Apply the one-vote veto check to a prediction.

    Steps:
    1. If *prediction* is ``None`` or config is disabled, return no-veto result.
    2. Compute :class:`ConfidenceScore` via :func:`compute_confidence`.
    3. If composite < threshold, the prediction is vetoed.
    4. Determine fallback via evidence-based or lexical heuristic.
    """
    cfg = config or VetoConfig()

    # No veto when disabled or no prediction
    if not cfg.enabled or prediction is None:
        conf = compute_confidence(prediction, record, evidence_text, raw_prediction, votes, cfg)
        return VetoResult(
            vetoed=False,
            original_prediction=prediction,
            fallback_prediction=None,
            confidence=conf,
        )

    conf = compute_confidence(prediction, record, evidence_text, raw_prediction, votes, cfg)

    if conf.composite >= cfg.threshold:
        return VetoResult(
            vetoed=False,
            original_prediction=prediction,
            fallback_prediction=None,
            confidence=conf,
        )

    # Veto triggered: find a fallback. In safe mode, the fallback must be
    # strongly grounded and the original prediction must be weakly grounded.
    fallback, fallback_support = _evidence_based_fallback(record, evidence_text, exclude_option=prediction)
    support_margin = fallback_support - conf.evidence_support
    if cfg.safe_mode and not _safe_veto_allowed(cfg, conf, fallback, fallback_support, support_margin):
        return VetoResult(
            vetoed=False,
            original_prediction=prediction,
            fallback_prediction=fallback,
            confidence=conf,
            veto_reason=(
                "safe_mode_blocked; "
                f"fallback_support={fallback_support:.3f}; "
                f"support_margin={support_margin:.3f}"
            ),
            fallback_support=fallback_support,
            support_margin=support_margin,
        )

    reasons: list[str] = []
    reasons.append(f"composite={conf.composite:.3f} < threshold={cfg.threshold:.3f}")
    if conf.evidence_contradiction > 0.6:
        reasons.append(f"strong_alternative_evidence={conf.evidence_contradiction:.3f}")
    if conf.vote_confidence < 0.5:
        reasons.append(f"low_vote_confidence={conf.vote_confidence:.3f}")
    if conf.evidence_support < 0.2:
        reasons.append(f"low_evidence_support={conf.evidence_support:.3f}")

    return VetoResult(
        vetoed=True,
        original_prediction=prediction,
        fallback_prediction=fallback,
        confidence=conf,
        veto_reason="; ".join(reasons),
        fallback_support=fallback_support,
        support_margin=support_margin,
    )


def _safe_veto_allowed(
    cfg: VetoConfig,
    conf: ConfidenceScore,
    fallback: str | None,
    fallback_support: float,
    support_margin: float,
) -> bool:
    if fallback is None:
        return False
    if fallback_support < cfg.min_fallback_support:
        return False
    if conf.evidence_support > cfg.max_original_support:
        return False
    if support_margin < cfg.min_support_margin:
        return False
    if conf.vote_confidence > cfg.max_veto_vote_confidence:
        return False
    return True


def _evidence_based_fallback(
    record: EvalRecord,
    evidence_text: str,
    exclude_option: str,
) -> tuple[str | None, float]:
    """Find the best-supported alternative option when a prediction is vetoed."""
    best_letter: str | None = None
    best_score = 0.0
    for letter in record.options:
        if letter == exclude_option:
            continue
        score = score_evidence_support(letter, record, evidence_text)
        if score > best_score:
            best_score = score
            best_letter = letter
    # Only return if there is meaningful support (>= 0.3)
    if best_letter and best_score >= 0.3:
        return best_letter, best_score

    # Lexical fallback: count option text occurrences in evidence
    evidence_lower = evidence_text.lower()
    hits: list[tuple[int, str]] = []
    for letter, opt_text in record.options.items():
        if letter == exclude_option:
            continue
        opt = str(opt_text).strip().lower()
        if opt:
            count = evidence_lower.count(opt)
            if count:
                hits.append((count, letter))
    if hits:
        hits.sort(reverse=True)
        return hits[0][1], 0.3

    return None, 0.0


# ---------------------------------------------------------------------------
# Shared text utilities (mirrors evidence.py patterns)
# ---------------------------------------------------------------------------

_STOP_WORDS = {"the", "and", "for", "from", "with", "which", "what", "this", "that", "shown", "based", "option"}


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*|[\u4e00-\u9fff]{2,}", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOP_WORDS}


def _number_strings(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])-?\d[\d,]*(?:\.\d+)?%?", text)
