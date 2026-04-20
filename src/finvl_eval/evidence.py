from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser

from .records import EvalRecord


@dataclass(slots=True)
class EvidenceBundle:
    text: str
    stats: dict[str, int | str]


@dataclass(slots=True)
class EvidenceLine:
    source: str
    text: str
    score: int


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._current_table = []
        elif tag == "tr" and self._current_table is not None:
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []
            self._in_cell = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._current_row is not None and self._current_cell is not None:
            text = _squash_ws("".join(self._current_cell))
            self._current_row.append(text)
            self._current_cell = None
            self._in_cell = False
        elif tag == "tr" and self._current_table is not None and self._current_row is not None:
            if any(cell for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._current_table is not None:
            if self._current_table:
                self.tables.append(self._current_table)
            self._current_table = None

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._current_cell is not None:
            self._current_cell.append(data)


def build_grounded_evidence(record: EvalRecord, parsed_markdown: str, ocr_text: str) -> EvidenceBundle:
    query_text = " ".join([record.question, *record.options.values()])
    query_tokens = _keywords(query_text)
    query_numbers = set(_number_strings(query_text))
    route = _route(record)

    table_lines = _table_lines(parsed_markdown)
    markdown_lines = _plain_lines(_strip_html_tables(parsed_markdown), "paddle_vl_markdown")
    ocr_lines = _plain_lines(ocr_text, "paddle_ocr_text")
    all_lines = table_lines + markdown_lines + ocr_lines
    scored_lines = [
        EvidenceLine(line.source, line.text, _score_text(line.text, query_tokens, query_numbers))
        for line in all_lines
    ]
    relevant = [line for line in sorted(scored_lines, key=lambda item: item.score, reverse=True) if line.score > 0]
    if len(relevant) < 12:
        relevant = sorted(scored_lines, key=lambda item: item.score, reverse=True)

    option_evidence = _option_evidence(record, "\n".join(line.text for line in all_lines), relevant[:30])
    numeric_candidates = _numeric_candidates(record, relevant[:30])

    sections = [
        "Paddle-grounded evidence packet",
        f"task_route: {route}",
        f"capability: {record.capability}",
        f"complexity: {record.complexity}",
        "",
        "option_evidence:",
        *option_evidence,
        "",
        "calculation_candidates_exploratory:",
        *(numeric_candidates or ["- none"]),
        "",
        "most_relevant_paddle_rows:",
    ]
    for idx, line in enumerate(relevant[:24], start=1):
        sections.append(f"{idx}. [{line.source} score={line.score}] {line.text}")

    background = record.context.get("image_background")
    if background:
        sections.extend(["", "image_background:", _trim(str(background), 2500)])
    analysis = record.context.get("analysis_information")
    if analysis:
        sections.extend(["", "analysis_information:", _trim(str(analysis), 2500)])

    stats = {
        "route": route,
        "table_rows": len(table_lines),
        "ocr_lines": len(ocr_lines),
        "markdown_lines": len(markdown_lines),
        "relevant_rows": len(relevant),
    }
    return EvidenceBundle(text="\n".join(sections), stats=stats)


def build_remote_markdown_grounded_evidence(record: EvalRecord, parsed_markdown: str) -> EvidenceBundle:
    query_text = " ".join([record.question, *record.options.values()])
    query_tokens = _keywords(query_text)
    query_numbers = set(_number_strings(query_text))
    route = _route(record)

    table_lines = _table_lines(parsed_markdown)
    markdown_lines = _plain_lines(_strip_html_tables(parsed_markdown), "paddle_vl_markdown")
    all_lines = table_lines + markdown_lines
    scored_lines = [
        EvidenceLine(line.source, line.text, _score_text(line.text, query_tokens, query_numbers))
        for line in all_lines
    ]
    relevant = [line for line in sorted(scored_lines, key=lambda item: item.score, reverse=True) if line.score > 0]
    if len(relevant) < 12:
        relevant = sorted(scored_lines, key=lambda item: item.score, reverse=True)

    option_evidence = _option_evidence(record, "\n".join(line.text for line in all_lines), relevant[:30])
    numeric_candidates = _numeric_candidates(record, relevant[:30])

    sections = [
        "PaddleOCR-VL grounded evidence packet",
        f"task_route: {route}",
        f"capability: {record.capability}",
        f"complexity: {record.complexity}",
        "",
        "option_evidence:",
        *option_evidence,
        "",
        "calculation_candidates_exploratory:",
        *(numeric_candidates or ["- none"]),
        "",
        "most_relevant_paddle_rows:",
    ]
    for idx, line in enumerate(relevant[:24], start=1):
        sections.append(f"{idx}. [{line.source} score={line.score}] {line.text}")

    background = record.context.get("image_background")
    if background:
        sections.extend(["", "image_background:", _trim(str(background), 2500)])
    analysis = record.context.get("analysis_information")
    if analysis:
        sections.extend(["", "analysis_information:", _trim(str(analysis), 2500)])

    stats = {
        "route": route,
        "table_rows": len(table_lines),
        "markdown_lines": len(markdown_lines),
        "relevant_rows": len(relevant),
    }
    return EvidenceBundle(text="\n".join(sections), stats=stats)


def _route(record: EvalRecord) -> str:
    question = record.question.lower()
    capability = (record.capability or "").lower()
    if capability == "calculation_analysis" or re.search(
        r"\b(percentage|percent|change|decrease|increase|ratio|difference|average|sum|total|growth)\b",
        question,
    ):
        return "calculation_or_numeric_comparison"
    if capability == "data_extraction" or re.search(r"\b(what|which|in \d{4}|shown|value)\b", question):
        return "table_or_chart_lookup"
    if capability == "perception" or re.search(r"\b(color|legend|line|bar|axis|label)\b", question):
        return "visual_label_or_legend_lookup"
    if capability == "pattern_recognition" or re.search(r"\b(trend|highest|lowest|peak|decline|correlat)\b", question):
        return "pattern_or_trend_recognition"
    if capability == "logical_reasoning":
        return "grounded_logical_reasoning"
    if capability == "decision_support":
        return "grounded_decision_support"
    return "general_grounded_selection"


def _table_lines(markdown: str) -> list[EvidenceLine]:
    parser = _TableParser()
    parser.feed(markdown)
    lines: list[EvidenceLine] = []
    for table_idx, table in enumerate(parser.tables, start=1):
        header = table[0] if table else []
        for row_idx, row in enumerate(table[:80], start=1):
            cells = []
            for idx, cell in enumerate(row):
                if header and row_idx > 1 and idx < len(header) and header[idx] != cell:
                    cells.append(f"{header[idx]}={cell}")
                else:
                    cells.append(cell)
            text = " | ".join(cell for cell in cells if cell)
            if text:
                lines.append(EvidenceLine(f"paddle_vl_table_{table_idx}", text, 0))
    return lines


def _plain_lines(text: str, source: str) -> list[EvidenceLine]:
    return [
        EvidenceLine(source, line, 0)
        for line in (_squash_ws(raw) for raw in text.splitlines())
        if line and line not in {"|", "-"}
    ][:160]


def _strip_html_tables(text: str) -> str:
    return re.sub(r"<table\b.*?</table>", "\n", text, flags=re.IGNORECASE | re.DOTALL)


def _option_evidence(record: EvalRecord, evidence_text: str, relevant: list[EvidenceLine]) -> list[str]:
    evidence_lower = evidence_text.lower()
    relevant_text = "\n".join(line.text for line in relevant).lower()
    rows: list[str] = []
    for letter, option in record.options.items():
        option_text = str(option)
        option_tokens = _keywords(option_text)
        option_numbers = _number_strings(option_text)
        exact_hit = option_text.strip().lower() in evidence_lower if option_text.strip() else False
        relevant_hits = sum(1 for token in option_tokens if token in relevant_text)
        number_hits = [num for num in option_numbers if num in evidence_text]
        rows.append(
            "- "
            f"{letter}: text={option_text!r}; "
            f"exact_in_paddle={exact_hit}; "
            f"keyword_hits_in_relevant={relevant_hits}; "
            f"number_hits={number_hits}"
        )
    return rows


def _numeric_candidates(record: EvalRecord, relevant: list[EvidenceLine]) -> list[str]:
    question = record.question.lower()
    option_values = {
        letter: [_parse_number(num) for num in _number_strings(text)]
        for letter, text in record.options.items()
    }
    option_values = {letter: [num for num in nums if num is not None] for letter, nums in option_values.items()}
    evidence_values: list[tuple[float, str]] = []
    for line in relevant:
        for raw in _number_strings(line.text):
            value = _parse_number(raw)
            if value is not None:
                evidence_values.append((value, line.text))
    evidence_values = evidence_values[:80]

    candidates: list[tuple[float, str]] = []
    if any(word in question for word in ["percentage decrease", "percent decrease", "decrease"]):
        for old, old_line in evidence_values:
            if old == 0:
                continue
            for new, new_line in evidence_values:
                pct = ((old - new) / abs(old)) * 100
                _append_option_matches(candidates, option_values, pct, f"percentage_decrease {old:g} -> {new:g} = {pct:.2f}% | {old_line} / {new_line}")
    if any(word in question for word in ["percentage increase", "percent increase", "increase", "growth"]):
        for old, old_line in evidence_values:
            if old == 0:
                continue
            for new, new_line in evidence_values:
                pct = ((new - old) / abs(old)) * 100
                _append_option_matches(candidates, option_values, pct, f"percentage_increase {old:g} -> {new:g} = {pct:.2f}% | {old_line} / {new_line}")
    if any(word in question for word in ["difference", "change"]):
        for left, left_line in evidence_values:
            for right, right_line in evidence_values:
                diff = left - right
                _append_option_matches(candidates, option_values, diff, f"difference {left:g} - {right:g} = {diff:.2f} | {left_line} / {right_line}")

    deduped: list[str] = []
    seen: set[str] = set()
    for _, text in sorted(candidates, key=lambda item: item[0])[:12]:
        if text not in seen:
            deduped.append(f"- {text}")
            seen.add(text)
    return deduped


def _append_option_matches(
    candidates: list[tuple[float, str]],
    option_values: dict[str, list[float]],
    computed: float,
    description: str,
) -> None:
    for letter, values in option_values.items():
        for option_value in values:
            tolerance = max(0.75, abs(option_value) * 0.025)
            delta = abs(computed - option_value)
            if delta <= tolerance:
                candidates.append((delta, f"matches option {letter}: {description}"))


def _score_text(text: str, query_tokens: set[str], query_numbers: set[str]) -> int:
    lowered = text.lower()
    tokens = _keywords(text)
    number_hits = sum(4 for number in query_numbers if number and number in text)
    token_hits = sum(1 for token in query_tokens if token in tokens or token in lowered)
    unit_bonus = 2 if re.search(r"[%$€£¥]|percent|percentage|ratio|rate|gdp|revenue|profit|debt", lowered) else 0
    return token_hits + number_hits + unit_bonus


def _keywords(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "from",
        "with",
        "which",
        "what",
        "this",
        "that",
        "shown",
        "based",
        "option",
        "invest",
        "into",
        "without",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*|[\u4e00-\u9fff]{2,}", text.lower())
    return {token for token in tokens if len(token) > 2 and token not in stop}


def _number_strings(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])-?\d[\d,]*(?:\.\d+)?%?", text)


def _parse_number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").rstrip("%"))
    except ValueError:
        return None


def _squash_ws(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _trim(text: str, max_chars: int) -> str:
    text = _squash_ws(text)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars // 2]} ...[omitted]... {text[-max_chars // 2 :]}"
