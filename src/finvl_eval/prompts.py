from __future__ import annotations

import json

from .records import EvalRecord


CONTEXT_MODES = {
    "no_context",
    "image_background_only",
    "analysis_information_only",
    "both_contexts",
}

PROMPT_STYLES = {
    "direct",
    "pyramid",
}


def build_mcq_prompt(
    record: EvalRecord,
    context_mode: str = "image_background_only",
    prompt_style: str = "direct",
) -> str:
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"Unknown context mode: {context_mode}")
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    parts: list[str] = []
    if context_mode in {"image_background_only", "both_contexts"}:
        background = record.context.get("image_background")
        if background:
            parts.append(f"image_background:\n{background}")
    if context_mode in {"analysis_information_only", "both_contexts"}:
        analysis = record.context.get("analysis_information")
        if analysis:
            parts.append(f"analysis_information:\n{analysis}")

    options = json.dumps(record.options, ensure_ascii=False)
    option_letters = ", ".join(record.options.keys())
    if prompt_style == "pyramid":
        instruction = (
            "Answer the multiple-choice question by following this fixed pyramid procedure:\n"
            "1. Identify the chart/document type, legend, axes, units, key regions, and visual marks.\n"
            "2. Extract only the values, categories, labels, or trends directly relevant to the question.\n"
            "3. Compare, calculate, or reason over that extracted evidence as needed.\n"
            "4. Select the best supported option.\n"
            f"Output ONLY one option letter from: {option_letters}. "
            "Do not output explanations, punctuation, or formatting."
        )
    else:
        instruction = (
            "Based on the image and the provided information, select the correct option. "
            f"Output ONLY one option letter from: {option_letters}. "
            "Do not output explanations, punctuation, or formatting."
        )
    parts.extend(
        [
            f"question:\n{record.question}",
            f"options:\n{options}",
            instruction,
        ]
    )
    return "\n\n".join(parts)
