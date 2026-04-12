from __future__ import annotations

import json

from .records import EvalRecord


CONTEXT_MODES = {
    "no_context",
    "image_background_only",
    "analysis_information_only",
    "both_contexts",
}


def build_mcq_prompt(record: EvalRecord, context_mode: str = "image_background_only") -> str:
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"Unknown context mode: {context_mode}")

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
    parts.extend(
        [
            f"question:\n{record.question}",
            f"options:\n{options}",
            (
                "Based on the image and the provided information, select the correct option. "
                f"Output ONLY one option letter from: {option_letters}. "
                "Do not output explanations, punctuation, or formatting."
            ),
        ]
    )
    return "\n\n".join(parts)
