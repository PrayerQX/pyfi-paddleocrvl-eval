from __future__ import annotations

from pathlib import Path

from .config import Settings, require_value
from .ernie_client import ErnieClient
from .io import collect_markdown, save_layout_result
from .paddleocr_client import FileType, LayoutOptions, PaddleOCRRemoteClient


def parse_document(
    settings: Settings,
    file_path: str | Path,
    *,
    file_type: FileType | None,
    output_dir: str | Path,
    options: LayoutOptions,
) -> tuple[dict, list[Path]]:
    client = PaddleOCRRemoteClient(
        api_url=settings.paddleocr_api_url,
        api_token=require_value(settings.paddleocr_api_token, "PADDLEOCR_API_TOKEN"),
        timeout=settings.timeout,
    )
    result = client.parse_file(file_path, file_type=file_type, options=options)
    written = save_layout_result(result, output_dir, timeout=settings.timeout)
    return result, written


def answer_from_document(
    settings: Settings,
    file_path: str | Path,
    *,
    file_type: FileType | None,
    output_dir: str | Path,
    options: LayoutOptions,
    question: str,
    choices: dict[str, str] | None = None,
    web_search: bool = True,
    max_completion_tokens: int = 65536,
    include_reasoning: bool = False,
) -> tuple[str, dict, list[Path]]:
    result, written = parse_document(
        settings,
        file_path,
        file_type=file_type,
        output_dir=output_dir,
        options=options,
    )
    markdown = collect_markdown(result)
    prompt = build_document_qa_prompt(markdown, question, choices)
    ernie = ErnieClient(
        api_key=require_value(settings.ernie_api_key, "ERNIE_API_KEY"),
        base_url=settings.ernie_base_url,
        model=settings.ernie_model,
        timeout=settings.timeout,
    )
    answer = ernie.complete(
        prompt,
        web_search=web_search,
        max_completion_tokens=max_completion_tokens,
        stream=True,
        include_reasoning=include_reasoning,
    )
    answer_path = Path(output_dir) / "answer.md"
    answer_path.write_text(answer, encoding="utf-8")
    written.append(answer_path)
    return answer, result, written


def build_document_qa_prompt(
    markdown: str,
    question: str,
    choices: dict[str, str] | None = None,
) -> str:
    lines = [
        "你是金融文档、图表和版面解析问答助手。",
        "下面是 PaddleOCR 远程 API 解析出的 Markdown 证据。",
        "请优先基于证据回答；证据不足时明确说明不足，不要编造。",
        "",
        "## 问题",
        question.strip(),
    ]
    if choices:
        lines.extend(
            [
                "",
                "## 选项",
                *[f"{key}. {value}" for key, value in sorted(choices.items())],
                "",
                "如果这是选择题，请先给出选项字母，再给出简短依据。",
            ]
        )
    lines.extend(["", "## PaddleOCR Markdown 证据", markdown.strip() or "(empty)"])
    return "\n".join(lines)
