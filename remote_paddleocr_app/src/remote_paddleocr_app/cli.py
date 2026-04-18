from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_settings, require_value
from .ernie_client import ErnieClient
from .eval_pyfi import add_eval_parser, evaluate
from .paddleocr_client import LayoutOptions
from .pipeline import answer_from_document, parse_document
from .stack_results import add_stack_parser, stack_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote PaddleOCR layout parsing and ERNIE QA.")
    parser.add_argument("--env-file", type=Path, help="Optional .env path.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse a PDF or image with remote PaddleOCR.")
    add_document_args(parse_parser)

    ask_parser = subparsers.add_parser("ask", help="Parse a document and ask ERNIE about it.")
    add_document_args(ask_parser)
    ask_parser.add_argument("--question", required=True)
    ask_parser.add_argument(
        "--option",
        action="append",
        default=[],
        help="Choice in KEY=VALUE form. Can be repeated, for example --option A=Yes.",
    )
    ask_parser.add_argument("--disable-web-search", action="store_true")
    ask_parser.add_argument("--max-completion-tokens", type=int, default=65536)
    ask_parser.add_argument("--include-reasoning", action="store_true")

    chat_parser = subparsers.add_parser("chat", help="Ask ERNIE directly.")
    chat_parser.add_argument("prompt")
    chat_parser.add_argument("--disable-web-search", action="store_true")
    chat_parser.add_argument("--max-completion-tokens", type=int, default=65536)
    chat_parser.add_argument("--include-reasoning", action="store_true")

    add_eval_parser(subparsers)
    add_stack_parser(subparsers)

    args = parser.parse_args()
    settings = load_settings(args.env_file)

    if args.command == "parse":
        result, written = parse_document(
            settings,
            args.file,
            file_type=args.file_type,
            output_dir=args.out or settings.output_dir,
            options=layout_options_from_args(args),
        )
        print(f"Parsed {len(result.get('layoutParsingResults', []))} layout result(s).")
        print_written(written)
        return

    if args.command == "ask":
        answer, _result, written = answer_from_document(
            settings,
            args.file,
            file_type=args.file_type,
            output_dir=args.out or settings.output_dir,
            options=layout_options_from_args(args),
            question=args.question,
            choices=parse_options(args.option),
            web_search=not args.disable_web_search,
            max_completion_tokens=args.max_completion_tokens,
            include_reasoning=args.include_reasoning,
        )
        print(answer)
        print_written(written)
        return

    if args.command == "chat":
        ernie = ErnieClient(
            api_key=require_value(settings.ernie_api_key, "ERNIE_API_KEY"),
            base_url=settings.ernie_base_url,
            model=settings.ernie_model,
            timeout=settings.timeout,
        )
        print(
            ernie.complete(
                args.prompt,
                web_search=not args.disable_web_search,
                max_completion_tokens=args.max_completion_tokens,
                stream=True,
                include_reasoning=args.include_reasoning,
            )
        )
        return

    if args.command == "eval-pyfi":
        evaluate(args, settings)
        return

    if args.command == "stack-results":
        stack_results(args)
        return


def add_document_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("file", type=Path, help="Local PDF or image path.")
    parser.add_argument("--file-type", choices=["pdf", "image"], help="Defaults to extension inference.")
    parser.add_argument("--out", type=Path, help="Output directory.")
    parser.add_argument("--use-doc-orientation-classify", action="store_true")
    parser.add_argument("--use-doc-unwarping", action="store_true")
    parser.add_argument("--use-chart-recognition", action="store_true")


def layout_options_from_args(args: argparse.Namespace) -> LayoutOptions:
    return LayoutOptions(
        use_doc_orientation_classify=args.use_doc_orientation_classify,
        use_doc_unwarping=args.use_doc_unwarping,
        use_chart_recognition=args.use_chart_recognition,
    )


def parse_options(values: list[str]) -> dict[str, str] | None:
    if not values:
        return None
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --option value, expected KEY=VALUE: {value}")
        key, option_text = value.split("=", 1)
        key = key.strip().upper()
        option_text = option_text.strip()
        if not key or not option_text:
            raise SystemExit(f"Invalid --option value, expected KEY=VALUE: {value}")
        parsed[key] = option_text
    return parsed


def print_written(paths: list[Path]) -> None:
    if not paths:
        return
    print("Written files:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
