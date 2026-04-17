from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from openai import OpenAI


class ErnieClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 180) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def complete(
        self,
        prompt: str,
        *,
        web_search: bool = True,
        max_completion_tokens: int = 65536,
        stream: bool = True,
        include_reasoning: bool = False,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        extra_body = {"web_search": {"enable": web_search}}
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            extra_body=extra_body,
            max_completion_tokens=max_completion_tokens,
        )
        if stream:
            return collect_stream(completion, include_reasoning=include_reasoning)
        choice = completion.choices[0]
        return choice.message.content or ""


def collect_stream(chunks: Iterable[Any], *, include_reasoning: bool = False) -> str:
    parts: list[str] = []
    for chunk in chunks:
        if not getattr(chunk, "choices", None):
            continue
        if len(chunk.choices) == 0:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        content = getattr(delta, "content", None)
        if include_reasoning and reasoning:
            parts.append(reasoning)
        if content:
            parts.append(content)
    return "".join(parts)
