from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_PADDLEOCR_API_URL = "https://i0u1edb895ael4d6.aistudio-app.com/layout-parsing"
DEFAULT_ERNIE_BASE_URL = "https://aistudio.baidu.com/llm/lmapi/v3"
DEFAULT_ERNIE_MODEL = "ernie-5.0-thinking-preview"


@dataclass(frozen=True, slots=True)
class Settings:
    paddleocr_api_url: str
    paddleocr_api_token: str
    ernie_api_key: str
    ernie_base_url: str
    ernie_model: str
    output_dir: Path
    timeout: float


def load_settings(env_file: str | Path | None = None) -> Settings:
    if env_file is not None:
        load_dotenv(env_file)
    else:
        load_dotenv()

    paddle_token = os.getenv("PADDLEOCR_API_TOKEN", "").strip()
    ernie_key = os.getenv("ERNIE_API_KEY", "").strip()
    output_dir = Path(os.getenv("REMOTE_PADDLEOCR_OUTPUT_DIR", "output"))
    timeout = float(os.getenv("REMOTE_PADDLEOCR_TIMEOUT", "180"))

    return Settings(
        paddleocr_api_url=os.getenv("PADDLEOCR_API_URL", DEFAULT_PADDLEOCR_API_URL).strip(),
        paddleocr_api_token=paddle_token,
        ernie_api_key=ernie_key,
        ernie_base_url=os.getenv("ERNIE_BASE_URL", DEFAULT_ERNIE_BASE_URL).strip(),
        ernie_model=os.getenv("ERNIE_MODEL", DEFAULT_ERNIE_MODEL).strip(),
        output_dir=output_dir,
        timeout=timeout,
    )


def require_value(value: str, name: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is required. Set it in environment variables or .env.")
    return value
