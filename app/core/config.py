from __future__ import annotations

import os

from pydantic import BaseModel


def _parse_origins(value: str | None) -> list[str]:
    if not value:
        return []

    return [origin.strip() for origin in value.split(",") if origin.strip()]


class Settings(BaseModel):
    """Application settings loaded from process environment."""

    app_name: str = "RAG Debugger API"
    groq_api_key: str | None = None
    frontend_origins: list[str] = []


settings = Settings(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    frontend_origins=_parse_origins(os.getenv("FRONTEND_ORIGINS")),
)
