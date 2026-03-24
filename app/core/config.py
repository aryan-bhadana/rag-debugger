from __future__ import annotations

import os

from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings placeholder."""

    app_name: str = "RAG Debugger API"
    groq_api_key: str | None = None


settings = Settings(
    groq_api_key=os.getenv("GROQ_API_KEY"),
)
