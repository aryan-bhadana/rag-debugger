from pydantic import BaseModel, validator


class HealthResponse(BaseModel):
    """Example response schema placeholder."""

    message: str


class QueryRequest(BaseModel):
    query: str
    auto_fix: bool = False

    @validator("query", pre=True)
    def strip_query(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value
