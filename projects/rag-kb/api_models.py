# api_models.py
from config import TOP_K
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    k: int = Field(default=TOP_K, ge=1, le=10)
