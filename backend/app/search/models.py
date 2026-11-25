from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SearchMode = Literal["keyword", "dense", "hybrid", "hybrid_rerank"]


class SearchFilters(BaseModel):
    location: str | None = Field(None, description="City or region, e.g. 'Bengaluru'")
    remote: bool | None = Field(None, description="True = remote jobs only")
    remote_ok: bool = Field(False, description="With a location: also include remote jobs")
    min_salary: int | None = Field(None, description="Yearly salary in INR")
    posted_within_days: int | None = Field(None, ge=1, le=365)
    exclude_ids: list[int] = Field(
        default_factory=list, description="Jobs to hide, including their reposts"
    )


class JobHit(BaseModel):
    id: int
    title: str
    company: str | None
    location: str | None
    remote: bool
    salary_min: int | None
    salary_max: int | None
    url: str | None
    posted_at: datetime | None
    snippet: str
    score: float = Field(description="Final ranking score (higher is better)")
    dense_rank: int | None = None
    keyword_rank: int | None = None
    personal_rank: int | None = None
    rerank_score: float | None = None
