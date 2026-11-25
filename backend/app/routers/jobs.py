from typing import Annotated
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import get_settings
from app.db.pool import get_pool
from app.llm import llm_available
from app.ratelimit import limiter
from app.search.hybrid import JOB_COLUMNS, search
from app.search.models import JobHit, SearchFilters, SearchMode

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "/search",
    response_model=list[JobHit],
    dependencies=[Depends(limiter("search", get_settings().search_limit_per_min, 60))],
)
def search_jobs(
    q: Annotated[str, Query(min_length=2, max_length=300)],
    location: str | None = None,
    remote: bool | None = None,
    remote_ok: bool = False,
    min_salary: Annotated[int | None, Query(ge=0)] = None,
    posted_within_days: Annotated[int | None, Query(ge=1, le=365)] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    mode: SearchMode = "hybrid",
    user_id: UUID | None = None,
):
    if mode == "hybrid_rerank" and not llm_available():
        mode = "hybrid"  # reranker needs GOOGLE_API_KEY; degrade instead of failing
    filters = SearchFilters(
        location=location, remote=remote, remote_ok=remote_ok,
        min_salary=min_salary, posted_within_days=posted_within_days,
    )
    with get_pool().connection() as conn:
        return search(conn, q, filters, limit=limit, mode=mode,
                      user_id=str(user_id) if user_id else None)


@router.get("/{job_id}")
def get_job(job_id: int):
    with get_pool().connection() as conn:
        cur = conn.cursor(row_factory=psycopg.rows.dict_row)
        row = cur.execute(
            f"SELECT {JOB_COLUMNS}, description, is_active FROM jobs WHERE id = %s", (job_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    return row
