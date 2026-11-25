import psycopg
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.db.pool import get_pool

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    try:
        with get_pool().connection() as conn:
            active_jobs = conn.execute(
                "SELECT count(*) FROM jobs WHERE is_active"
            ).fetchone()[0]
        return {"status": "ok", "db": "ok", "active_jobs": active_jobs}
    except (psycopg.Error, RuntimeError):
        return JSONResponse(status_code=503, content={"status": "degraded", "db": "unreachable"})
