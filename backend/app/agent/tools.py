"""Agent tools. The user id is never a tool argument: it comes from the run config, so the
model cannot read or change another user's data."""

import json

import psycopg
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from psycopg.rows import dict_row

from app import profile as profiles
from app.db.pool import get_pool
from app.embeddings import EmbeddingUnavailable
from app.log import get_logger
from app.search.hybrid import search
from app.search.models import SearchFilters

log = get_logger("tools")


def _user_id(config: RunnableConfig) -> str:
    return config["configurable"]["user_id"]


def _lpa(v: int | None) -> str | None:
    return f"{v / 100_000:.1f} LPA" if v else None


def _dump(obj) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


@tool
def search_jobs(
    query: str,
    config: RunnableConfig,
    location: str | None = None,
    remote_only: bool = False,
    min_salary_lpa: float | None = None,
    posted_within_days: int | None = None,
    limit: int = 8,
) -> str:
    """Search active job postings, ranked for this user (profile, saved and rated jobs).
    `query` is the role and skills in plain words, e.g. 'python backend internship'.
    `location` is a city (Bengaluru, Gurugram, Pune...); remote jobs are included with a location.
    Jobs the user disliked are left out. Returns jobs with their `id`."""
    filters = SearchFilters(
        location=location,
        remote_ok=bool(location),
        remote=remote_only or None,
        min_salary=int(min_salary_lpa * 100_000) if min_salary_lpa else None,
        posted_within_days=posted_within_days,
    )
    with get_pool().connection() as conn:
        hits = search(conn, query, filters, limit=max(1, min(limit, 10)), mode="hybrid",
                      user_id=_user_id(config))
    return _dump([
        {
            "id": h.id, "title": h.title, "company": h.company, "location": h.location,
            "remote": h.remote, "salary": _lpa(h.salary_max or h.salary_min),
            "posted": h.posted_at.date().isoformat() if h.posted_at else None,
            "snippet": h.snippet[:180],
        }
        for h in hits
    ])


@tool
def get_job(job_id: int) -> str:
    """Full details of one job posting by id."""
    with get_pool().connection() as conn:
        row = conn.cursor(row_factory=dict_row).execute(
            "SELECT id, title, company, location, remote, salary_min, salary_max, url, posted_at, "
            "is_active, left(description, 1500) AS description FROM jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
    return _dump(row) if row else _dump({"error": f"no job with id {job_id}"})


@tool
def save_job(job_id: int, config: RunnableConfig) -> str:
    """Save a job to the user's saved list (also counts as liking it for future rankings)."""
    with get_pool().connection() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO saved_jobs (user_id, job_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (_user_id(config), job_id),
            )
        except psycopg.errors.ForeignKeyViolation:
            return _dump({"error": f"no job with id {job_id}"})
    return _dump({"saved": job_id, "already_saved": cur.rowcount == 0})


@tool
def rate_job(job_id: int, liked: bool, config: RunnableConfig) -> str:
    """Record that the user likes or dislikes a job ('not interested', 'not for me', 'more like
    this'). Disliked jobs are hidden from future searches; both shape future rankings."""
    with get_pool().connection() as conn:
        try:
            conn.execute(
                "INSERT INTO feedback (user_id, job_id, rating) VALUES (%s, %s, %s)",
                (_user_id(config), job_id, 1 if liked else -1),
            )
        except psycopg.errors.ForeignKeyViolation:
            return _dump({"error": f"no job with id {job_id}"})
    return _dump({"rated": job_id, "liked": liked})


@tool
def list_saved_jobs(config: RunnableConfig) -> str:
    """The user's saved jobs, newest first."""
    with get_pool().connection() as conn:
        rows = conn.cursor(row_factory=dict_row).execute(
            "SELECT j.id, j.title, j.company, j.location, j.is_active, s.created_at AS saved_at "
            "FROM saved_jobs s JOIN jobs j ON j.id = s.job_id "
            "WHERE s.user_id = %s ORDER BY s.created_at DESC LIMIT 50",
            (_user_id(config),),
        ).fetchall()
    return _dump(rows)


@tool
def update_profile(
    config: RunnableConfig,
    add_skills: list[str] | None = None,
    add_titles: list[str] | None = None,
    add_locations: list[str] | None = None,
    remote_ok: bool | None = None,
    min_salary_lpa: float | None = None,
    years_exp: float | None = None,
    summary: str | None = None,
) -> str:
    """Remember facts the user shares about themselves. Lists are added to what is stored;
    other fields replace the stored value. Only pass what the user actually said."""
    uid = _user_id(config)
    with get_pool().connection() as conn:
        profiles.merge_profile(
            conn, uid, skills=add_skills, titles=add_titles, locations=add_locations,
            remote_ok=remote_ok, min_salary=int(min_salary_lpa * 100_000) if min_salary_lpa else None,
            years_exp=years_exp, summary=summary,
        )
        try:
            profiles.refresh_profile_embedding(conn, uid)
        except EmbeddingUnavailable as exc:  # quota must not lose the profile update
            log.warning("profile embedding not refreshed: %s", str(exc)[:200])
        return _dump(load_profile(uid, conn))


def load_profile(user_id: str, conn: psycopg.Connection | None = None) -> dict | None:
    """Profile for the system prompt / tool output (salary shown in LPA)."""
    if conn is None:
        with get_pool().connection() as c:
            return load_profile(user_id, c)
    row = profiles.load_profile(conn, user_id)
    if row:
        row["min_salary"] = _lpa(row["min_salary"])
    return row


# There is no get_profile tool: the profile is already in the system prompt.
TOOLS = [search_jobs, get_job, save_job, rate_job, list_saved_jobs, update_profile]
