"""User profile: long-term memory used by the agent prompt and by personalised ranking."""

from uuid import UUID

import psycopg
from pgvector import Vector
from psycopg.rows import dict_row

from app.log import get_logger

log = get_logger("profile")

PROFILE_COLUMNS = "skills, titles, locations, remote_ok, min_salary, years_exp, summary"

# Lists are unioned with what's stored (case-insensitive dedupe, original order kept).
_MERGE_LIST = """(SELECT coalesce(array_agg(v ORDER BY o), '{{}}') FROM (
    SELECT DISTINCT ON (lower(v)) v, o FROM unnest(profiles.{col} || %({col})s::text[])
    WITH ORDINALITY AS t(v, o) ORDER BY lower(v), o) d)"""

UPSERT_SQL = f"""
INSERT INTO profiles (user_id, skills, titles, locations, remote_ok, min_salary, years_exp, summary)
VALUES (%(uid)s, %(skills)s, %(titles)s, %(locations)s, %(remote_ok)s, %(min_salary)s,
        %(years_exp)s, %(summary)s)
ON CONFLICT (user_id) DO UPDATE SET
    skills = {_MERGE_LIST.format(col="skills")},
    titles = {_MERGE_LIST.format(col="titles")},
    locations = {_MERGE_LIST.format(col="locations")},
    remote_ok = coalesce(%(remote_ok)s, profiles.remote_ok),
    min_salary = coalesce(%(min_salary)s, profiles.min_salary),
    years_exp = coalesce(%(years_exp)s, profiles.years_exp),
    summary = coalesce(%(summary)s, profiles.summary),
    updated_at = now()
"""


def _clean(items: list[str] | None) -> list[str]:
    return [s.strip() for s in items or [] if s and s.strip()]


def ensure_user(conn: psycopg.Connection, user_id: UUID | str) -> None:
    conn.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))


def merge_profile(
    conn: psycopg.Connection,
    user_id: UUID | str,
    *,
    skills: list[str] | None = None,
    titles: list[str] | None = None,
    locations: list[str] | None = None,
    remote_ok: bool | None = None,
    min_salary: int | None = None,
    years_exp: float | None = None,
    summary: str | None = None,
) -> None:
    conn.execute(UPSERT_SQL, {
        "uid": user_id, "skills": _clean(skills), "titles": _clean(titles),
        "locations": _clean(locations), "remote_ok": remote_ok, "min_salary": min_salary,
        "years_exp": years_exp, "summary": summary,
    })


def load_profile(conn: psycopg.Connection, user_id: UUID | str) -> dict | None:
    return conn.cursor(row_factory=dict_row).execute(
        f"SELECT {PROFILE_COLUMNS} FROM profiles WHERE user_id = %s", (user_id,)
    ).fetchone()


def profile_text(p: dict) -> str:
    """What the profile embedding represents: the kind of job this person fits.
    Locations are left out on purpose; they are hard filters, not meaning."""
    parts = []
    if p.get("titles"):
        parts.append("Target roles: " + ", ".join(p["titles"]))
    if p.get("skills"):
        parts.append("Skills: " + ", ".join(p["skills"]))
    if p.get("years_exp") is not None:
        parts.append(f"Experience: {p['years_exp']:g} years")
    if p.get("summary"):
        parts.append(p["summary"])
    return ". ".join(parts)


def refresh_profile_embedding(conn: psycopg.Connection, user_id: UUID | str) -> bool:
    """Re-embed the profile (cached by text, so unchanged profiles cost nothing)."""
    from app.search.hybrid import embed_query_cached

    p = load_profile(conn, user_id)
    text = profile_text(p) if p else ""
    if not text:
        return False
    vec = embed_query_cached(conn, text)
    conn.execute(
        "UPDATE profiles SET embedding = %s WHERE user_id = %s", (Vector(vec), user_id)
    )
    return True
