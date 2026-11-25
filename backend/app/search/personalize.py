"""Personalisation signals for ranking.

The user vector is a Rocchio-style combination:
    u = a * profile_embedding + b * mean(liked or saved jobs) - c * mean(disliked jobs)
It only *reorders* jobs that already match the query (see hybrid.search), so a Python developer
searching for "frontend React" still gets frontend jobs, just the ones closest to their profile.
Jobs the user disliked are excluded.
"""

import math
from dataclasses import dataclass, field
from uuid import UUID

import psycopg

ALPHA, BETA, GAMMA = 1.0, 0.75, 0.25

# Latest rating per job wins; saved jobs count as liked.
SIGNALS_SQL = """
WITH ratings AS (
    SELECT DISTINCT ON (job_id) job_id, rating FROM feedback
    WHERE user_id = %(uid)s AND job_id IS NOT NULL ORDER BY job_id, created_at DESC
),
liked AS (
    SELECT job_id FROM ratings WHERE rating = 1
    UNION SELECT job_id FROM saved_jobs WHERE user_id = %(uid)s
),
disliked AS (SELECT job_id FROM ratings WHERE rating = -1)
SELECT
    (SELECT embedding FROM profiles WHERE user_id = %(uid)s),
    (SELECT avg(j.embedding) FROM jobs j JOIN liked l ON l.job_id = j.id
        WHERE j.id NOT IN (SELECT job_id FROM disliked)),
    (SELECT avg(j.embedding) FROM jobs j JOIN disliked d ON d.job_id = j.id),
    (SELECT coalesce(array_agg(job_id), '{}') FROM disliked)
"""


@dataclass
class UserContext:
    vector: list[float] | None = None
    disliked_ids: list[int] = field(default_factory=list)


def _norm(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def rocchio(profile, liked, disliked) -> list[float] | None:
    parts = [(ALPHA, profile), (BETA, liked), (-GAMMA, disliked)]
    parts = [(w, _norm(list(v))) for w, v in parts if v is not None]
    if not parts or all(w < 0 for w, _ in parts):
        return None
    dim = len(parts[0][1])
    return _norm([sum(w * v[i] for w, v in parts) for i in range(dim)])


def _backfill_profile_embedding(conn: psycopg.Connection, user_id: UUID | str) -> bool:
    """A profile saved while the embedding API was unavailable has no vector yet; try now."""
    from app.embeddings import EmbeddingUnavailable
    from app.profile import refresh_profile_embedding

    missing = conn.execute(
        "SELECT 1 FROM profiles WHERE user_id = %s AND embedding IS NULL", (user_id,)
    ).fetchone()
    if not missing:
        return False
    try:
        return refresh_profile_embedding(conn, user_id)
    except EmbeddingUnavailable:
        return False


def user_context(conn: psycopg.Connection, user_id: UUID | str) -> UserContext:
    profile, liked, disliked, disliked_ids = conn.execute(
        SIGNALS_SQL, {"uid": user_id}
    ).fetchone()
    if profile is None and _backfill_profile_embedding(conn, user_id):
        profile = conn.execute(
            "SELECT embedding FROM profiles WHERE user_id = %s", (user_id,)
        ).fetchone()[0]
    tolist = (lambda v: v.to_list() if v is not None else None)
    return UserContext(
        vector=rocchio(tolist(profile), tolist(liked), tolist(disliked)),
        disliked_ids=list(disliked_ids),
    )
