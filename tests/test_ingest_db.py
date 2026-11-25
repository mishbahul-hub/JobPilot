"""Integration tests against a real Postgres + pgvector. Skipped unless TEST_DATABASE_URL is set.
WARNING: deletes rows with source='pytest' from the jobs table."""

import os

import pytest
from test_ingest import make

from app.ingest.pipeline import ingest

DB_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL not set")
SOURCE = "pytest"


class FakeEmbedder:
    def __init__(self):
        self.calls = 0
        self.texts = 0

    def __call__(self, texts):
        self.calls += 1
        self.texts += len(texts)
        return [[0.01] * 768 for _ in texts]


@pytest.fixture
def conn():
    import psycopg
    from pgvector.psycopg import register_vector

    from app.db.pool import apply_schema

    apply_schema(DB_URL)
    with psycopg.connect(DB_URL) as c:
        register_vector(c)
        c.execute("DELETE FROM jobs WHERE source = %s", (SOURCE,))
        c.commit()
        yield c
        c.execute("DELETE FROM jobs WHERE source = %s", (SOURCE,))
        c.commit()


def records(**kw):
    return [make(str(i), source=SOURCE, **kw) for i in range(3)]


def test_second_run_embeds_nothing(conn):
    emb = FakeEmbedder()
    first = ingest(conn, records(), SOURCE, emb)
    assert first["embedded_and_written"] == 3

    second = ingest(conn, records(), SOURCE, emb)
    assert second["embedded_and_written"] == 0 and second["unchanged"] == 3
    assert emb.texts == 3


def test_changed_job_is_re_embedded(conn):
    emb = FakeEmbedder()
    ingest(conn, records(), SOURCE, emb)
    changed = records()
    changed[1] = make("1", source=SOURCE, description="Now with Kubernetes.")
    stats = ingest(conn, changed, SOURCE, emb)
    assert stats["embedded_and_written"] == 1
    desc = conn.execute(
        "SELECT description FROM jobs WHERE source = %s AND external_id = '1'", (SOURCE,)
    ).fetchone()[0]
    assert desc == "Now with Kubernetes."


def test_daily_quota_stops_cleanly_and_resumes(conn):
    from app.embeddings import DailyQuotaExceeded

    class QuotaAfterOneBatch(FakeEmbedder):
        def __call__(self, texts):
            if self.calls >= 1:
                raise DailyQuotaExceeded("quota")
            return super().__call__(texts)

    many = [make(str(i), source=SOURCE) for i in range(150)]
    first = ingest(conn, many, SOURCE, QuotaAfterOneBatch())
    assert first["stopped_early"] and first["embedded_and_written"] == 100
    assert first["pending_next_run"] == 50

    emb = FakeEmbedder()
    second = ingest(conn, many, SOURCE, emb)
    assert second["embedded_and_written"] == 50 and second["unchanged"] == 100
    assert emb.texts == 50


def test_max_embed_caps_run_newest_first(conn):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    recs = [make(str(i), source=SOURCE, posted_at=now - timedelta(days=i)) for i in range(5)]
    stats = ingest(conn, recs, SOURCE, FakeEmbedder(), max_embed=2)
    assert stats["embedded_and_written"] == 2 and stats["pending_next_run"] == 3
    stored = conn.execute(
        "SELECT external_id FROM jobs WHERE source = %s ORDER BY 1", (SOURCE,)
    ).fetchall()
    assert [r[0] for r in stored] == ["0", "1"]  # the two newest


def test_stale_jobs_expire(conn):
    ingest(conn, records(), SOURCE, FakeEmbedder())
    conn.execute(
        "UPDATE jobs SET last_seen_at = now() - interval '10 days' "
        "WHERE source = %s AND external_id = '0'",
        (SOURCE,),
    )
    conn.commit()
    stats = ingest(conn, records()[1:], SOURCE, FakeEmbedder(), stale_days=7)
    assert stats["expired"] == 1
    active = conn.execute(
        "SELECT external_id FROM jobs WHERE source = %s AND is_active ORDER BY 1", (SOURCE,)
    ).fetchall()
    assert [r[0] for r in active] == ["1", "2"]


def test_search_sql_with_filters(conn):
    from app.search.hybrid import dense_ids, keyword_ids
    from app.search.models import SearchFilters

    recs = [
        make("blr", source=SOURCE, location="Bangalore, Karnataka"),
        make("pune", source=SOURCE, location="Pune"),
        make("rem", source=SOURCE, location="Chennai", remote=True),
    ]
    ingest(conn, recs, SOURCE, FakeEmbedder())
    ids = dict(conn.execute(
        "SELECT external_id, id FROM jobs WHERE source = %s", (SOURCE,)).fetchall())
    mine = set(ids.values())

    def kw(f):
        return {i for i in keyword_ids(conn, "fastapi", f, 500) if i in mine}

    assert kw(SearchFilters(location="Bengaluru")) == {ids["blr"]}  # alias match
    assert kw(SearchFilters(location="Bengaluru", remote_ok=True)) == {ids["blr"], ids["rem"]}
    assert kw(SearchFilters(remote=True)) == {ids["rem"]}
    assert kw(SearchFilters(exclude_ids=[ids["pune"]])) == {ids["blr"], ids["rem"]}

    dense = dense_ids(conn, [0.01] * 768, SearchFilters(location="pune"), 500)
    assert {i for i in dense if i in mine} == {ids["pune"]}


def _onehot(i: int) -> list[float]:
    v = [0.0] * 768
    v[i] = 1.0
    return v


@pytest.fixture
def user(conn):
    import uuid

    from app.profile import ensure_user

    uid = uuid.uuid4()
    ensure_user(conn, uid)
    conn.commit()
    yield uid
    conn.execute("DELETE FROM users WHERE id = %s", (uid,))
    conn.commit()


def test_merge_profile_unions_lists(conn, user):
    from app.profile import load_profile, merge_profile

    merge_profile(conn, user, skills=["Python", "SQL"], years_exp=1)
    merge_profile(conn, user, skills=["python", "Docker", " "], titles=["Backend Intern"])
    p = load_profile(conn, user)
    assert p["skills"] == ["Python", "SQL", "Docker"]  # case-insensitive dedupe, order kept
    assert p["titles"] == ["Backend Intern"] and p["years_exp"] == 1  # untouched field kept


def test_personalised_search_reorders_and_hides_disliked(conn, user):
    from pgvector import Vector

    from app.profile import merge_profile
    from app.search.hybrid import search
    from app.search.personalize import user_context

    recs = [make(str(i), source=SOURCE, title=f"FastAPI Dev {i}") for i in range(3)]
    recs.append(make("1-repost", source=SOURCE, title="FastAPI Dev 1"))  # same job, new id
    ingest(conn, recs, SOURCE, FakeEmbedder())
    ids = dict(conn.execute(
        "SELECT external_id, id FROM jobs WHERE source = %s", (SOURCE,)).fetchall())
    for ext, dim in (("0", 0), ("1", 1), ("2", 2), ("1-repost", 1)):
        conn.execute("UPDATE jobs SET embedding = %s WHERE id = %s",
                     (Vector(_onehot(dim)), ids[ext]))
    merge_profile(conn, user, skills=["FastAPI"])
    conn.execute("UPDATE profiles SET embedding = %s WHERE user_id = %s",
                 (Vector(_onehot(2)), user))
    conn.execute("INSERT INTO saved_jobs (user_id, job_id) VALUES (%s, %s)", (user, ids["0"]))
    conn.execute("INSERT INTO feedback (user_id, job_id, rating) VALUES (%s, %s, -1)",
                 (user, ids["1"]))
    conn.commit()

    ctx = user_context(conn, user)
    assert ctx.disliked_ids == [ids["1"]]
    assert ctx.vector[2] > ctx.vector[0] > 0 > ctx.vector[1]  # profile > saved > disliked

    mine = set(ids.values())
    hits = search(conn, "fastapi", limit=50, mode="keyword", user_id=str(user),
                  personal_weight=2.0)
    order = [h.id for h in hits if h.id in mine]
    assert ids["1"] not in order  # disliked job hidden
    assert ids["1-repost"] not in order  # ...and its repost
    assert order[0] == ids["2"]  # closest to the profile comes first


def test_full_text_column_is_generated(conn):
    ingest(conn, records(), SOURCE, FakeEmbedder())
    hit = conn.execute(
        "SELECT count(*) FROM jobs WHERE source = %s AND tsv @@ plainto_tsquery('english', 'fastapi')",
        (SOURCE,),
    ).fetchone()[0]
    assert hit == 3
