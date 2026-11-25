"""Hybrid job search: pgvector (meaning) + Postgres full-text (keywords), fused with
Reciprocal Rank Fusion, optionally reranked by an LLM.

    keyword  - ts_rank_cd over the weighted tsv column (title > company > description)
    dense    - cosine distance on Gemini embeddings (HNSW index)
    hybrid   - RRF(dense top-50, keyword top-50)
    hybrid_rerank - hybrid top-20 reranked by the LLM
"""

import hashlib
import re

import psycopg
from pgvector import Vector

from app.config import get_settings
from app.embeddings import EmbeddingUnavailable, embed_query
from app.log import get_logger
from app.search.locations import aliases_of, find_city, location_patterns
from app.search.models import JobHit, SearchFilters, SearchMode

log = get_logger("search")

CANDIDATES = 50  # per retriever, before fusion
RRF_K = 60  # standard RRF constant
KEYWORD_WEIGHT = 0.5  # sweep 1.0/0.7/0.5/0.3 -> nDCG@10 .777/.785/.788/.789; 0.5 avoids overfitting
RERANK_POOL = 20
# Personalisation eval (vague queries) nDCG@10: none .041, w 0.5/1/2/3/4 -> .091/.131/.179/.238/.269.
# It keeps rising because that eval never penalises ignoring the query; 1.0 = same weight as the
# dense list, so the profile can tie the query's own ranking but never outvote it.
PERSONAL_WEIGHT = 1.0

_WORD_RE = re.compile(r"[a-z0-9]+")

JOB_COLUMNS = (
    "id, title, company, location, remote, salary_min, salary_max, url, posted_at, "
    "left(description, 300) AS snippet"
)


# ---------- filters ----------

def build_filters(f: SearchFilters) -> tuple[str, dict]:
    """SQL conditions (AND-ed after `is_active`) and their parameters."""
    clauses, params = [], {}
    if f.location:
        params["loc"] = location_patterns(f.location)
        loc_sql = "location ILIKE ANY(%(loc)s)"
        clauses.append(f"({loc_sql} OR remote)" if f.remote_ok else loc_sql)
    if f.remote:
        clauses.append("remote")
    if f.min_salary is not None:
        # Most postings have no salary; don't drop them, only those that are clearly below.
        clauses.append("(salary_max IS NULL OR salary_max >= %(min_salary)s)")
        params["min_salary"] = f.min_salary
    if f.posted_within_days:
        clauses.append("posted_at >= now() - make_interval(days => %(days)s)")
        params["days"] = f.posted_within_days
    if f.exclude_ids:
        # Also hide reposts of excluded jobs (same title, company, location, new id).
        clauses.append(
            "id <> ALL(%(exclude)s) AND NOT EXISTS (SELECT 1 FROM jobs x "
            "WHERE x.id = ANY(%(exclude)s) AND lower(x.title) = lower(jobs.title) "
            "AND x.company IS NOT DISTINCT FROM jobs.company "
            "AND x.location IS NOT DISTINCT FROM jobs.location)"
        )
        params["exclude"] = f.exclude_ids
    return "".join(f" AND {c}" for c in clauses), params


# ---------- retrievers ----------

# Words that describe the search itself, not the job; they match almost every posting.
FILLER = {
    "job", "jobs", "role", "roles", "position", "positions", "opening", "openings", "vacancy",
    "vacancies", "hiring", "looking", "someone", "who", "work", "career", "opportunity",
}


def or_tsquery(text: str) -> str | None:
    """'Python backend jobs, Bengaluru!' -> 'python | backend | bengaluru | bangalore'
    (OR, not AND: a job rarely contains every word of a natural-language query; city
    aliases added because postings spell cities differently)."""
    words = [w for w in _WORD_RE.findall(text.lower()) if w not in FILLER]
    expanded = []
    for w in words:
        expanded += sorted(aliases_of(w)) if find_city(w) else [w]
    return " | ".join(dict.fromkeys(expanded)) or None


def understand_query(query: str, f: SearchFilters) -> SearchFilters:
    """A city named in the query becomes a location filter (remote jobs still allowed)."""
    if f.location or f.remote:
        return f
    city = find_city(query)
    return f.model_copy(update={"location": city, "remote_ok": True}) if city else f


def keyword_ids(conn: psycopg.Connection, query: str, f: SearchFilters, k: int) -> list[int]:
    tsq = or_tsquery(query)
    if not tsq:
        return []
    where, params = build_filters(f)
    sql = f"""
        SELECT id FROM jobs, to_tsquery('english', %(tsq)s) q
        WHERE is_active AND tsv @@ q {where}
        ORDER BY ts_rank_cd(tsv, q, 32) DESC, posted_at DESC NULLS LAST
        LIMIT %(k)s
    """
    return [r[0] for r in conn.execute(sql, {**params, "tsq": tsq, "k": k}).fetchall()]


def dense_ids(conn: psycopg.Connection, qvec: list[float], f: SearchFilters, k: int) -> list[int]:
    where, params = build_filters(f)
    # With filters, iterative scan keeps HNSW from returning too few rows (pgvector >= 0.8).
    conn.execute("SET LOCAL hnsw.ef_search = 100")
    conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
    sql = f"""
        SELECT id FROM jobs
        WHERE is_active AND embedding IS NOT NULL {where}
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
    """
    return [r[0] for r in conn.execute(sql, {**params, "q": Vector(qvec), "k": k}).fetchall()]


def rrf_fuse(
    *rankings: list[int], weights: list[float] | None = None, k: int = RRF_K
) -> list[tuple[int, float]]:
    """Weighted Reciprocal Rank Fusion: score(d) = sum over lists of w / (k + rank)."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ---------- query embedding cache ----------

def _cache_key(text: str) -> str:
    s = get_settings()
    norm = " ".join(text.lower().split())
    return hashlib.sha256(f"{s.embedding_model}:{s.embedding_dim}:{norm}".encode()).hexdigest()


def embed_query_cached(conn: psycopg.Connection, text: str) -> list[float]:
    key = _cache_key(text)
    row = conn.execute(
        "UPDATE query_cache SET hits = hits + 1 WHERE query_key = %s RETURNING embedding", (key,)
    ).fetchone()
    if row:
        return row[0].to_list()
    vec = embed_query(text)
    conn.execute(
        "INSERT INTO query_cache (query_key, embedding) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (key, Vector(vec)),
    )
    return vec


# ---------- public API ----------

def dedupe_hits(hits) -> list[JobHit]:
    """Job boards repost the same job under new IDs; keep the best-ranked copy."""
    seen, out = set(), []
    for h in hits:
        key = (h.title.lower().strip(), (h.company or "").lower(), (h.location or "").lower())
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


def fetch_jobs(conn: psycopg.Connection, ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    cur = conn.cursor(row_factory=psycopg.rows.dict_row)
    rows = cur.execute(f"SELECT {JOB_COLUMNS} FROM jobs WHERE id = ANY(%s)", (ids,)).fetchall()
    return {r["id"]: r for r in rows}


def personal_ids(conn: psycopg.Connection, candidates: list[int], uvec: list[float]) -> list[int]:
    """Order the query's candidate jobs by closeness to the user vector."""
    if not candidates:
        return []
    return [r[0] for r in conn.execute(
        "SELECT id FROM jobs WHERE id = ANY(%s) AND embedding IS NOT NULL "
        "ORDER BY embedding <=> %s",
        (candidates, Vector(uvec)),
    ).fetchall()]


def search(
    conn: psycopg.Connection,
    query: str,
    filters: SearchFilters | None = None,
    limit: int = 10,
    mode: SearchMode = "hybrid",
    auto_location: bool = True,
    user_id: str | None = None,
    personal_weight: float | None = None,
) -> list[JobHit]:
    f = filters or SearchFilters()
    if auto_location:
        f = understand_query(query, f)

    uctx = None
    if user_id:
        from app.search.personalize import user_context

        uctx = user_context(conn, user_id)
        if uctx.disliked_ids:
            f = f.model_copy(update={"exclude_ids": [*f.exclude_ids, *uctx.disliked_ids]})

    dn: list[int] = []
    if mode != "keyword":
        try:
            dn = dense_ids(conn, embed_query_cached(conn, query), f, CANDIDATES)
        except EmbeddingUnavailable as exc:
            # Degrade to keyword search instead of failing (e.g. daily embedding quota).
            log.warning("dense search unavailable, keyword only: %s", str(exc)[:120])
            mode = "keyword" if mode == "dense" else mode
    kw = keyword_ids(conn, query, f, CANDIDATES) if mode != "dense" else []

    # Each retriever contributes a ranked list; personalisation adds a third list that only
    # reorders the query's own candidates, so it can't pull in off-topic jobs.
    lists, weights = [], []
    if mode != "keyword":
        lists.append(dn)
        weights.append(1.0)
    if mode != "dense":
        lists.append(kw)
        weights.append(KEYWORD_WEIGHT if mode != "keyword" else 1.0)
    pers: list[int] = []
    if uctx and uctx.vector:
        pers = personal_ids(conn, list(dict.fromkeys(dn + kw)), uctx.vector)
        lists.append(pers)
        weights.append(PERSONAL_WEIGHT if personal_weight is None else personal_weight)
    ranked = rrf_fuse(*lists, weights=weights)

    pool_size = max(limit, RERANK_POOL if mode == "hybrid_rerank" else limit)
    top = ranked[: pool_size * 2]  # extra room for duplicates dropped below
    rows = fetch_jobs(conn, [i for i, _ in top])
    kw_rank = {i: r for r, i in enumerate(kw, 1)}
    dn_rank = {i: r for r, i in enumerate(dn, 1)}
    pr_rank = {i: r for r, i in enumerate(pers, 1)}
    hits = dedupe_hits(
        JobHit(**rows[i], score=round(s, 6), dense_rank=dn_rank.get(i),
               keyword_rank=kw_rank.get(i), personal_rank=pr_rank.get(i))
        for i, s in top
        if i in rows
    )[:pool_size]

    if mode == "hybrid_rerank":
        from app.search.rerank import llm_rerank

        hits = llm_rerank(query, hits)
    return hits[:limit]
