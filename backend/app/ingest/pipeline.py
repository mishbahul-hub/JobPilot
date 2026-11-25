"""Ingestion pipeline: fetch -> normalise -> dedupe -> diff by content hash ->
embed only new/changed jobs -> upsert -> bump last_seen_at -> expire stale jobs.

Usage (from backend/):
    python -m app.ingest.pipeline adzuna --pages 5
    python -m app.ingest.pipeline adzuna --query "python developer" --pages 1 --dry-run
    python -m app.ingest.pipeline csv path/to/jobs.csv --source kaggle --limit 5000
"""

import argparse
import json
import time
from collections.abc import Callable

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.config import get_settings
from app.db.pool import apply_schema
from app.embeddings import DailyQuotaExceeded
from app.ingest.normalize import JobRecord, dedupe, plan_changes
from app.log import get_logger, setup_logging

log = get_logger("ingest")

WRITE_BATCH = 100

UPSERT_SQL = """
INSERT INTO jobs (source, external_id, title, company, location, remote, salary_min, salary_max,
                  description, url, posted_at, content_hash, embedding, is_active, last_seen_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, true, now())
ON CONFLICT (source, external_id) DO UPDATE SET
    title = EXCLUDED.title, company = EXCLUDED.company, location = EXCLUDED.location,
    remote = EXCLUDED.remote, salary_min = EXCLUDED.salary_min, salary_max = EXCLUDED.salary_max,
    description = EXCLUDED.description, url = EXCLUDED.url, posted_at = EXCLUDED.posted_at,
    content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding,
    is_active = true, last_seen_at = now()
"""

TOUCH_SQL = """
UPDATE jobs SET last_seen_at = now(), is_active = true
WHERE source = %s AND external_id = ANY(%s)
"""

# A job is stale if no run has seen it for `stale_days`, or it was posted too long ago.
EXPIRE_SQL = """
UPDATE jobs SET is_active = false
WHERE source = %s AND is_active
  AND (last_seen_at < now() - make_interval(days => %s)
       OR posted_at < now() - make_interval(days => %s))
"""


def existing_hashes(conn: psycopg.Connection, source: str, ids: list[str]) -> dict[str, str]:
    """Hashes of rows that are already embedded; rows without an embedding count as changed."""
    rows = conn.execute(
        "SELECT external_id, content_hash FROM jobs "
        "WHERE source = %s AND external_id = ANY(%s) AND embedding IS NOT NULL",
        (source, ids),
    ).fetchall()
    return dict(rows)


def _row(r: JobRecord, embedding: list[float]) -> tuple:
    return (
        r.source, r.external_id, r.title, r.company, r.location, r.remote, r.salary_min,
        r.salary_max, r.description, r.url, r.posted_at, r.content_hash, Vector(embedding),
    )


def ingest(
    conn: psycopg.Connection,
    records: list[JobRecord],
    source: str,
    embed: Callable[[list[str]], list[list[float]]],
    stale_days: int = 7,
    max_age_days: int = 60,
    max_embed: int | None = None,
) -> dict:
    started = time.monotonic()
    records = dedupe([r for r in records if r.source == source])
    plan = plan_changes(records, existing_hashes(conn, source, [r.external_id for r in records]))
    log.info("%d unique records: %s", len(records), plan.summary)

    # Newest jobs first, and optionally cap embeddings per run to leave quota for live searches.
    to_write = sorted(
        plan.to_write, key=lambda r: r.posted_at.timestamp() if r.posted_at else 0, reverse=True
    )
    if max_embed is not None:
        to_write = to_write[:max_embed]

    # Embed and write in batches and commit each one, so a crash keeps earlier progress.
    written, stopped_early = 0, False
    for i in range(0, len(to_write), WRITE_BATCH):
        batch = to_write[i : i + WRITE_BATCH]
        try:
            vectors = embed([r.embed_text for r in batch])
        except DailyQuotaExceeded:
            # Not fatal: the remaining jobs have no stored hash, so the next run picks them up.
            log.warning("daily embedding quota reached; %d jobs left for the next run",
                        len(plan.to_write) - written)
            stopped_early = True
            break
        with conn.cursor() as cur:
            cur.executemany(UPSERT_SQL, [_row(r, v) for r, v in zip(batch, vectors, strict=True)])
        conn.commit()
        written += len(batch)
        log.info("wrote %d/%d", written, len(to_write))

    # Mark every fetched job as seen (unchanged ones, and stored ones whose update is still
    # pending), so a job that is still listed never expires just because of the embed cap.
    written_ids = {r.external_id for r in to_write[:written]}
    seen = [r.external_id for r in records if r.external_id not in written_ids]
    if seen:
        conn.execute(TOUCH_SQL, (source, seen))
    expired = conn.execute(EXPIRE_SQL, (source, stale_days, max_age_days)).rowcount
    conn.commit()

    return {
        "source": source,
        "unique_records": len(records),
        "embedded_and_written": written,
        "pending_next_run": len(plan.to_write) - written,
        "unchanged": len(plan.unchanged),
        "stopped_early": stopped_early,
        "expired": expired,
        "seconds": round(time.monotonic() - started, 1),
    }


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load jobs into the JobPilot index.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("adzuna", help="fetch from the Adzuna API")
    a.add_argument("--query", action="append", help="search term (repeatable); default: built-in list")
    a.add_argument("--pages", type=int, default=5, help="pages per query (50 jobs each)")
    a.add_argument("--max-days-old", type=int, default=30)

    c = sub.add_parser("csv", help="load a CSV file")
    c.add_argument("path")
    c.add_argument("--source", default="csv")
    c.add_argument("--limit", type=int)

    for sp in (a, c):
        sp.add_argument("--stale-days", type=int, default=7)
        sp.add_argument("--max-age-days", type=int, default=60)
        sp.add_argument(
            "--max-embed", type=int, default=800,
            help="max jobs to embed this run (Gemini free tier: 1000/day shared with searches)",
        )
        sp.add_argument("--dry-run", action="store_true", help="fetch and normalise only")
    return p.parse_args(argv)


def main(argv=None) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    args = _parse_args(argv)

    if args.cmd == "adzuna":
        from app.ingest import adzuna

        source = adzuna.SOURCE
        records = adzuna.fetch(args.query or adzuna.DEFAULT_QUERIES, args.pages, args.max_days_old)
    else:
        from app.ingest.csv_source import load_csv

        source = args.source
        records = load_csv(args.path, source=source, limit=args.limit)

    if args.dry_run:
        unique = dedupe(records)
        print(json.dumps({"fetched": len(records), "unique": len(unique)}, indent=2))
        for r in unique[:3]:
            print(f"- {r.title} | {r.company} | {r.location} | remote={r.remote}")
        return

    from app.embeddings import embed_documents

    apply_schema(settings.database_url)
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        stats = ingest(
            conn, records, source, embed_documents, args.stale_days, args.max_age_days,
            max_embed=args.max_embed,
        )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
