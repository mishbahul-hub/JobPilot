"""Retrieval eval: compares keyword / dense / hybrid (/ hybrid_rerank) on golden queries.

Relevance labels are rules (regexes on title, location, company, remote) applied to the live
index, so they stay valid as jobs change daily. Rules judge mostly by job title, which is a
fair but strict proxy: a relevant job with an unusual title counts as a miss for every mode.

Usage (from repo root):
    .venv/Scripts/python evals/retrieval_eval.py              # all available modes
    .venv/Scripts/python evals/retrieval_eval.py --modes keyword hybrid --no-save
Query embeddings are cached in Postgres, so reruns cost no embedding quota.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from app.config import get_settings
from app.db.pool import apply_schema
from app.llm import llm_available
from app.search.hybrid import search
from app.search.metrics import ndcg_at_k, precision_at_k, reciprocal_rank
from app.search.models import SearchFilters

GOLDEN = ROOT / "evals" / "golden_queries.jsonl"
RESULTS = ROOT / "evals" / "results"
K = 10


def relevant_ids(conn: psycopg.Connection, rel: dict) -> set[int]:
    clauses, params = ["is_active"], {}
    for field, col in [("title", "title"), ("title2", "title"), ("location", "location"),
                       ("company", "company")]:
        if field in rel:
            clauses.append(f"{col} ~* %({field})s")
            params[field] = rel[field]
    if "remote" in rel:
        clauses.append("remote = %(remote)s")
        params["remote"] = rel["remote"]
    sql = f"SELECT id FROM jobs WHERE {' AND '.join(clauses)}"
    return {r[0] for r in conn.execute(sql, params).fetchall()}


def run(modes: list[str], save: bool) -> dict:
    settings = get_settings()
    golden = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line]
    apply_schema(settings.database_url)

    per_query, skipped = [], []
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        active = conn.execute("SELECT count(*) FROM jobs WHERE is_active").fetchone()[0]
        for g in golden:
            rel = relevant_ids(conn, g["rel"])
            if not rel:
                skipped.append(g["id"])
                continue
            filters = SearchFilters(**g.get("filters", {}))
            row = {"id": g["id"], "query": g["query"], "n_relevant": len(rel)}
            for mode in modes:
                ranked = [h.id for h in search(conn, g["query"], filters, limit=K, mode=mode)]
                row[mode] = {
                    "p@10": precision_at_k(ranked, rel, K),
                    "mrr": reciprocal_rank(ranked, rel),
                    "ndcg@10": ndcg_at_k(ranked, rel, K),
                }
            per_query.append(row)
            conn.commit()  # keep cached query embeddings even if a later query fails

        summary = {
            mode: {m: round(mean(q[mode][m] for q in per_query), 3)
                   for m in ("p@10", "mrr", "ndcg@10")}
            for mode in modes
        }
        report = {
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "active_jobs": active,
            "n_queries": len(per_query),
            "skipped_no_relevant": skipped,
            "summary": summary,
            "per_query": per_query,
        }
        if save:
            conn.execute(
                "INSERT INTO eval_runs (kind, config, metrics) VALUES ('retrieval', %s, %s)",
                (Jsonb({"modes": modes, "k": K, "golden": GOLDEN.name}),
                 Jsonb({k: v for k, v in report.items() if k != "per_query"})),
            )
            conn.commit()
    return report


def to_markdown(report: dict) -> str:
    modes = list(report["summary"])
    lines = [
        f"# Retrieval eval — {report['created_at']}",
        "",
        (f"{report['n_queries']} queries, {report['active_jobs']} active jobs, k={K}. "
        f"Skipped (no relevant jobs in index): {', '.join(report['skipped_no_relevant']) or 'none'}."),
        "",
        "| Mode | P@10 | MRR | nDCG@10 |",
        "|---|---|---|---|",
    ]
    for mode in modes:
        s = report["summary"][mode]
        lines.append(f"| {mode} | {s['p@10']:.3f} | {s['mrr']:.3f} | {s['ndcg@10']:.3f} |")
    lines += ["", "## Per query (nDCG@10)", "",
              "| Query | #rel | " + " | ".join(modes) + " |",
              "|---|---|" + "---|" * len(modes)]
    for q in report["per_query"]:
        cells = " | ".join(f"{q[m]['ndcg@10']:.2f}" for m in modes)
        lines.append(f"| {q['query']} | {q['n_relevant']} | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles
    p = argparse.ArgumentParser()
    p.add_argument("--modes", nargs="+", choices=["keyword", "dense", "hybrid", "hybrid_rerank"])
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()
    modes = args.modes or ["keyword", "dense", "hybrid"] + (
        ["hybrid_rerank"] if llm_available() else []
    )

    report = run(modes, save=not args.no_save)
    md = to_markdown(report)
    print(md)
    if not args.no_save:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "retrieval_latest.md").write_text(md, encoding="utf-8")
        (RESULTS / "retrieval_latest.json").write_text(json.dumps(report, indent=2), "utf-8")


if __name__ == "__main__":
    main()
