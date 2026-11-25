"""Personalisation eval: do profile-aware rankings help when the query is vague?

Each persona (evals/personas.jsonl) gets a temporary user with a profile + profile embedding.
The same vague queries ("software engineer", ...) are run without and with the user, at several
personal weights. Relevance = the persona's role rule (+ the city, if the query names one).

Usage (from repo root):
    .venv/Scripts/python evals/personalization_eval.py [--weights 0.5 1 2] [--no-save]
Embeddings are cached, so reruns cost no quota. Temporary users are deleted afterwards.
"""

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "evals"))

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from retrieval_eval import relevant_ids

from app.config import get_settings
from app.db.pool import apply_schema
from app.profile import ensure_user, merge_profile, refresh_profile_embedding
from app.search.hybrid import search
from app.search.metrics import ndcg_at_k, precision_at_k

PERSONAS = ROOT / "evals" / "personas.jsonl"
RESULTS = ROOT / "evals" / "results"
QUERIES = [
    ("software engineer", None),
    ("developer jobs in Bengaluru", "bangalore|bengaluru"),
    ("engineering jobs", None),
]
NAMESPACE = uuid.UUID("6f1c2b7e-7c39-4e0e-9a43-1d1f5a0c0e11")
K = 10


def run(weights: list[float], save: bool) -> dict:
    s = get_settings()
    apply_schema(s.database_url)
    personas = [json.loads(x) for x in PERSONAS.read_text(encoding="utf-8").splitlines() if x]
    configs = ["none", *[f"w={w:g}" for w in weights]]
    rows = []
    with psycopg.connect(s.database_url) as conn:
        register_vector(conn)
        uids = []
        try:
            for p in personas:
                uid = uuid.uuid5(NAMESPACE, p["id"])
                uids.append(uid)
                ensure_user(conn, uid)
                conn.execute("DELETE FROM profiles WHERE user_id = %s", (uid,))
                merge_profile(conn, uid, **p["profile"])
                refresh_profile_embedding(conn, uid)
                conn.commit()

                for query, loc_rule in QUERIES:
                    rel_rule = dict(p["rel"], **({"location": loc_rule} if loc_rule else {}))
                    rel = relevant_ids(conn, rel_rule)
                    if not rel:
                        continue
                    row = {"persona": p["id"], "query": query, "n_relevant": len(rel)}
                    for cfg, w in zip(configs, [None, *weights], strict=True):
                        hits = search(conn, query, limit=K, user_id=str(uid) if w else None,
                                      personal_weight=w)
                        ranked = [h.id for h in hits]
                        row[cfg] = {"ndcg@10": ndcg_at_k(ranked, rel, K),
                                    "p@10": precision_at_k(ranked, rel, K)}
                    rows.append(row)
                    conn.commit()
        finally:
            conn.rollback()
            conn.execute("DELETE FROM users WHERE id = ANY(%s)", (uids,))
            conn.commit()

        summary = {c: {m: round(mean(r[c][m] for r in rows), 3) for m in ("ndcg@10", "p@10")}
                   for c in configs}
        report = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                  "n_cases": len(rows), "summary": summary, "per_case": rows}
        if save:
            conn.execute(
                "INSERT INTO eval_runs (kind, config, metrics) VALUES ('personalization', %s, %s)",
                (Jsonb({"weights": weights, "queries": [q for q, _ in QUERIES]}),
                 Jsonb({k: v for k, v in report.items() if k != "per_case"})),
            )
            conn.commit()
    return report


def to_markdown(r: dict) -> str:
    configs = list(r["summary"])
    out = [f"# Personalisation eval — {r['created_at']}", "",
           f"{r['n_cases']} persona × query cases, k={K}. 'none' = no user; w = personal weight in RRF.",
           "", "| Ranking | nDCG@10 | P@10 |", "|---|---|---|"]
    out += [f"| {c} | {r['summary'][c]['ndcg@10']:.3f} | {r['summary'][c]['p@10']:.3f} |"
            for c in configs]
    out += ["", "## Per case (nDCG@10)", "", "| Persona | Query | #rel | " + " | ".join(configs) + " |",
            "|---|---|---|" + "---|" * len(configs)]
    out += [f"| {x['persona']} | {x['query']} | {x['n_relevant']} | "
            + " | ".join(f"{x[c]['ndcg@10']:.2f}" for c in configs) + " |" for x in r["per_case"]]
    return "\n".join(out) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()
    report = run(args.weights, save=not args.no_save)
    md = to_markdown(report)
    print(md)
    if not args.no_save:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "personalization_latest.md").write_text(md, encoding="utf-8")
        (RESULTS / "personalization_latest.json").write_text(json.dumps(report, indent=2), "utf-8")


if __name__ == "__main__":
    main()
