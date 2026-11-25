"""Usage report for the live demo: real numbers for the README, resume and interviews.

Usage (from repo root):  .venv/Scripts/python scripts/usage_report.py [--days 30]
Reads only aggregate counts; prints no user data.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import psycopg

from app.config import get_settings

QUERIES = {
    "Users (any activity)": "SELECT count(DISTINCT c.user_id) FROM conversations c "
                            "WHERE c.created_at > now() - make_interval(days => %(d)s)",
    "Conversations": "SELECT count(*) FROM conversations "
                     "WHERE created_at > now() - make_interval(days => %(d)s)",
    "User messages": "SELECT count(*) FROM messages WHERE role = 'user' "
                     "AND created_at > now() - make_interval(days => %(d)s)",
    "Avg user messages per conversation":
        "SELECT round(avg(n), 1) FROM (SELECT count(*) n FROM messages m "
        "JOIN conversations c ON c.id = m.conversation_id WHERE m.role = 'user' "
        "AND c.created_at > now() - make_interval(days => %(d)s) GROUP BY c.id) x",
    "Jobs recommended": "SELECT coalesce(sum(jsonb_array_length(tool_calls -> 'matches')), 0) "
                        "FROM messages WHERE role = 'assistant' "
                        "AND created_at > now() - make_interval(days => %(d)s)",
    "Ungrounded job ids dropped": "SELECT coalesce(sum(jsonb_array_length("
                                  "tool_calls -> 'dropped_job_ids')), 0) FROM messages "
                                  "WHERE role = 'assistant' "
                                  "AND created_at > now() - make_interval(days => %(d)s)",
    "Profiles with skills": "SELECT count(*) FROM profiles WHERE cardinality(skills) > 0",
    "Saved jobs": "SELECT count(*) FROM saved_jobs "
                  "WHERE created_at > now() - make_interval(days => %(d)s)",
    "Feedback 👍": "SELECT count(*) FROM feedback WHERE rating = 1 "
                  "AND created_at > now() - make_interval(days => %(d)s)",
    "Feedback 👎": "SELECT count(*) FROM feedback WHERE rating = -1 "
                  "AND created_at > now() - make_interval(days => %(d)s)",
    "Active jobs in index": "SELECT count(*) FROM jobs WHERE is_active",
    "Jobs added in period": "SELECT count(*) FROM jobs "
                            "WHERE first_seen_at > now() - make_interval(days => %(d)s)",
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    days = p.parse_args().days
    with psycopg.connect(get_settings().database_url) as conn:
        print(f"JobPilot usage, last {days} days\n")
        for label, sql in QUERIES.items():
            print(f"{label:<38} {conn.execute(sql, {'d': days}).fetchone()[0]}")
        print("\nLatest evals:")
        for kind, created, metrics in conn.execute(
            # Alias the date: ORDER BY created_at would otherwise sort by the ::date output
            # column (same day for every run) instead of the timestamp.
            "SELECT DISTINCT ON (kind) kind, created_at::date AS day, metrics FROM eval_runs "
            "ORDER BY kind, eval_runs.created_at DESC"
        ).fetchall():
            short = metrics.get("summary", metrics)
            print(f"  {kind:<16} {created}  {short}")


if __name__ == "__main__":
    main()
