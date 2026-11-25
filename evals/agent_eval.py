"""Agent eval: scripted conversations against the real agent (live Gemini + live index).

Per turn, deterministic checks:
  expect_tools / forbid_tools   tool-call accuracy
  expect_matches                recommends jobs (or deliberately doesn't)
  match_title / match_location / match_remote   share of matches that satisfy the request
  saved_prev_match / excludes_prev_match        ordinal resolution and dislike handling
  grounding                     no job ids dropped by the grounding check
Plus an LLM judge (1-5) for relevance and explanation faithfulness on turns with matches.
The judge prefers a different model than the agent (LLM_FALLBACK_MODELS[-1]) to limit
self-preference bias.

Usage (from repo root):
    .venv/Scripts/python evals/agent_eval.py [--only id1 id2] [--no-judge] [--no-save]
"""

import argparse
import json
import re
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from app.agent.schemas import ChatRequest
from app.agent.service import close_agent, open_agent, run_chat
from app.config import get_settings
from app.db.pool import apply_schema, close_pool, get_pool, open_pool
from app.llm import get_llm

SCENARIOS = ROOT / "evals" / "agent_scenarios.jsonl"
RESULTS = ROOT / "evals" / "results"
MIN_MATCH_SHARE = 0.6  # at least 60% of matches must satisfy title/location/remote rules


class Judgement(BaseModel):
    relevance: int = Field(ge=1, le=5, description="Do the jobs fit what the user asked for?")
    faithfulness: int = Field(
        ge=1, le=5, description="Are the why_it_fits claims supported by the job data shown?"
    )
    reason: str = Field(description="One sentence")


JUDGE_PROMPT = """You are grading a job-search assistant. Be strict.

User request: {request}
Assistant reply: {reply}

Recommended jobs (with the real job data):
{jobs}

Score 1-5:
- relevance: do these jobs fit the request (role, level, location)? 5 = all clearly fit.
- faithfulness: is each why_it_fits supported by that job's data? 5 = nothing invented.
"""


def judge(request: str, reply: str, matches: list[dict], jobs: dict[int, dict]) -> dict:
    lines = []
    for m in matches:
        j = jobs.get(m["job_id"], {})
        lines.append(
            f"- {j.get('title')} | {j.get('company')} | {j.get('location')}"
            f"{' | remote' if j.get('remote') else ''}\n  data: {j.get('snippet', '')[:300]}\n"
            f"  why_it_fits: {m['why_it_fits']}"
        )
    s = get_settings()
    models = [get_llm(0.0, m) for m in [*reversed(s.llm_fallback_models), s.llm_model]]
    main, *rest = [m.with_structured_output(Judgement) for m in models]
    chain = main.with_fallbacks(rest) if rest else main
    j: Judgement = chain.invoke(JUDGE_PROMPT.format(request=request, reply=reply,
                                                    jobs="\n".join(lines)))
    return j.model_dump()


def job_rows(ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    with get_pool().connection() as conn:
        rows = conn.cursor(row_factory=dict_row).execute(
            "SELECT id, title, company, location, remote, left(description, 400) AS snippet "
            "FROM jobs WHERE id = ANY(%s)", (ids,)).fetchall()
    return {r["id"]: r for r in rows}


def saved_ids(user_id: uuid.UUID) -> set[int]:
    with get_pool().connection() as conn:
        return {r[0] for r in conn.execute(
            "SELECT job_id FROM saved_jobs WHERE user_id = %s", (user_id,)).fetchall()}


def share(matches, jobs, pred) -> float:
    return sum(1 for m in matches if pred(jobs.get(m["job_id"], {}))) / len(matches)


def check_turn(t: dict, res, user_id, prev_matches, jobs) -> dict[str, bool]:
    used = set(res.tools_used)
    matches = [m.model_dump() for m in res.matches]
    c: dict[str, bool] = {}
    if "expect_tools" in t:
        c["expected_tools"] = set(t["expect_tools"]) <= used
    if "forbid_tools" in t:
        c["no_forbidden_tools"] = not (set(t["forbid_tools"]) & used)
    if "expect_matches" in t:
        c["matches_presence"] = bool(matches) == t["expect_matches"]
    if matches and "match_title" in t:
        rx = re.compile(t["match_title"], re.IGNORECASE)
        c["match_title"] = share(matches, jobs, lambda j: bool(rx.search(j.get("title") or ""))) \
            >= MIN_MATCH_SHARE
    if matches and "match_location" in t:
        rx = re.compile(t["match_location"], re.IGNORECASE)
        c["match_location"] = share(
            matches, jobs, lambda j: j.get("remote") or bool(rx.search(j.get("location") or ""))
        ) >= MIN_MATCH_SHARE
    if matches and t.get("match_remote"):
        c["match_remote"] = share(matches, jobs, lambda j: bool(j.get("remote"))) >= MIN_MATCH_SHARE
    if "saved_prev_match" in t:
        n = t["saved_prev_match"]
        c["saved_right_job"] = len(prev_matches) >= n and \
            prev_matches[n - 1]["job_id"] in saved_ids(user_id)
    if "excludes_prev_match" in t:
        n = t["excludes_prev_match"]
        c["excludes_disliked"] = len(prev_matches) >= n and \
            prev_matches[n - 1]["job_id"] not in {m["job_id"] for m in matches}
    c["grounded"] = not res.dropped_job_ids
    return c


def run(only: list[str] | None, use_judge: bool, save: bool) -> dict:
    s = get_settings()
    apply_schema(s.database_url)
    open_pool(s.database_url)
    open_agent(s.database_url)
    scenarios = [json.loads(x) for x in SCENARIOS.read_text(encoding="utf-8").splitlines() if x]
    scenarios = [sc for sc in scenarios if not only or sc["id"] in only]
    results, users = [], []
    try:
        for sc in scenarios:
            user_id = uuid.uuid4()
            users.append(user_id)
            conv, prev = None, []
            for i, t in enumerate(sc["turns"], 1):
                if t.get("new_conversation"):
                    conv = None
                started = time.monotonic()
                row = {"scenario": sc["id"], "turn": i, "user": t["user"]}
                try:
                    res = run_chat(ChatRequest(user_id=user_id, conversation_id=conv,
                                               message=t["user"]))
                except Exception as exc:  # noqa: BLE001 - record the failure, keep evaluating
                    row.update(error=f"{type(exc).__name__}: {str(exc)[:200]}",
                               checks={"no_error": False}, latency_s=None)
                    results.append(row)
                    print(f"[{sc['id']} #{i}] ERROR {row['error']}", flush=True)
                    break
                conv = res.conversation_id
                matches = [m.model_dump() for m in res.matches]
                jobs = job_rows([m["job_id"] for m in matches])
                row.update(
                    latency_s=round(time.monotonic() - started, 1),
                    tools=res.tools_used, reply=res.reply,
                    matches=[{**m, "location": jobs.get(m["job_id"], {}).get("location")}
                             for m in matches],
                    checks=check_turn(t, res, user_id, prev, jobs),
                )
                if use_judge and matches:
                    try:
                        row["judge"] = judge(t["user"], res.reply, matches, jobs)
                    except Exception as exc:  # noqa: BLE001 - judge failure must not hide results
                        row["judge_error"] = str(exc)[:200]
                results.append(row)
                ok = all(row["checks"].values())
                print(f"[{sc['id']} #{i}] {'PASS' if ok else 'FAIL'} {row['latency_s']}s "
                      f"tools={res.tools_used} checks={row['checks']} "
                      f"judge={row.get('judge', {}).get('relevance')}", flush=True)
                prev = matches or prev
    finally:
        with get_pool().connection() as conn:
            conn.execute("DELETE FROM users WHERE id = ANY(%s)", (users,))
        close_agent()
        close_pool()

    all_checks = [v for r in results for v in r["checks"].values()]
    tool_checks = [r["checks"][k] for r in results for k in ("expected_tools", "no_forbidden_tools")
                   if k in r["checks"]]
    judged = [r["judge"] for r in results if "judge" in r]
    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    summary = {
        "scenarios": len(scenarios),
        "turns": len(results),
        "turn_pass_rate": round(mean(all(r["checks"].values()) for r in results), 3),
        "scenario_pass_rate": round(mean(
            all(all(r["checks"].values()) for r in results if r["scenario"] == sc["id"])
            for sc in scenarios), 3),
        "check_pass_rate": round(mean(all_checks), 3),
        "tool_accuracy": round(mean(tool_checks), 3) if tool_checks else None,
        "grounded_rate": round(mean(r["checks"].get("grounded", False) for r in results), 3),
        "judge_relevance": round(mean(j["relevance"] for j in judged), 2) if judged else None,
        "judge_faithfulness": round(mean(j["faithfulness"] for j in judged), 2) if judged else None,
        "latency_median_s": median(latencies) if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
        "errors": sum(1 for r in results if "error" in r),
    }
    report = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "model": s.llm_model, "summary": summary, "turns": results}
    if save:
        with open_db() as conn:
            conn.execute(
                "INSERT INTO eval_runs (kind, config, metrics) VALUES ('agent', %s, %s)",
                (Jsonb({"model": s.llm_model, "scenarios": [sc["id"] for sc in scenarios],
                        "judge": use_judge}), Jsonb(summary)),
            )
    return report


def open_db():
    import psycopg

    return psycopg.connect(get_settings().database_url, autocommit=True)


def to_markdown(r: dict) -> str:
    s = r["summary"]
    out = [f"# Agent eval — {r['created_at']} ({r['model']})", "",
           "| Metric | Value |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in s.items()]
    out += ["", "## Turns", "", "| Scenario | # | Pass | Tools | Failed checks | Judge rel/faith | s |",
            "|---|---|---|---|---|---|---|"]
    for t in r["turns"]:
        failed = [k for k, v in t["checks"].items() if not v]
        j = t.get("judge")
        out.append(
            f"| {t['scenario']} | {t['turn']} | {'✅' if not failed else '❌'} | "
            f"{', '.join(t.get('tools', [])) or '-'} | {', '.join(failed) or '-'} | "
            f"{f'{j['relevance']}/{j['faithfulness']}' if j else '-'} | {t.get('latency_s')} |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()
    report = run(args.only, use_judge=not args.no_judge, save=not args.no_save)
    md = to_markdown(report)
    print(md)
    if not args.no_save:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "agent_latest.md").write_text(md, encoding="utf-8")
        (RESULTS / "agent_latest.json").write_text(json.dumps(report, indent=2, default=str),
                                                   "utf-8")


if __name__ == "__main__":
    main()
