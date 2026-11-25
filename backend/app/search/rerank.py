"""LLM reranker: one call scores the top hybrid candidates against the query (0-10).
Listwise scoring in a single structured-output call keeps cost to one request per search."""

from pydantic import BaseModel, Field

from app.llm import get_llm_pair
from app.log import get_logger
from app.search.models import JobHit

log = get_logger("rerank")

PROMPT = """You rank job postings for a job seeker's search.
Score each job 0-10 for how well it matches the search: role, seniority, skills and location.
10 = exactly what they asked for, 5 = related, 0 = unrelated. Judge only from the text given.

Search: {query}

Jobs:
{jobs}"""


class JobScore(BaseModel):
    job_id: int
    score: int = Field(ge=0, le=10)


class RerankResult(BaseModel):
    scores: list[JobScore]


def _describe(h: JobHit) -> str:
    loc = h.location or "unknown location"
    return f"[{h.id}] {h.title} | {h.company or '?'} | {loc}{' | remote' if h.remote else ''}\n" \
           f"    {h.snippet[:200]}"


def llm_rerank(query: str, hits: list[JobHit]) -> list[JobHit]:
    if not hits:
        return hits
    main, *rest = [m.with_structured_output(RerankResult) for m in get_llm_pair()]
    chain = main.with_fallbacks(rest) if rest else main
    result: RerankResult = chain.invoke(
        PROMPT.format(query=query, jobs="\n".join(_describe(h) for h in hits))
    )
    scores = {s.job_id: s.score for s in result.scores}
    if len(scores) < len(hits):
        log.warning("reranker scored %d of %d jobs", len(scores), len(hits))
    # Sort by LLM score; ties (and unscored jobs) keep their hybrid order.
    order = {h.id: n for n, h in enumerate(hits)}
    ranked = sorted(hits, key=lambda h: (-scores.get(h.id, -1), order[h.id]))
    return [h.model_copy(update={"rerank_score": scores.get(h.id)}) for h in ranked]
