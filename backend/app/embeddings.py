"""Gemini embeddings, shared by ingestion (documents) and search (queries).

Free-tier limits (seen in practice): 100 embedded texts per minute, 1000 per day.
Document calls are paced to stay under the per-minute limit; hitting the daily limit raises
DailyQuotaExceeded right away so the pipeline can stop cleanly and resume on the next run.
"""

import time
from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.log import get_logger

log = get_logger("embeddings")

BATCH_SIZE = 100
MAX_ATTEMPTS = 8


class EmbeddingUnavailable(RuntimeError):
    """Embedding failed after retries (quota, outage...)."""


class DailyQuotaExceeded(EmbeddingUnavailable):
    pass


@lru_cache
def get_embedder() -> GoogleGenerativeAIEmbeddings:
    s = get_settings()
    if not s.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    return GoogleGenerativeAIEmbeddings(
        model=s.embedding_model,
        google_api_key=s.google_api_key,
        output_dimensionality=s.embedding_dim,
    )


def _with_retry(fn, *args, max_attempts: int = MAX_ATTEMPTS, **kwargs):
    """Retry with backoff on transient errors (mostly per-minute 429s)."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc)
            if "per day" in msg or "PerDay" in msg:
                raise DailyQuotaExceeded("embedding daily quota exceeded") from exc
            if attempt == max_attempts:
                raise EmbeddingUnavailable(msg[:300]) from exc
            wait = min(60, 5 * 2 ** (attempt - 1))  # 5, 10, 20, 40, 60...
            reason = "rate limited (429)" if "RESOURCE_EXHAUSTED" in msg else msg[:200]
            log.warning("embedding call failed: %s; retry %d in %ds", reason, attempt, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


_next_doc_call_at = 0.0


def _pace(n_texts: int) -> None:
    """Spread document calls so we send at most `embed_texts_per_minute` texts per minute."""
    global _next_doc_call_at
    wait = _next_doc_call_at - time.monotonic()
    if wait > 0:
        log.info("pacing: waiting %.0fs for the per-minute quota", wait)
        time.sleep(wait)
    per_minute = get_settings().embed_texts_per_minute
    _next_doc_call_at = time.monotonic() + 60.0 * n_texts / per_minute


def embed_documents(texts: list[str]) -> list[list[float]]:
    _pace(len(texts))
    return _with_retry(
        get_embedder().embed_documents,
        texts,
        batch_size=BATCH_SIZE,
        task_type="RETRIEVAL_DOCUMENT",
    )


def embed_query(text: str) -> list[float]:
    # Live requests: fail fast (5s + 10s of backoff) instead of the batch job's long waits.
    return _with_retry(
        get_embedder().embed_query, text, max_attempts=3, task_type="RETRIEVAL_QUERY"
    )
