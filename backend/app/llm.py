"""Shared chat model (Gemini). The same GOOGLE_API_KEY serves chat and embeddings."""

from functools import lru_cache

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings


def llm_available() -> bool:
    return bool(get_settings().google_api_key)


@lru_cache
def _rate_limiter() -> InMemoryRateLimiter:
    # One limiter shared by every model instance: the free-tier quota is per key.
    rpm = get_settings().llm_requests_per_minute
    return InMemoryRateLimiter(requests_per_second=rpm / 60, max_bucket_size=2)


# Provider errors that mean "busy, try later" rather than a bug.
_TRANSIENT = ("503", "UNAVAILABLE", "504", "DEADLINE_EXCEEDED", "429", "RESOURCE_EXHAUSTED",
              "overloaded", "high demand")


class LLMUnavailable(RuntimeError):
    """Every configured model is busy or out of quota."""


def is_transient(exc: BaseException) -> bool:
    msg = str(exc)
    return any(t in msg for t in _TRANSIENT)


@lru_cache
def get_llm(temperature: float = 0.0, model: str | None = None) -> ChatGoogleGenerativeAI:
    s = get_settings()
    if not s.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    return ChatGoogleGenerativeAI(
        model=model or s.llm_model,
        google_api_key=s.google_api_key,
        temperature=temperature,
        rate_limiter=_rate_limiter(),
        max_retries=2,
        timeout=60,
    )


def get_llm_pair(temperature: float = 0.0) -> list[ChatGoogleGenerativeAI]:
    """Main model, then each fallback model, in order."""
    s = get_settings()
    return [get_llm(temperature, m) for m in [s.llm_model, *s.llm_fallback_models]]
