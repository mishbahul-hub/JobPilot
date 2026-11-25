"""Per-client sliding-window rate limits for expensive endpoints.

A public demo on free-tier LLM/embedding quotas can be drained by one script; these limits keep
the quota for real users. In-memory, so per process: fine for a single small instance.
"""

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

_lock = threading.Lock()
_hits: dict[str, deque[float]] = defaultdict(deque)


def client_ip(request: Request) -> str:
    # Behind a proxy (Render etc.) uvicorn runs with --proxy-headers, so this is the real client.
    return request.client.host if request.client else "unknown"


def check(key: str, limit: int, window_s: int) -> None:
    now = time.monotonic()
    with _lock:
        q = _hits[key]
        while q and q[0] <= now - window_s:
            q.popleft()
        if len(q) >= limit:
            retry = int(q[0] + window_s - now) + 1
            raise HTTPException(
                status_code=429,
                detail=f"Too many requests. Try again in {retry} seconds.",
                headers={"Retry-After": str(retry)},
            )
        q.append(now)


def limiter(name: str, limit: int, window_s: int):
    """FastAPI dependency: `dependencies=[Depends(limiter("chat", 30, 600))]`."""

    def dep(request: Request) -> None:
        check(f"{name}:{client_ip(request)}", limit, window_s)

    return dep


def reset() -> None:
    with _lock:
        _hits.clear()
