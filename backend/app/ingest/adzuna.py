"""Adzuna job search API client. Docs: https://developer.adzuna.com/docs/search

Free keys are rate limited (roughly 25 calls/minute, 250/day), so calls are spaced out
and the default plan (14 queries x 5 pages x 50 results) stays well under the daily cap.
"""

import time
from collections.abc import Iterator
from datetime import datetime

import httpx

from app.config import get_settings
from app.ingest.normalize import JobRecord, clean_text, detect_remote, to_int
from app.log import get_logger

log = get_logger("adzuna")

SOURCE = "adzuna"
BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
RESULTS_PER_PAGE = 50
MIN_SECONDS_BETWEEN_CALLS = 2.6

DEFAULT_QUERIES = [
    "software engineer",
    "software intern",
    "python developer",
    "backend developer",
    "frontend developer",
    "full stack developer",
    "machine learning engineer",
    "AI engineer",
    "data scientist",
    "data engineer",
    "data analyst",
    "devops engineer",
    "android developer",
    "product manager",
]


class AdzunaClient:
    def __init__(self, app_id: str, app_key: str, country: str = "in"):
        if not app_id or not app_key:
            raise RuntimeError("ADZUNA_APP_ID and ADZUNA_APP_KEY must be set")
        self.app_id, self.app_key, self.country = app_id, app_key, country
        self.http = httpx.Client(timeout=30)
        self._last_call = 0.0
        self.calls = 0

    @classmethod
    def from_settings(cls) -> "AdzunaClient":
        s = get_settings()
        return cls(s.adzuna_app_id, s.adzuna_app_key, s.adzuna_country)

    def _get(self, page: int, params: dict) -> dict:
        for attempt in range(1, 5):
            wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            self.calls += 1
            resp = self.http.get(
                BASE_URL.format(country=self.country, page=page),
                params={"app_id": self.app_id, "app_key": self.app_key, **params},
            )
            if resp.status_code == 429:
                log.warning("rate limited, backing off (attempt %d)", attempt)
                time.sleep(15 * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("Adzuna kept returning 429")

    def search(self, what: str, pages: int, max_days_old: int) -> Iterator[dict]:
        for page in range(1, pages + 1):
            data = self._get(
                page,
                {
                    "what": what,
                    "results_per_page": RESULTS_PER_PAGE,
                    "max_days_old": max_days_old,
                    "sort_by": "date",
                    "content-type": "application/json",
                },
            )
            results = data.get("results", [])
            log.info("query=%r page=%d results=%d", what, page, len(results))
            yield from results
            if len(results) < RESULTS_PER_PAGE:
                break


def to_record(raw: dict) -> JobRecord | None:
    title = clean_text(raw.get("title"))
    if not raw.get("id") or not title:
        return None
    description = clean_text(raw.get("description"))
    location = clean_text((raw.get("location") or {}).get("display_name")) or None
    # Adzuna fills in estimated salaries; keep only salaries the employer actually posted.
    predicted = str(raw.get("salary_is_predicted", "0")) == "1"
    created = raw.get("created")
    return JobRecord(
        source=SOURCE,
        external_id=str(raw["id"]),
        title=title,
        company=clean_text((raw.get("company") or {}).get("display_name")) or None,
        location=location,
        remote=detect_remote(title, location, description),
        salary_min=None if predicted else to_int(raw.get("salary_min")),
        salary_max=None if predicted else to_int(raw.get("salary_max")),
        description=description,
        url=raw.get("redirect_url"),
        posted_at=datetime.fromisoformat(created) if created else None,
    )


def fetch(
    queries: list[str], pages: int, max_days_old: int, client: AdzunaClient | None = None
) -> list[JobRecord]:
    client = client or AdzunaClient.from_settings()
    records = []
    for q in queries:
        for raw in client.search(q, pages, max_days_old):
            if (rec := to_record(raw)) is not None:
                records.append(rec)
    log.info("fetched %d records in %d API calls", len(records), client.calls)
    return records
