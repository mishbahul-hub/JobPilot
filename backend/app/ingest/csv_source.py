"""Load jobs from a CSV file: the fallback when there's no Adzuna key, and the source for
fixed eval fixtures.

Column names are matched against aliases, so both our own format and the Kaggle
"LinkedIn Job Postings 2023-24" `postings.csv` load without changes.
"""

import csv
from datetime import UTC, datetime
from pathlib import Path

from app.ingest.normalize import JobRecord, clean_text, detect_remote, to_int

ALIASES = {
    "external_id": ["external_id", "job_id", "id"],
    "title": ["title", "job_title"],
    "company": ["company", "company_name"],
    "location": ["location"],
    "description": ["description", "job_description"],
    "url": ["url", "job_posting_url", "application_url"],
    "posted_at": ["posted_at", "listed_time", "original_listed_time", "created"],
    "salary_min": ["salary_min", "min_salary"],
    "salary_max": ["salary_max", "max_salary"],
    "remote": ["remote", "remote_allowed"],
}


def _pick(row: dict, field: str) -> str | None:
    for name in ALIASES[field]:
        if row.get(name) not in (None, ""):
            return row[name]
    return None


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        num = float(value)  # Kaggle uses epoch milliseconds
        return datetime.fromtimestamp(num / 1000 if num > 1e11 else num, tz=UTC)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        return None


def _truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def load_csv(path: str | Path, source: str = "csv", limit: int | None = None) -> list[JobRecord]:
    records = []
    csv.field_size_limit(10_000_000)  # long descriptions
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ext_id, title = _pick(row, "external_id"), clean_text(_pick(row, "title"))
            if not ext_id or not title:
                continue
            description = clean_text(_pick(row, "description"))
            location = clean_text(_pick(row, "location")) or None
            records.append(
                JobRecord(
                    source=source,
                    external_id=str(ext_id),
                    title=title,
                    company=clean_text(_pick(row, "company")) or None,
                    location=location,
                    remote=_truthy(_pick(row, "remote"))
                    or detect_remote(title, location, description),
                    salary_min=to_int(_pick(row, "salary_min")),
                    salary_max=to_int(_pick(row, "salary_max")),
                    description=description,
                    url=_pick(row, "url"),
                    posted_at=_parse_time(_pick(row, "posted_at")),
                )
            )
            if limit and len(records) >= limit:
                break
    return records
