"""Source-independent job record, cleaning, hashing and change detection."""

import hashlib
import html
import re
from dataclasses import dataclass
from datetime import datetime

MAX_EMBED_CHARS = 4000

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_REMOTE_STRONG = re.compile(r"\b(remote|work from home|work-from-home|wfh)\b", re.IGNORECASE)
_REMOTE_IN_DESC = re.compile(
    r"\b(fully remote|100% remote|remote[- ]first|remote position|remote role|"
    r"work from home|work-from-home|wfh)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class JobRecord:
    source: str
    external_id: str
    title: str
    company: str | None
    location: str | None
    remote: bool
    salary_min: int | None
    salary_max: int | None
    description: str
    url: str | None
    posted_at: datetime | None

    @property
    def content_hash(self) -> str:
        """Changes whenever a stored field changes, so the row is rewritten and re-embedded."""
        parts = [
            self.title,
            self.company or "",
            self.location or "",
            str(self.remote),
            str(self.salary_min or ""),
            str(self.salary_max or ""),
            self.description,
            self.url or "",
        ]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    @property
    def embed_text(self) -> str:
        lines = [self.title]
        if self.company:
            lines.append(f"Company: {self.company}")
        if self.location:
            lines.append(f"Location: {self.location}" + (" (remote)" if self.remote else ""))
        elif self.remote:
            lines.append("Location: remote")
        lines.append(self.description)
        return "\n".join(lines)[:MAX_EMBED_CHARS]


def clean_text(text: str | None) -> str:
    """Strip HTML tags and entities, collapse whitespace."""
    if not text:
        return ""
    text = html.unescape(_TAG_RE.sub(" ", text))
    return _WS_RE.sub(" ", text).strip()


def detect_remote(title: str, location: str | None, description: str) -> bool:
    if _REMOTE_STRONG.search(title) or (location and _REMOTE_STRONG.search(location)):
        return True
    return bool(_REMOTE_IN_DESC.search(description))


def to_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return round(float(value))
    except (TypeError, ValueError):
        return None


@dataclass
class ChangePlan:
    to_write: list[JobRecord]  # new or changed: embed and upsert
    unchanged: list[JobRecord]  # only bump last_seen_at

    @property
    def summary(self) -> dict[str, int]:
        return {"to_write": len(self.to_write), "unchanged": len(self.unchanged)}


def dedupe(records: list[JobRecord]) -> list[JobRecord]:
    """Keep the last record per external_id (the same job shows up under several queries)."""
    by_id: dict[str, JobRecord] = {}
    for r in records:
        by_id[r.external_id] = r
    return list(by_id.values())


def plan_changes(records: list[JobRecord], existing: dict[str, str]) -> ChangePlan:
    """`existing` maps external_id -> content_hash for rows that already have an embedding."""
    to_write, unchanged = [], []
    for r in records:
        (unchanged if existing.get(r.external_id) == r.content_hash else to_write).append(r)
    return ChangePlan(to_write=to_write, unchanged=unchanged)
