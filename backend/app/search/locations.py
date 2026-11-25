"""Indian city name aliases: users write 'Bengaluru', job boards often write 'Bangalore'."""

import re

ALIASES: list[set[str]] = [
    {"bengaluru", "bangalore", "blr"},
    {"gurugram", "gurgaon"},
    {"mumbai", "bombay", "navi mumbai", "thane"},
    {"delhi", "new delhi", "delhi ncr", "ncr"},
    {"noida", "greater noida"},
    {"chennai", "madras"},
    {"kolkata", "calcutta"},
    {"pune", "poona"},
    {"hyderabad", "secunderabad"},
    {"thiruvananthapuram", "trivandrum"},
    {"kochi", "cochin"},
    {"mysuru", "mysore"},
]

# "NCR" covers several cities.
REGIONS = {
    "ncr": {"delhi", "gurgaon", "gurugram", "noida", "ghaziabad", "faridabad"},
    "delhi ncr": {"delhi", "gurgaon", "gurugram", "noida", "ghaziabad", "faridabad"},
}


_ALL_NAMES = sorted({n for g in ALIASES for n in g} | set(REGIONS), key=len, reverse=True)
_CITY_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in _ALL_NAMES) + r")\b", re.IGNORECASE)


def find_city(text: str) -> str | None:
    """First known city/region mentioned in free text ('ML jobs in Gurugram' -> 'gurugram')."""
    m = _CITY_RE.search(text)
    return m.group(1).lower() if m else None


def aliases_of(name: str) -> set[str]:
    key = name.lower()
    return next((g for g in ALIASES if key in g), {key})


def location_patterns(location: str) -> list[str]:
    """ILIKE patterns matching the location and its aliases."""
    key = " ".join(location.lower().replace(",", " ").split())
    names = {key}
    for group in ALIASES:
        if key in group:
            names |= group
    names |= REGIONS.get(key, set())
    # 'ncr' and 'blr' are too short to substring-match safely.
    names -= {"ncr", "blr"} if len(names) > 1 else set()
    return sorted(f"%{n}%" for n in names)
