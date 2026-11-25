from datetime import UTC, datetime

from app.ingest.adzuna import to_record
from app.ingest.csv_source import load_csv
from app.ingest.normalize import JobRecord, clean_text, dedupe, detect_remote, plan_changes


def make(ext_id="1", **kw) -> JobRecord:
    base = {
        "source": "test", "external_id": ext_id, "title": "Python Developer", "company": "Acme",
        "location": "Bengaluru", "remote": False, "salary_min": None, "salary_max": None,
        "description": "Build APIs with FastAPI.", "url": None, "posted_at": None,
    }
    return JobRecord(**{**base, **kw})


def test_clean_text_strips_html_and_whitespace():
    assert clean_text("<p>Build&nbsp;<b>APIs</b></p>\n\n with  Python") == "Build APIs with Python"
    assert clean_text(None) == ""


def test_detect_remote():
    assert detect_remote("Remote Backend Engineer", "Pune", "")
    assert detect_remote("Engineer", "Work From Home", "")
    assert detect_remote("Engineer", "Pune", "This is a fully remote role.")
    assert not detect_remote("Engineer", "Pune", "We are near the remote control factory.")


def test_content_hash_changes_only_when_content_changes():
    assert make().content_hash == make().content_hash
    assert make().content_hash != make(description="Different").content_hash
    assert make().content_hash != make(salary_min=500000).content_hash


def test_embed_text_contains_key_fields():
    text = make(remote=True).embed_text
    assert "Python Developer" in text and "Acme" in text and "(remote)" in text


def test_dedupe_keeps_last():
    out = dedupe([make("1", title="Old"), make("2"), make("1", title="New")])
    assert len(out) == 2 and {r.title for r in out} == {"New", "Python Developer"}


def test_plan_changes():
    a, b, c = make("a"), make("b"), make("c")
    existing = {"a": a.content_hash, "b": "stale-hash"}
    plan = plan_changes([a, b, c], existing)
    assert [r.external_id for r in plan.unchanged] == ["a"]
    assert sorted(r.external_id for r in plan.to_write) == ["b", "c"]


def test_adzuna_to_record():
    raw = {
        "id": 4242,
        "title": "<strong>ML</strong> Engineer",
        "description": "Work from home. Train models&hellip;",
        "company": {"display_name": "Acme AI"},
        "location": {"display_name": "Bengaluru, Karnataka"},
        "salary_min": 1200000.0,
        "salary_max": 1800000.0,
        "salary_is_predicted": "0",
        "created": "2026-11-20T10:00:00Z",
        "redirect_url": "https://example.com/job",
    }
    r = to_record(raw)
    assert r.external_id == "4242" and r.title == "ML Engineer"
    assert r.remote and r.salary_min == 1200000
    assert r.posted_at == datetime(2026, 11, 20, 10, tzinfo=UTC)

    predicted = to_record({**raw, "salary_is_predicted": "1"})
    assert predicted.salary_min is None and predicted.salary_max is None
    assert to_record({"id": 1, "title": ""}) is None


def test_load_csv_kaggle_columns(tmp_path):
    path = tmp_path / "postings.csv"
    path.write_text(
        "job_id,title,company_name,location,description,job_posting_url,listed_time,"
        "min_salary,max_salary,remote_allowed\n"
        "9,Data Engineer,Beta,Gurugram,Build pipelines,https://x.y,1732000000000,,,1.0\n"
        ",No id,Beta,Gurugram,skip me,,,,,\n",
        encoding="utf-8",
    )
    records = load_csv(path, source="kaggle")
    assert len(records) == 1
    r = records[0]
    assert (r.source, r.external_id, r.company, r.remote) == ("kaggle", "9", "Beta", True)
    assert r.posted_at.year == 2024
