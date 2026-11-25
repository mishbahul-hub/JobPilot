import math

import pytest

from app.search.hybrid import build_filters, dedupe_hits, or_tsquery, rrf_fuse, understand_query
from app.search.locations import find_city, location_patterns
from app.search.metrics import ndcg_at_k, precision_at_k, reciprocal_rank
from app.search.models import JobHit, SearchFilters


def test_rrf_rewards_agreement():
    fused = dict(rrf_fuse([1, 2, 3], [3, 4]))
    assert fused[3] > fused[1] > fused[2]  # in both lists beats top of one list
    assert math.isclose(fused[1], 1 / 61)


def test_weighted_rrf():
    fused = dict(rrf_fuse([1], [2], weights=[1.0, 0.5]))
    assert fused[1] == pytest.approx(2 * fused[2])


def test_or_tsquery_drops_filler_and_expands_cities():
    assert or_tsquery("Python jobs in Bengaluru!") == "python | in | bangalore | bengaluru | blr"
    assert or_tsquery("jobs") is None
    assert or_tsquery("c++ / node.js") == "c | node | js"  # only safe tsquery tokens


def test_location_aliases():
    assert "%bangalore%" in location_patterns("Bengaluru")
    assert "%gurgaon%" in location_patterns("gurugram")
    assert "%noida%" in location_patterns("Delhi NCR")
    assert location_patterns("Indore") == ["%indore%"]
    assert find_city("ML jobs in Gurugram please") == "gurugram"
    assert find_city("remote only") is None


def test_understand_query_sets_location_unless_given():
    f = understand_query("backend jobs in Pune", SearchFilters())
    assert f.location == "pune" and f.remote_ok
    assert understand_query("backend in Pune", SearchFilters(location="Chennai")).location == "Chennai"
    assert understand_query("backend", SearchFilters()).location is None


def test_build_filters():
    sql, params = build_filters(
        SearchFilters(location="Pune", remote_ok=True, min_salary=10, exclude_ids=[1])
    )
    assert "OR remote" in sql and "salary_max IS NULL" in sql and "<> ALL" in sql
    assert params["min_salary"] == 10 and params["exclude"] == [1]
    assert build_filters(SearchFilters()) == ("", {})


def _hit(i, title="Dev", company="A", location="Pune"):
    return JobHit(id=i, title=title, company=company, location=location, remote=False,
                  salary_min=None, salary_max=None, url=None, posted_at=None, snippet="", score=1)


def test_dedupe_hits_keeps_first_copy():
    out = dedupe_hits([_hit(1), _hit(2, title="dev "), _hit(3, company="B")])
    assert [h.id for h in out] == [1, 3]


def test_metrics():
    rel = {2, 4}
    assert precision_at_k([1, 2, 3, 4], rel, k=4) == 0.5
    assert reciprocal_rank([1, 2], rel) == 0.5 and reciprocal_rank([1], rel) == 0
    assert ndcg_at_k([2, 4, 1], rel) == pytest.approx(1.0)
    assert 0 < ndcg_at_k([1, 2, 4], rel) < 1
