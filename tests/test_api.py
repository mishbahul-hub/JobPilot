"""API tests that need no database or LLM: validation, error mapping, rate limits.
The client is created without the lifespan, so nothing connects to Postgres or Gemini."""

import uuid

import pytest
from fastapi.testclient import TestClient

from app import ratelimit
from app.agent import service
from app.agent.schemas import ChatResult
from app.llm import LLMUnavailable
from app.main import app

client = TestClient(app)
USER = str(uuid.uuid4())


@pytest.fixture(autouse=True)
def _reset_limits():
    ratelimit.reset()
    yield
    ratelimit.reset()


def test_search_validates_query():
    assert client.get("/jobs/search", params={"q": "a"}).status_code == 422
    assert client.get("/jobs/search", params={"q": "python", "limit": 99}).status_code == 422


def test_chat_validates_body():
    assert client.post("/chat", json={"user_id": "not-a-uuid", "message": "hi"}).status_code == 422
    assert client.post("/chat", json={"user_id": USER, "message": ""}).status_code == 422


def test_chat_maps_busy_model_to_503(monkeypatch):
    def busy(req):
        raise LLMUnavailable

    monkeypatch.setattr("app.routers.chat.run_chat", busy)
    r = client.post("/chat", json={"user_id": USER, "message": "hi"})
    assert r.status_code == 503 and r.headers["retry-after"] == "30"


def test_chat_maps_foreign_conversation_to_404(monkeypatch):
    def not_found(req):
        raise service.ConversationNotFound

    monkeypatch.setattr("app.routers.chat.run_chat", not_found)
    r = client.post("/chat", json={"user_id": USER, "conversation_id": USER, "message": "hi"})
    assert r.status_code == 404


def test_chat_rate_limit(monkeypatch):
    monkeypatch.setattr(
        "app.routers.chat.run_chat",
        lambda req: ChatResult(reply="ok", conversation_id=uuid.uuid4()),
    )
    codes = [client.post("/chat", json={"user_id": USER, "message": "hi"}).status_code
             for _ in range(31)]
    assert codes[:30] == [200] * 30 and codes[30] == 429


def test_resume_rejects_non_pdf_without_touching_db():
    r = client.post(f"/users/{USER}/resume",
                    files={"file": ("cv.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 422 and "not a PDF" in r.json()["detail"]


def test_ratelimit_window_slides(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(ratelimit.time, "monotonic", lambda: t[0])
    ratelimit.check("k", 2, 60)
    ratelimit.check("k", 2, 60)
    with pytest.raises(Exception, match="Too many"):
        ratelimit.check("k", 2, 60)
    t[0] += 61
    ratelimit.check("k", 2, 60)  # window passed
