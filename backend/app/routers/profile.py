from typing import Literal
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import get_settings
from app.db.pool import get_pool
from app.embeddings import EmbeddingUnavailable
from app.llm import is_transient
from app.profile import ensure_user, load_profile, merge_profile, refresh_profile_embedding
from app.ratelimit import limiter
from app.resume.parse import MAX_BYTES, ResumeError, extract_text, parse_resume

router = APIRouter(tags=["profile"])


@router.get("/users/{user_id}/profile")
def get_profile(user_id: UUID):
    with get_pool().connection() as conn:
        return load_profile(conn, user_id) or {}


@router.post(
    "/users/{user_id}/resume",
    dependencies=[Depends(limiter("resume", get_settings().resume_limit_per_hour, 3600))],
)
def upload_resume(user_id: UUID, file: UploadFile):
    data = file.file.read(MAX_BYTES + 1)
    try:
        text = extract_text(data)
    except ResumeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    try:
        parsed = parse_resume(text)
    except Exception as exc:
        if is_transient(exc):
            raise HTTPException(status_code=503, detail="AI model busy, try again soon") from None
        raise
    with get_pool().connection() as conn:
        ensure_user(conn, user_id)
        merge_profile(conn, user_id, skills=parsed.skills, titles=parsed.titles,
                      locations=parsed.locations, years_exp=parsed.years_exp,
                      summary=parsed.summary)
        conn.commit()  # keep the profile even if embedding fails below
        try:
            pending = not refresh_profile_embedding(conn, user_id)
        except EmbeddingUnavailable:
            pending = True  # personalised ranking starts once the embedding is computed
        return {"parsed": parsed, "profile": load_profile(conn, user_id),
                "embedding_pending": pending}


class Feedback(BaseModel):
    user_id: UUID
    job_id: int
    rating: Literal[1, -1]
    message_id: int | None = None


@router.post("/feedback", status_code=201)
def feedback(fb: Feedback):
    with get_pool().connection() as conn:
        ensure_user(conn, fb.user_id)
        try:
            conn.execute(
                "INSERT INTO feedback (user_id, job_id, message_id, rating) VALUES (%s, %s, %s, %s)",
                (fb.user_id, fb.job_id, fb.message_id, fb.rating),
            )
        except psycopg.errors.ForeignKeyViolation:
            raise HTTPException(status_code=404, detail="job or message not found") from None
    return {"ok": True}


