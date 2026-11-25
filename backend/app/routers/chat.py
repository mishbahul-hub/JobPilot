from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from psycopg.rows import dict_row
from pydantic import BaseModel

from app.agent.schemas import ChatRequest, ChatResult
from app.agent.service import ConversationNotFound, run_chat
from app.config import get_settings
from app.db.pool import get_pool
from app.llm import LLMUnavailable
from app.profile import ensure_user
from app.ratelimit import limiter

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResult,
    dependencies=[Depends(limiter("chat", get_settings().chat_limit_per_10min, 600))],
)
def chat(req: ChatRequest):
    try:
        return run_chat(req)
    except ConversationNotFound:
        raise HTTPException(status_code=404, detail="conversation not found") from None
    except LLMUnavailable:
        raise HTTPException(
            status_code=503,
            detail="The AI model is busy right now. Please try again in a minute.",
            headers={"Retry-After": "30"},
        ) from None


@router.get("/users/{user_id}/conversations")
def list_conversations(user_id: UUID):
    with get_pool().connection() as conn:
        return conn.cursor(row_factory=dict_row).execute(
            "SELECT id, title, created_at FROM conversations WHERE user_id = %s "
            "ORDER BY created_at DESC LIMIT 50",
            (user_id,),
        ).fetchall()


@router.get("/users/{user_id}/conversations/{conversation_id}/messages")
def conversation_messages(user_id: UUID, conversation_id: UUID):
    with get_pool().connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        owner = cur.execute(
            "SELECT user_id FROM conversations WHERE id = %s", (conversation_id,)
        ).fetchone()
        if not owner or owner["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="conversation not found")
        return cur.execute(
            "SELECT id, role, content, tool_calls -> 'matches' AS matches, created_at "
            "FROM messages WHERE conversation_id = %s ORDER BY created_at, id",
            (conversation_id,),
        ).fetchall()


class SaveJob(BaseModel):
    job_id: int


@router.post("/users/{user_id}/saved", status_code=201)
def save_job(user_id: UUID, body: SaveJob):
    with get_pool().connection() as conn:
        ensure_user(conn, user_id)
        try:
            conn.execute(
                "INSERT INTO saved_jobs (user_id, job_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (user_id, body.job_id),
            )
        except psycopg.errors.ForeignKeyViolation:
            raise HTTPException(status_code=404, detail="job not found") from None
    return {"saved": body.job_id}


@router.delete("/users/{user_id}/saved/{job_id}", status_code=204)
def unsave_job(user_id: UUID, job_id: int):
    with get_pool().connection() as conn:
        conn.execute("DELETE FROM saved_jobs WHERE user_id = %s AND job_id = %s",
                     (user_id, job_id))


@router.get("/users/{user_id}/saved")
def saved_jobs(user_id: UUID):
    with get_pool().connection() as conn:
        return conn.cursor(row_factory=dict_row).execute(
            "SELECT j.id, j.title, j.company, j.location, j.remote, j.salary_min, j.salary_max, "
            "j.url, j.is_active, s.created_at AS saved_at "
            "FROM saved_jobs s JOIN jobs j ON j.id = s.job_id WHERE s.user_id = %s "
            "ORDER BY s.created_at DESC",
            (user_id,),
        ).fetchall()
