from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class JobMatch(BaseModel):
    """A job the assistant recommends. job_id must come from a tool result in this chat."""

    job_id: int = Field(description="The job's id exactly as returned by search_jobs/get_job")
    title: str
    company: str | None = None
    fit_score: int = Field(ge=1, le=10, description="How well it fits the user, 1-10")
    why_it_fits: str = Field(description="One or two sentences, specific to this user")
    missing_skills: list[str] = Field(
        default_factory=list, description="Skills the job asks for that the user hasn't mentioned"
    )


class ChatResponse(BaseModel):
    """Final answer to the user. Call this exactly once, when you are done using other tools."""

    reply: str = Field(description="What to say to the user, in plain text, concise")
    matches: list[JobMatch] = Field(
        default_factory=list, description="Recommended jobs, best first; empty if none"
    )


class ChatRequest(BaseModel):
    user_id: UUID = Field(description="Anonymous id generated and stored by the client")
    conversation_id: UUID | None = Field(None, description="Omit to start a new conversation")
    message: str = Field(min_length=1, max_length=2000)


class MatchOut(JobMatch):
    """A match plus job details from the database, for display (never from the model)."""

    location: str | None = None
    remote: bool = False
    salary_min: int | None = None
    salary_max: int | None = None
    url: str | None = None
    posted_at: datetime | None = None


class ChatResult(ChatResponse):
    matches: list[MatchOut] = Field(default_factory=list)
    message_id: int | None = Field(None, description="Id of the stored assistant message")
    conversation_id: UUID
    tools_used: list[str] = Field(default_factory=list)
    dropped_job_ids: list[int] = Field(
        default_factory=list, description="Matches removed because no tool returned that job"
    )
