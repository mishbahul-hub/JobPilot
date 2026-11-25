"""Runs one chat turn: user/conversation bookkeeping, the agent, and chat history rows."""

from uuid import UUID, uuid4

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphRecursionError
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.agent.graph import FINAL, build_graph
from app.agent.schemas import ChatRequest, ChatResult, MatchOut
from app.agent.tools import TOOLS, load_profile
from app.db.pool import get_pool
from app.llm import LLMUnavailable, get_llm_pair, is_transient
from app.log import get_logger

log = get_logger("chat")

MAX_STEPS = 14  # graph steps per turn (each tool round = 2)

_cp_pool: ConnectionPool | None = None
_graph = None


class ConversationNotFound(Exception):
    pass


def open_agent(database_url: str) -> None:
    """Checkpointer pool + compiled graph, built once at startup."""
    global _cp_pool, _graph
    # PostgresSaver needs autocommit + dict rows; prepare_threshold=0 for Neon's pgbouncer.
    _cp_pool = ConnectionPool(
        database_url, min_size=1, max_size=5, open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    saver = PostgresSaver(_cp_pool)
    saver.setup()
    _graph = build_graph(get_llm_pair(), TOOLS, checkpointer=saver, profile_loader=load_profile)
    log.info("agent ready")


def close_agent() -> None:
    global _cp_pool, _graph
    if _cp_pool is not None:
        _cp_pool.close()
    _cp_pool, _graph = None, None


def _prepare(user_id: UUID, conversation_id: UUID | None, message: str) -> UUID:
    with get_pool().connection() as conn:
        conn.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
        if conversation_id is None:
            conversation_id = uuid4()
            conn.execute(
                "INSERT INTO conversations (id, user_id, title) VALUES (%s, %s, %s)",
                (conversation_id, user_id, message[:80]),
            )
        else:
            owner = conn.execute(
                "SELECT user_id FROM conversations WHERE id = %s", (conversation_id,)
            ).fetchone()
            if not owner or owner[0] != user_id:
                raise ConversationNotFound
    return conversation_id


def _tools_used_this_turn(messages) -> list[str]:
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    return [m.name for m in messages[last_human:]
            if isinstance(m, ToolMessage) and m.name != FINAL]


def run_chat(req: ChatRequest) -> ChatResult:
    if _graph is None:
        raise RuntimeError("agent not initialised")
    conv_id = _prepare(req.user_id, req.conversation_id, req.message)
    config = {
        "configurable": {"thread_id": str(conv_id), "user_id": str(req.user_id)},
        "recursion_limit": MAX_STEPS,
    }
    try:
        state = _graph.invoke(
            {"messages": [HumanMessage(req.message)], "final": None, "dropped_job_ids": []},
            config=config,
        )
        final = state.get("final") or {"reply": "Sorry, I couldn't answer that.", "matches": []}
        tools_used = _tools_used_this_turn(state["messages"])
        dropped = state.get("dropped_job_ids") or []
    except Exception as exc:
        if isinstance(exc, GraphRecursionError):
            log.warning("conversation %s hit the step limit", conv_id)
            final = {"reply": "That took too many steps. Could you narrow the request?",
                     "matches": []}
            tools_used, dropped = [], []
        elif is_transient(exc):
            log.warning("all models busy for conversation %s: %s", conv_id, str(exc)[:200])
            raise LLMUnavailable from exc
        else:
            raise
    with get_pool().connection() as conn:
        details = conn.cursor(row_factory=dict_row).execute(
            "SELECT id, location, remote, salary_min, salary_max, url, posted_at "
            "FROM jobs WHERE id = ANY(%s)",
            ([m["job_id"] for m in final.get("matches", [])],),
        ).fetchall()
        by_id = {d.pop("id"): d for d in details}
        matches = [MatchOut(**m, **by_id.get(m["job_id"], {})) for m in final.get("matches", [])]
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', %s)",
            (conv_id, req.message),
        )
        message_id = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, tool_calls) "
            "VALUES (%s, 'assistant', %s, %s) RETURNING id",
            (conv_id, final["reply"], Jsonb({
                "tools_used": tools_used,
                "matches": [m.model_dump(mode="json") for m in matches],
                "dropped_job_ids": dropped,
            })),
        ).fetchone()[0]
    return ChatResult(reply=final["reply"], matches=matches, message_id=message_id,
                      conversation_id=conv_id, tools_used=tools_used, dropped_job_ids=dropped)
