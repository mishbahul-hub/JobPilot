"""The JobPilot agent as a small LangGraph state machine.

    agent ──tool calls──▶ tools ──▶ agent ...
      └──ChatResponse──▶ finalize ──▶ END

The model must call a tool every step (tool_choice="any"); its final answer is a call to the
ChatResponse "tool", which gives a validated structured answer without a second LLM call.
`finalize` then drops any recommended job that no tool returned in this conversation, so a
hallucinated job id can never reach the user.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from app.agent.prompts import system_prompt
from app.agent.schemas import ChatResponse
from app.log import get_logger

log = get_logger("agent")

FINAL = ChatResponse.__name__
MAX_HISTORY_TOKENS = 12_000  # approx (chars / 4); older turns are trimmed, not summarised
GROUNDING_TOOLS = {"search_jobs", "get_job", "list_saved_jobs"}


class AgentState(MessagesState):
    final: dict[str, Any] | None
    dropped_job_ids: list[int]


def _approx_tokens(messages) -> int:
    return sum(len(str(m.content)) + len(json.dumps(getattr(m, "tool_calls", []) or [])) for m in
               messages) // 4


def seen_jobs(messages) -> dict[int, dict]:
    """Jobs (id -> tool data) that appeared in grounding tool results in the conversation."""
    jobs: dict[int, dict] = {}

    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("id"), int) and "title" in x:
                jobs[x["id"]] = x
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for m in messages:
        if isinstance(m, ToolMessage) and m.name in GROUNDING_TOOLS:
            try:
                walk(json.loads(m.content))
            except (TypeError, json.JSONDecodeError):
                continue
    return jobs


def build_graph(models, tools, checkpointer=None, profile_loader=None):
    """`models`: chat models in priority order (main, then fallbacks). Tools and the
    ChatResponse schema are bound to each; a failing model falls through to the next.
    `profile_loader(user_id) -> dict | None` puts the user's profile in the system prompt, so
    the model doesn't spend a tool round fetching it."""
    models = models if isinstance(models, list) else [models]
    bound_all = [m.bind_tools([*tools, ChatResponse], tool_choice="any") for m in models]
    bound = bound_all[0].with_fallbacks(bound_all[1:]) if len(bound_all) > 1 else bound_all[0]

    def agent(state: AgentState, config: RunnableConfig):
        history = trim_messages(
            state["messages"],
            max_tokens=MAX_HISTORY_TOKENS,
            token_counter=_approx_tokens,
            strategy="last",
            start_on="human",
            include_system=False,
        )
        user_id = config.get("configurable", {}).get("user_id")
        profile = profile_loader(user_id) if profile_loader and user_id else None
        ai = bound.invoke([SystemMessage(system_prompt(profile)), *history])
        return {"messages": [ai]}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            return "finalize"  # model answered in plain text despite tool_choice
        if any(c["name"] == FINAL for c in calls):
            return "finalize"
        return "tools"

    def finalize(state: AgentState):
        last: AIMessage = state["messages"][-1]
        calls = last.tool_calls or []
        final_call = next((c for c in calls if c["name"] == FINAL), None)
        try:
            resp = (ChatResponse.model_validate(final_call["args"]) if final_call
                    else ChatResponse(reply=str(last.content) or "Sorry, something went wrong."))
        except ValidationError as exc:
            log.warning("invalid ChatResponse args: %s", exc)
            resp = ChatResponse(reply=final_call["args"].get("reply") or "Sorry, please try again.")

        seen = seen_jobs(state["messages"])
        # Keep only grounded jobs, and take title/company from the data, not the model.
        kept = [
            m.model_copy(update={"title": seen[m.job_id]["title"],
                                 "company": seen[m.job_id].get("company")})
            for m in resp.matches if m.job_id in seen
        ]
        dropped = [m.job_id for m in resp.matches if m.job_id not in seen]
        if dropped:
            log.warning("dropped ungrounded job ids %s", dropped)
        resp = resp.model_copy(update={"matches": kept})

        # Every tool call needs a reply message, or the next turn's history is invalid.
        acks = [
            ToolMessage(
                content="delivered to user" if c["name"] == FINAL else "skipped: answer was final",
                tool_call_id=c["id"], name=c["name"],
            )
            for c in calls
        ]
        return {"messages": acks, "final": resp.model_dump(), "dropped_job_ids": dropped}

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_node("finalize", finalize)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
