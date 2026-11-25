"""Agent graph tests with a scripted fake model: no API calls, no database."""

import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_graph, seen_jobs
from app.llm import is_transient


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


@tool
def search_jobs(query: str) -> str:
    """Stub search."""
    return json.dumps([{"id": 1, "title": "Backend Intern", "company": "Acme"}])


@tool
def whoami(config: RunnableConfig) -> str:
    """Returns the user id from the run config."""
    return config["configurable"]["user_id"]


def run(graph, text, thread="t1"):
    cfg = {"configurable": {"thread_id": thread, "user_id": "u-123"}}
    return graph.invoke({"messages": [HumanMessage(text)], "final": None,
                         "dropped_job_ids": []}, config=cfg)


def test_grounding_drops_invented_jobs_and_fixes_titles():
    model = ScriptedModel(messages=iter([
        call("search_jobs", {"query": "backend"}, "c1"),
        call("ChatResponse", {"reply": "Here you go", "matches": [
            {"job_id": 1, "title": "Wrong title", "fit_score": 9, "why_it_fits": "Python"},
            {"job_id": 999, "title": "Made up", "fit_score": 10, "why_it_fits": "?"},
        ]}, "c2"),
    ]))
    out = run(build_graph(model, [search_jobs, whoami], InMemorySaver()), "find backend jobs")
    assert out["final"]["reply"] == "Here you go"
    assert [(m["job_id"], m["title"], m["company"]) for m in out["final"]["matches"]] == [
        (1, "Backend Intern", "Acme")
    ]
    assert out["dropped_job_ids"] == [999]
    # The final call got an acknowledgement, so the stored history stays valid.
    assert isinstance(out["messages"][-1], ToolMessage)


def test_memory_across_turns_and_config_injection():
    model = ScriptedModel(messages=iter([
        call("ChatResponse", {"reply": "hi"}, "a1"),
        call("whoami", {}, "b1"),
        call("ChatResponse", {"reply": "done"}, "b2"),
    ]))
    graph = build_graph(model, [search_jobs, whoami], InMemorySaver())
    run(graph, "hello")
    out = run(graph, "who am I?")
    humans = [m.content for m in out["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["hello", "who am I?"]  # first turn remembered
    who = [m for m in out["messages"] if isinstance(m, ToolMessage) and m.name == "whoami"]
    assert who[0].content == "u-123"


def test_plain_text_answer_is_accepted():
    model = ScriptedModel(messages=iter([AIMessage(content="Plain answer")]))
    out = run(build_graph(model, [search_jobs], InMemorySaver()), "hi")
    assert out["final"] == {"reply": "Plain answer", "matches": []}


def test_seen_jobs_only_counts_grounding_tools():
    msgs = [
        ToolMessage(content=json.dumps([{"id": 5, "title": "A"}]), tool_call_id="1",
                    name="search_jobs"),
        ToolMessage(content=json.dumps({"id": 6, "title": "B"}), tool_call_id="2",
                    name="get_job"),
        ToolMessage(content=json.dumps({"saved": 7}), tool_call_id="3", name="save_job"),
        ToolMessage(content="not json", tool_call_id="4", name="search_jobs"),
    ]
    assert set(seen_jobs(msgs)) == {5, 6}


def test_is_transient():
    assert is_transient(RuntimeError("503 UNAVAILABLE. high demand"))
    assert is_transient(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not is_transient(RuntimeError("404 NOT_FOUND"))
