"""Run the SAME agent nodes on the real LangGraph runtime.

`workflow.py` is a tiny local engine with LangGraph's API shape so the project runs and is testable with
no extra dependency. If you have LangGraph installed (`pip install langgraph`), this module registers the
identical node functions and routing on a `langgraph.graph.StateGraph`:

    from controlsguard.agent import ControlsAgent
    from controlsguard.langgraph_app import chat_with_langgraph
    resp = chat_with_langgraph(ControlsAgent(), "Are there control gaps on ledger-core?")

NOTE: this adapter was written against LangGraph's documented StateGraph API but could not be executed in
the environment the project was generated in (no LangGraph available). Run it once and expect to adjust
imports if your LangGraph version differs.
"""
from __future__ import annotations

from typing import Any, TypedDict

from .agent import ControlsAgent, AgentResponse


class AgentState(TypedDict, total=False):
    raw: str
    session: Any
    trace: Any
    registry: Any
    events: list
    status: str
    tool_trace: list
    usage: dict
    t0: float
    query: str
    rq: str
    answer: str
    hits: list
    graph_facts: list
    graph_extra_chunks: list
    ctx: Any
    grounding: dict
    response: AgentResponse
    node_trail: list


def build_langgraph(agent: ControlsAgent):
    from langgraph.graph import END, StateGraph  # imported lazily: optional dependency

    g = StateGraph(AgentState)
    for name, fn in agent.node_functions().items():
        g.add_node(name, fn)
    g.set_entry_point("guard_input")
    g.add_conditional_edges("guard_input", lambda s: "respond" if s["status"] == "blocked" else "retrieve",
                            {"respond": "respond", "retrieve": "retrieve"})
    g.add_conditional_edges("retrieve", lambda s: "respond" if s["status"] == "abstained" else "graph_expand",
                            {"respond": "respond", "graph_expand": "graph_expand"})
    g.add_edge("graph_expand", "build_context")
    g.add_edge("build_context", "act")
    g.add_edge("act", "verify")
    g.add_edge("verify", "respond")
    g.add_edge("respond", END)
    return g.compile()


def chat_with_langgraph(agent: ControlsAgent, message: str, session_id: str | None = None) -> AgentResponse:
    graph = build_langgraph(agent)
    out = graph.invoke(agent.initial_state(message, session_id))
    return out["response"]
