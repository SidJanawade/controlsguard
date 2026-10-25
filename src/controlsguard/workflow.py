"""A minimal state-graph engine with the same shape as LangGraph's StateGraph API.

Nodes are plain functions `state -> partial state update`. Because the agent's nodes only depend on this
contract, `langgraph_app.py` can register the very same functions with the real LangGraph runtime.
This local engine exists so the project runs (and is unit-tested) without that dependency.
"""
from __future__ import annotations

import time
from typing import Callable

END = "__end__"
State = dict
Node = Callable[[State], State]


class StateGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, str] = {}
        self.conditional: dict[str, tuple[Callable[[State], str], dict[str, str]]] = {}
        self.entry: str | None = None

    def add_node(self, name: str, fn: Node) -> None:
        self.nodes[name] = fn

    def set_entry_point(self, name: str) -> None:
        self.entry = name

    def add_edge(self, src: str, dst: str) -> None:
        self.edges[src] = dst

    def add_conditional_edges(self, src: str, router: Callable[[State], str], mapping: dict[str, str]) -> None:
        self.conditional[src] = (router, mapping)

    def compile(self) -> "CompiledGraph":
        assert self.entry, "entry point not set"
        for n in list(self.edges.values()) + [v for _, m in self.conditional.values() for v in m.values()]:
            assert n == END or n in self.nodes, f"unknown node {n}"
        return CompiledGraph(self)


class CompiledGraph:
    MAX_STEPS = 50

    def __init__(self, g: StateGraph):
        self.g = g

    def invoke(self, state: State) -> State:
        state = dict(state)
        trail: list[dict] = state.setdefault("node_trail", [])
        current = self.g.entry
        for _ in range(self.MAX_STEPS):
            if current == END:
                return state
            t0 = time.perf_counter()
            update = self.g.nodes[current](state) or {}
            state.update(update)
            trail.append({"node": current, "ms": round((time.perf_counter() - t0) * 1000, 2)})
            if current in self.g.conditional:
                router, mapping = self.g.conditional[current]
                current = mapping[router(state)]
            else:
                current = self.g.edges.get(current, END)
        raise RuntimeError("workflow exceeded max steps (cycle?)")
