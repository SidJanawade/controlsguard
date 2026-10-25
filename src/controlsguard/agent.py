"""ControlsAgent: the orchestrated, guarded RAG + Graph RAG + tool-using agent.

Workflow (each box is a node function; see `workflow.py`):

    guard_input --blocked--> respond
        |
    retrieve ----abstain---> respond
        |
    graph_expand -> build_context -> act (LLM <-> tools loop) -> verify -> respond
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

from . import guardrails as gr
from .config import Settings
from .context import ContextBuilder, SourceRegistry
from .graph import KnowledgeGraph
from .ingestion import Chunk, ingest
from .llm import LLM, LLMError, make_llm
from .memory import SessionStore
from .observability import Metrics, Trace, audit
from .prompts import (ABSTAIN_MESSAGE, BLOCKED_MESSAGE, CLARIFY_MESSAGE, LEAK_CHECK_TEXT, LOW_CONFIDENCE_MESSAGE,
                      PROMPT_VERSION, SYSTEM_PROMPT)
from .retrieval import HybridRetriever
from .tools import RunContext, default_tools
from .workflow import END, StateGraph

PRONOUNS = {"it", "its", "this", "that", "those", "these", "they", "them", "same"}
# Subset that cannot stand on its own ("that"/"this" appear in ordinary sentences like "controls that protect X")
DANGLING = {"it", "its", "they", "them", "those", "these", "same"}


@dataclass
class AgentResponse:
    answer: str
    status: str  # ok | blocked | abstained | low_confidence | error
    session_id: str
    trace_id: str
    citations: list[dict] = field(default_factory=list)
    graph_facts: list[str] = field(default_factory=list)
    tool_trace: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    context_report: dict = field(default_factory=dict)
    grounding: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    prompt_version: str = PROMPT_VERSION

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class ControlsAgent:
    def __init__(self, settings: Settings | None = None, llm: LLM | None = None):
        self.settings = settings or Settings()
        _, self.chunks = ingest(self.settings.data_dir, self.settings.chunk_max_tokens, self.settings.chunk_overlap_tokens)
        self.retriever = HybridRetriever(self.chunks)
        self.graph = KnowledgeGraph(self.chunks)
        self.llm = llm or make_llm(self.settings)
        self.tools = default_tools()
        self.sessions = SessionStore(self.settings.session_ttl_s)
        self.metrics = Metrics()
        self.context_builder = ContextBuilder(self.settings.context_budget_tokens)
        self.workflow = self._build_workflow()

    # ------------------------------------------------------------------ workflow wiring
    def node_functions(self) -> dict:
        return {
            "guard_input": self.guard_input, "retrieve": self.retrieve, "graph_expand": self.graph_expand,
            "build_context": self.build_context, "act": self.act, "verify": self.verify, "respond": self.respond,
        }

    def _build_workflow(self):
        g = StateGraph()
        for name, fn in self.node_functions().items():
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

    # ------------------------------------------------------------------ public API
    def initial_state(self, message: str, session_id: str | None = None) -> dict:
        return {
            "raw": message, "session": self.sessions.get(session_id), "trace": Trace(), "registry": SourceRegistry(),
            "events": [], "status": "ok", "tool_trace": [], "usage": {"input_tokens": 0, "output_tokens": 0},
            "t0": time.perf_counter(),
        }

    def chat(self, message: str, session_id: str | None = None) -> AgentResponse:
        state = self.initial_state(message, session_id)
        session, trace = state["session"], state["trace"]
        try:
            out = self.workflow.invoke(state)
        except LLMError as e:
            self.metrics.inc("cg_requests_total", status="error")
            audit("error", trace_id=trace.trace_id, error=str(e)[:200])
            return AgentResponse("The language model is currently unavailable. Please try again.", "error",
                                 session.id, trace.trace_id)
        return out["response"]

    # ------------------------------------------------------------------ nodes
    def guard_input(self, s: dict) -> dict:
        raw: str = s["raw"][: self.settings.max_message_chars]
        events = s["events"]
        red = gr.redact_pii(raw)
        if red.findings:
            events.append({"type": "pii_redacted_input", "kinds": sorted(set(red.findings))})
        inj = gr.detect_injection(raw)
        if inj:
            events.append({"type": "prompt_injection_blocked", "patterns": len(inj)})
            return {"query": red.text, "status": "blocked", "answer": BLOCKED_MESSAGE}
        return {"query": red.text}

    def _retrieval_query(self, query: str, session) -> str:
        """Follow-up handling: resolve pronouns against the entity discussed last."""
        if not session.entities or self.graph.resolve(query):
            return query
        words = set(query.lower().replace("?", "").split())
        if words & PRONOUNS or len(words) <= 4:
            return f"{query} (regarding {session.entities[-1]})"
        return query

    def retrieve(self, s: dict) -> dict:
        words = set(s["query"].lower().replace("?", "").split())
        if (words & DANGLING and len(words) <= 8 and not s["session"].entities
                and not self.graph.resolve(s["query"])):
            s["events"].append({"type": "needs_clarification"})
            return {"status": "abstained", "answer": CLARIFY_MESSAGE, "hits": [], "rq": s["query"]}
        with s["trace"].span("retrieve"):
            rq = self._retrieval_query(s["query"], s["session"])
            hits = self.retriever.search(rq, k=self.settings.top_k)
            coverage = self.retriever.query_coverage(rq)
            lexical = any(h.ranks.get("bm25") for h in hits)
        if coverage < 0.4 or not lexical:
            s["events"].append({"type": "abstained_off_topic", "coverage": round(coverage, 2)})
            return {"status": "abstained", "answer": ABSTAIN_MESSAGE, "hits": [], "rq": rq}
        clean: list = []
        for h in hits:
            text, removed = gr.scrub_retrieved(h.chunk.text)
            if removed:
                s["events"].append({"type": "indirect_injection_scrubbed", "chunk": h.chunk.id, "lines": len(removed)})
                h = dataclasses.replace(h, chunk=dataclasses.replace(h.chunk, text=text))
            clean.append(h)
        return {"hits": clean, "rq": rq}

    def graph_expand(self, s: dict) -> dict:
        with s["trace"].span("graph_expand"):
            reg: SourceRegistry = s["registry"]
            for h in s["hits"]:
                reg.cite(h.chunk)
            seeds = self.graph.resolve(s["rq"])
            for h in s["hits"][:2]:
                if h.chunk.entity_id and h.chunk.entity_id.upper() not in seeds:
                    seeds.append(h.chunk.entity_id.upper())
            seeds = [x for x in seeds if self.graph.type_of(x) in ("control", "risk", "system")][:3]
            by_id = {c.id: c for c in self.chunks}
            facts: list[str] = []
            extra: list[Chunk] = []
            hit_ids = {h.chunk.id for h in s["hits"]}
            seen = set()
            for seed in seeds:
                for t in self.graph.neighbors(seed, depth=1, limit=8):
                    if (t.subject, t.relation, t.object) in seen:
                        continue
                    seen.add((t.subject, t.relation, t.object))
                    ch = by_id.get(t.chunk_id)
                    tag = f" [{reg.cite(ch)}]" if ch else ""
                    facts.append(t.text(self.graph) + tag)
                    for nid in (t.subject, t.object):
                        nc = by_id.get(self.graph.g.nodes[nid].get("chunk_id", "")) if nid in self.graph.g else None
                        if nc and nc.id not in hit_ids and nc not in extra and nid != seed and len(extra) < 2:
                            extra.append(nc)
            return {"graph_facts": facts[:12], "graph_extra_chunks": extra}

    def build_context(self, s: dict) -> dict:
        with s["trace"].span("build_context"):
            tool_specs = [t.spec() for t in self.tools.values()]
            chunks = [h.chunk for h in s["hits"]] + s.get("graph_extra_chunks", [])
            pkg = self.context_builder.build(
                system_prompt=SYSTEM_PROMPT, tool_specs=tool_specs, query=s.get("rq", s["query"]), chunks=chunks,
                graph_facts=s.get("graph_facts", []), memory=s["session"].window(int(self.settings.context_budget_tokens * 0.2)),
                registry=s["registry"])
        return {"ctx": pkg}

    def act(self, s: dict) -> dict:
        ctx = s["ctx"]
        run = RunContext(self.retriever, self.graph, s["registry"])
        messages = list(ctx.messages)
        specs = [t.spec() for t in self.tools.values()]
        usage = s["usage"]
        answer = ""
        for step in range(self.settings.max_tool_steps + 1):
            last = step >= self.settings.max_tool_steps
            offer = [] if last else specs
            with s["trace"].span("llm", step=step):
                resp = self.llm.complete(ctx.system, messages, offer)
            usage["input_tokens"] += resp.usage.get("input_tokens", 0)
            usage["output_tokens"] += resp.usage.get("output_tokens", 0)
            if last and resp.tool_calls:  # tools were not offered: never execute calls the model was not allowed to make
                s["events"].append({"type": "tool_budget_exhausted", "ignored_calls": len(resp.tool_calls)})
            if last or not resp.tool_calls:
                answer = resp.text
                break
            messages.append({"role": "assistant", "content": resp.text,
                             "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls]})
            for call in resp.tool_calls[:3]:
                t0 = time.perf_counter()
                tool = self.tools.get(call.name)
                if tool is None:
                    result = f"error: tool '{call.name}' is not available"
                    s["events"].append({"type": "tool_denied", "tool": call.name})
                else:
                    err = gr.validate_tool_args(tool.parameters, call.arguments)
                    if err:
                        result = f"error: {err}"
                        s["events"].append({"type": "tool_args_rejected", "tool": call.name, "reason": err})
                    else:
                        try:
                            result = tool.fn(run, call.arguments)
                        except Exception as e:  # tool failures must not crash the loop
                            result = f"error: tool failed ({type(e).__name__})"
                            s["events"].append({"type": "tool_error", "tool": call.name})
                result, removed = gr.scrub_retrieved(result)
                if removed:
                    s["events"].append({"type": "indirect_injection_scrubbed", "tool": call.name, "lines": len(removed)})
                self.metrics.inc("cg_tool_calls_total", tool=call.name)
                s["tool_trace"].append({"tool": call.name, "arguments": call.arguments,
                                        "result_preview": result[:240], "ms": round((time.perf_counter() - t0) * 1000, 2)})
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})
        return {"answer": answer}

    def verify(self, s: dict) -> dict:
        answer: str = s["answer"]
        if gr.leaks_system_prompt(answer, LEAK_CHECK_TEXT):
            s["events"].append({"type": "system_prompt_leak_blocked"})
            return {"status": "blocked", "answer": BLOCKED_MESSAGE, "grounding": {}}
        red = gr.redact_pii(answer)
        if red.findings:
            s["events"].append({"type": "pii_redacted_output", "kinds": sorted(set(red.findings))})
            answer = red.text
        evidence = {tag: ch.text for tag, ch in s["registry"].chunks.items()}
        g = gr.check_grounding(answer, evidence)
        grounding = {"ok": g.ok, "support": g.support, "cited": g.cited,
                     "invalid_citations": g.invalid_citations, "unsupported_sentences": g.unsupported_sentences}
        if not g.ok:
            s["events"].append({"type": "grounding_failed", "support": g.support})
            return {"status": "low_confidence", "answer": LOW_CONFIDENCE_MESSAGE, "grounding": grounding}
        return {"answer": answer, "grounding": grounding}

    def respond(self, s: dict) -> dict:
        latency = round((time.perf_counter() - s["t0"]) * 1000, 2)
        reg: SourceRegistry = s["registry"]
        cited = s.get("grounding", {}).get("cited", []) if s["status"] == "ok" else []
        citations = [{"tag": t, "chunk_id": reg.get(t).id, "document": reg.get(t).doc_title, "section": reg.get(t).section}
                     for t in cited if reg.get(t)]
        resp = AgentResponse(
            answer=s["answer"], status=s["status"], session_id=s["session"].id, trace_id=s["trace"].trace_id,
            citations=citations, graph_facts=s.get("graph_facts", []), tool_trace=s["tool_trace"], events=s["events"],
            context_report=s["ctx"].report if "ctx" in s else {}, grounding=s.get("grounding", {}),
            usage=s["usage"], latency_ms=latency)
        if s["status"] in ("ok", "abstained", "low_confidence"):
            ents = self.graph.resolve(s.get("rq", s["query"])) or [h.chunk.entity_id.upper() for h in s.get("hits", [])[:1] if h.chunk.entity_id]
            s["session"].add_turn(s["query"], s["answer"], ents[:2])
        self.metrics.inc("cg_requests_total", status=s["status"])
        self.metrics.observe("cg_request_seconds", latency / 1000)
        for e in s["events"]:
            self.metrics.inc("cg_guardrail_events_total", type=e["type"])
        audit("chat", trace_id=resp.trace_id, session=resp.session_id, status=resp.status, query=s["query"][:300],
              latency_ms=latency, tools=[t["tool"] for t in s["tool_trace"]], events=[e["type"] for e in s["events"]],
              tokens=s["usage"], prompt_version=PROMPT_VERSION, llm=getattr(self.llm, "name", "custom"))
        return {"response": resp}
