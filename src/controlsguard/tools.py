"""Agent tools (function calling). Each tool is a plain function with a JSON-schema contract.

The same registry backs the in-process agent and the MCP server, so there is one definition of what the
agent can do. Tool output is plain text with [S#] source tags so answers built from tools stay citable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .context import SourceRegistry
from .graph import KnowledgeGraph
from .retrieval import HybridRetriever


@dataclass
class RunContext:
    retriever: HybridRetriever
    graph: KnowledgeGraph
    registry: SourceRegistry = field(default_factory=SourceRegistry)

    def chunk_by_id(self, chunk_id: str):
        return next((c for c in self.retriever.chunks if c.id == chunk_id), None)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[[RunContext, dict], str]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _body(chunk) -> str:
    return chunk.text.split("\n", 1)[1] if "\n" in chunk.text else chunk.text


def _cite(ctx: RunContext, chunk_id: str) -> str:
    ch = ctx.chunk_by_id(chunk_id)
    return f"[{ctx.registry.cite(ch)}]" if ch else ""


def search_policies(ctx: RunContext, args: dict) -> str:
    k = max(1, min(int(args.get("k", 4)), 8))
    hits = ctx.retriever.search(args["query"], k=k)
    if not hits:
        return "No matching policy text found."
    return "\n\n".join(f"[{ctx.registry.cite(h.chunk)}] ({h.chunk.header})\n{_body(h.chunk)}" for h in hits)


def get_control(ctx: RunContext, args: dict) -> str:
    chunk = ctx.retriever.by_entity(args["control_id"].strip())
    if not chunk:
        return f"Unknown id '{args['control_id']}'. Valid ids look like C-AC-01 (control) or R-UNAUTH (risk)."
    return f"[{ctx.registry.cite(chunk)}] ({chunk.header})\n{_body(chunk)}"


def graph_neighbors(ctx: RunContext, args: dict) -> str:
    depth = max(1, min(int(args.get("depth", 1)), 2))
    entities = [e for e in ctx.graph.resolve(args["entity"]) if ctx.graph.type_of(e) != "owner"][:2]
    if not entities:
        return "No known control, risk or system was recognised in that text."
    phrases = {
        ("protects", "in"): "is protected by", ("protects", "out"): "protects",
        ("mitigates", "in"): "is mitigated by", ("mitigates", "out"): "mitigates",
        ("affects", "in"): "is affected by", ("affects", "out"): "affects",
        ("related_to", "in"): "is related to", ("related_to", "out"): "is related to",
        ("owned_by", "out"): "is owned by",
    }
    lines: list[str] = []
    for e in entities:
        # one grouped line per relation so long lists are complete and compact (never silently truncated)
        groups: dict[tuple[str, str], list] = {}
        deeper = []
        for t in ctx.graph.neighbors(e, depth=depth, limit=60):
            if t.object == e:
                groups.setdefault((t.relation, "in"), []).append((t.subject, t))
            elif t.subject == e:
                groups.setdefault((t.relation, "out"), []).append((t.object, t))
            else:
                deeper.append(t)
        for (rel, direction), items in groups.items():
            names = sorted({n for n, _ in items})
            cites = []
            for _, t in items:
                c = _cite(ctx, t.chunk_id)
                if c and c not in cites:
                    cites.append(c)
            lines.append(f"FACT: {e} {phrases.get((rel, direction), rel)} {', '.join(names)} {' '.join(cites)}.")
        for t in deeper[:6]:
            lines.append(f"FACT: {t.text(ctx.graph)} {_cite(ctx, t.chunk_id)}.")
    return "\n".join(lines) or "No relationships found."


def assess_coverage(ctx: RunContext, args: dict) -> str:
    systems = [e for e in ctx.graph.resolve(args["system"]) if ctx.graph.type_of(e) == "system"]
    if not systems:
        known = ", ".join(ctx.graph.nodes_of_type("system"))
        return f"No known system recognised. Known systems: {known}."
    gaps: list[str] = []
    covered: list[str] = []
    clean: list[str] = []
    for s in systems[:2]:
        cov = ctx.graph.coverage(s)
        if not cov["gaps"]:  # a "nothing wrong" conclusion still needs sources: the risks that were checked
            cites = " ".join(_cite(ctx, ctx.graph.g.nodes[r].get("chunk_id", "")) for r in cov["covered"])
            clean.append(f"COVERED: no coverage gaps found for {s}; every risk affecting it ({', '.join(cov['covered'])}) "
                         f"is mitigated by a control that protects it {cites}.")
        for risk in cov["gaps"]:
            rc = _cite(ctx, ctx.graph.g.nodes[risk].get("chunk_id", ""))
            gaps.append(f"GAP: risk {risk} affects {s}, but no control that mitigates {risk} protects {s} {rc}.")
        for risk, ctrls in cov["covered"].items():
            rc = _cite(ctx, ctx.graph.g.nodes[risk].get("chunk_id", ""))
            cc = " ".join(_cite(ctx, ctx.graph.g.nodes[c].get("chunk_id", "")) for c in ctrls)
            covered.append(f"COVERED: risk {risk} on {s} is mitigated by {', '.join(ctrls)} {rc} {cc}.".replace("  ", " "))
    # gaps first: they are the finding, and downstream truncation must never drop them.
    # Systems with no gaps get one summary line instead of one line per risk.
    return "\n".join(gaps + clean + [c for c in covered if not any(f" on {s} " in c for s in _clean_systems(clean))])


def _clean_systems(clean_lines: list[str]) -> list[str]:
    return [l.split("found for ")[1].split(";")[0] for l in clean_lines]


def default_tools() -> dict[str, Tool]:
    tools = [
        Tool("search_policies", "Semantic + keyword search over the policy corpus. Use for questions the current sources do not answer.",
             {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}, "required": ["query"]},
             search_policies),
        Tool("get_control", "Fetch the full text of one control (e.g. C-AC-01) or risk (e.g. R-UNAUTH) by id.",
             {"type": "object", "properties": {"control_id": {"type": "string"}}, "required": ["control_id"]},
             get_control),
        Tool("graph_neighbors", "Knowledge-graph lookup: relationships of a control, risk or system (mitigates, protects, affects, related_to, owned_by).",
             {"type": "object", "properties": {"entity": {"type": "string"}, "depth": {"type": "integer"}}, "required": ["entity"]},
             graph_neighbors),
        Tool("assess_coverage", "Control coverage check for a system: which risks that affect it are mitigated by a control protecting it, and which are gaps.",
             {"type": "object", "properties": {"system": {"type": "string"}}, "required": ["system"]},
             assess_coverage),
    ]
    return {t.name: t for t in tools}
