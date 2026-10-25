"""Knowledge graph + Graph RAG helpers.

The graph is built from the structured fields in the policy documents:

    policy --contains--> control --mitigates--> risk --affects--> system
                         control --protects--> system
                         control --related_to--> control
                         control --owned_by--> owner

Every edge keeps `chunk_id` provenance (which document section asserted it) so graph answers stay
traceable back to a citable source.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import networkx as nx

from .ingestion import Chunk

ID_RE = re.compile(r"\b(?:C-[A-Z]{2}-\d{2}|R-[A-Z]+(?:-[A-Z]+)*)\b", re.I)


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    chunk_id: str

    def text(self, g: "KnowledgeGraph") -> str:
        def name(n: str) -> str:
            lab = g.label(n)
            return n if lab == n else f"{n} ({lab})"

        return f"{name(self.subject)} --{self.relation}--> {name(self.object)}"


class KnowledgeGraph:
    def __init__(self, chunks: list[Chunk]):
        self.g = nx.MultiDiGraph()
        self._aliases: dict[str, str] = {}
        self._build(chunks)

    # -- construction ------------------------------------------------------------------------
    def _add_node(self, nid: str, ntype: str, label: str = "", chunk_id: str = "") -> None:
        if nid not in self.g:
            self.g.add_node(nid, type=ntype, label=label or nid, chunk_id=chunk_id)
        else:
            if label and self.g.nodes[nid].get("label") == nid:
                self.g.nodes[nid]["label"] = label
            if chunk_id and not self.g.nodes[nid].get("chunk_id"):
                self.g.nodes[nid]["chunk_id"] = chunk_id

    def _edge(self, s: str, rel: str, o: str, chunk_id: str) -> None:
        self.g.add_edge(s, o, key=rel, relation=rel, chunk_id=chunk_id)

    def _build(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            if not c.entity_id:
                continue
            eid = c.entity_id.upper()
            title = c.section[len(c.entity_id):].strip()
            ntype = "control" if eid.startswith("C-") else "risk"
            self._add_node(eid, ntype, title, c.id)
            self._add_node(c.doc_id, "policy", c.doc_title)
            if ntype == "control":
                self._edge(c.doc_id, "contains", eid, c.id)
            f = c.fields
            for r in f.get("risks", []):
                self._add_node(r.upper(), "risk")
                self._edge(eid, "mitigates", r.upper(), c.id)
            for s in f.get("systems", []):
                self._add_node(s, "system")
                self._edge(eid, "protects", s, c.id)
            for rel in f.get("related", []):
                self._add_node(rel.upper(), "control")
                self._edge(eid, "related_to", rel.upper(), c.id)
            for s in f.get("affects", []):
                self._add_node(s, "system")
                self._edge(eid, "affects", s, c.id)
            if "owner" in f:
                self._add_node(f["owner"], "owner")
                self._edge(eid, "owned_by", f["owner"], c.id)
        for nid, d in self.g.nodes(data=True):
            self._aliases[nid.lower()] = nid
            if d["type"] in ("system", "risk", "control", "owner") and d["label"] != nid:
                self._aliases[d["label"].lower()] = nid

    # -- basic access ------------------------------------------------------------------------
    def label(self, nid: str) -> str:
        return self.g.nodes[nid].get("label", nid) if nid in self.g else nid

    def type_of(self, nid: str) -> str:
        return self.g.nodes[nid]["type"] if nid in self.g else "unknown"

    def nodes_of_type(self, ntype: str) -> list[str]:
        return sorted(n for n, d in self.g.nodes(data=True) if d["type"] == ntype)

    def resolve(self, text: str) -> list[str]:
        """Find graph entities mentioned in free text: explicit ids, system names, risk/control titles."""
        found: list[str] = []
        low = text.lower()
        for m in ID_RE.finditer(text):
            nid = m.group(0).upper()
            if nid in self.g and nid not in found:
                found.append(nid)
        for alias, nid in sorted(self._aliases.items(), key=lambda kv: -len(kv[0])):
            if nid in found or self.type_of(nid) == "policy":
                continue
            if re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", low):
                found.append(nid)
        return found

    # -- traversal ---------------------------------------------------------------------------
    def neighbors(self, nid: str, depth: int = 1, limit: int = 40) -> list[Triple]:
        """Breadth-first expansion over both edge directions. Policy 'contains' edges are skipped
        (they add noise); owners are only followed at depth 1."""
        if nid not in self.g:
            return []
        seen_nodes = {nid}
        frontier = [nid]
        out: list[Triple] = []
        seen_edges: set[tuple] = set()
        for d in range(depth):
            nxt: list[str] = []
            for n in frontier:
                edges = [(u, v, k, a) for u, v, k, a in self.g.out_edges(n, keys=True, data=True)]
                edges += [(u, v, k, a) for u, v, k, a in self.g.in_edges(n, keys=True, data=True)]
                for u, v, k, a in edges:
                    if a["relation"] == "contains":
                        continue
                    other = v if u == n else u
                    if a["relation"] == "owned_by" and d > 0:
                        continue
                    key = (u, a["relation"], v)
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    out.append(Triple(u, a["relation"], v, a["chunk_id"]))
                    if other not in seen_nodes:
                        seen_nodes.add(other)
                        nxt.append(other)
                    if len(out) >= limit:
                        return out
            frontier = nxt
        return out

    def path(self, a: str, b: str) -> list[Triple]:
        if a not in self.g or b not in self.g:
            return []
        u = self.g.to_undirected(as_view=True)
        try:
            nodes = nx.shortest_path(u, a, b)
        except nx.NetworkXNoPath:
            return []
        triples: list[Triple] = []
        for x, y in zip(nodes, nodes[1:]):
            if self.g.has_edge(x, y):
                k, a_ = next(iter(self.g[x][y].items()))
                triples.append(Triple(x, a_["relation"], y, a_["chunk_id"]))
            else:
                k, a_ = next(iter(self.g[y][x].items()))
                triples.append(Triple(y, a_["relation"], x, a_["chunk_id"]))
        return triples

    # -- domain queries ----------------------------------------------------------------------
    def controls_mitigating(self, risk: str) -> list[str]:
        return sorted(u for u, _, a in self.g.in_edges(risk, data=True) if a["relation"] == "mitigates")

    def controls_protecting(self, system: str) -> list[str]:
        return sorted(u for u, _, a in self.g.in_edges(system, data=True) if a["relation"] == "protects")

    def risks_affecting(self, system: str) -> list[str]:
        return sorted(u for u, _, a in self.g.in_edges(system, data=True) if a["relation"] == "affects")

    def coverage(self, system: str) -> dict:
        """For each risk that affects `system`, is there a control that mitigates it AND protects `system`?"""
        protecting = set(self.controls_protecting(system))
        covered, gaps = {}, []
        for risk in self.risks_affecting(system):
            ctrls = sorted(set(self.controls_mitigating(risk)) & protecting)
            if ctrls:
                covered[risk] = ctrls
            else:
                gaps.append(risk)
        return {"system": system, "covered": covered, "gaps": gaps}

    def stats(self) -> dict:
        types: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            types[d["type"]] = types.get(d["type"], 0) + 1
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(), "by_type": types}
