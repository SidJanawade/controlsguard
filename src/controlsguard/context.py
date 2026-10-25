"""Context engineering: assemble system instructions, retrieved knowledge, graph facts, tool
definitions and conversation memory into one token-budgeted prompt.

Priority when the budget is tight (highest first):
  1. system instructions + tool definitions + the question (never dropped)
  2. top-ranked source chunks (dropped from the bottom of the ranking, last one truncated if useful)
  3. graph facts (capped at ~20% of the budget)
  4. conversation memory (capped at ~20% of the budget; older turns summarised)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .ingestion import Chunk
from .textutil import count_tokens


class SourceRegistry:
    """Stable [S#] tags for chunks across the whole request (retrieval, graph and tool results)."""

    def __init__(self) -> None:
        self._tags: dict[str, str] = {}
        self.chunks: dict[str, Chunk] = {}

    def cite(self, chunk: Chunk) -> str:
        if chunk.id not in self._tags:
            self._tags[chunk.id] = f"S{len(self._tags) + 1}"
            self.chunks[self._tags[chunk.id]] = chunk
        return self._tags[chunk.id]

    def get(self, tag: str) -> Chunk | None:
        return self.chunks.get(tag)

    def tags(self) -> list[str]:
        return list(self.chunks)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    words = text.split()
    while words and count_tokens(" ".join(words) + " …") > max_tokens:
        words = words[: int(len(words) * 0.9)]
    return " ".join(words) + " …"


@dataclass
class ContextPackage:
    system: str
    messages: list[dict]
    report: dict = field(default_factory=dict)


class ContextBuilder:
    def __init__(self, budget_tokens: int = 1800):
        self.budget = budget_tokens

    def build(self, *, system_prompt: str, tool_specs: list[dict], query: str, chunks: list[Chunk],
              graph_facts: list[str], memory: list[dict], registry: SourceRegistry) -> ContextPackage:
        fixed = count_tokens(system_prompt) + count_tokens(json.dumps(tool_specs)) + count_tokens(query) + 30
        remaining = max(self.budget - fixed, 0)
        mem_cap, graph_cap = int(remaining * 0.2), int(remaining * 0.2)

        # conversation memory
        mem_msgs: list[dict] = []
        mem_used = 0
        if memory and mem_cap:
            mem_msgs = memory
            mem_used = sum(count_tokens(m["content"]) for m in memory)
            while mem_msgs and mem_used > mem_cap:  # drop oldest pair first
                dropped = mem_msgs[:2]
                mem_msgs = mem_msgs[2:]
                mem_used -= sum(count_tokens(m["content"]) for m in dropped)
        else:
            mem_msgs = []

        # graph facts
        facts_kept: list[str] = []
        graph_used = 0
        for f in graph_facts:
            c = count_tokens(f) + 2
            if graph_used + c > graph_cap:
                break
            facts_kept.append(f)
            graph_used += c

        # sources get everything that is left
        src_budget = max(remaining - mem_used - graph_used, 0)
        src_lines: list[str] = []
        src_used = 0
        included, dropped_chunks = 0, 0
        seen: set[str] = set()
        for ch in chunks:
            if ch.id in seen:
                continue
            seen.add(ch.id)
            tag = registry.cite(ch)
            line = f"[{tag}] ({ch.header})\n{ch.text.split(chr(10), 1)[1] if chr(10) in ch.text else ch.text}"
            cost = count_tokens(line)
            if src_used + cost > src_budget:
                room = src_budget - src_used
                if room >= 60:
                    line = _truncate_to_tokens(line, room)
                    cost = count_tokens(line)
                else:
                    dropped_chunks += 1
                    continue
            src_lines.append(line)
            src_used += cost
            included += 1

        blocks = []
        if facts_kept:
            blocks.append("<graph_facts>\n" + "\n".join(f"- {f}" for f in facts_kept) + "\n</graph_facts>")
        blocks.append("<sources>\n" + ("\n\n".join(src_lines) if src_lines else "(none)") + "\n</sources>")
        user = "\n".join(blocks) + f"\n\nQuestion: {query}"
        messages = list(mem_msgs) + [{"role": "user", "content": user}]
        report = {
            "budget": self.budget,
            "tokens": {"fixed(system+tools+question)": fixed, "memory": mem_used, "graph": graph_used, "sources": src_used},
            "total": fixed + mem_used + graph_used + src_used,
            "sources_included": included,
            "sources_dropped": dropped_chunks,
            "graph_facts_included": len(facts_kept),
            "graph_facts_dropped": len(graph_facts) - len(facts_kept),
        }
        return ContextPackage(system=system_prompt, messages=messages, report=report)
