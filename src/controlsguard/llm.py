"""LLM abstraction.

All providers implement `complete(system, messages, tools) -> LLMResponse` using one internal message
format, so the agent never depends on a vendor SDK:

    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "arguments"}]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

* MockLLM            deterministic and offline. Plans tool calls with simple rules and writes extractive,
                     cited answers. It exists so the whole pipeline (retrieval, graph, tools, guardrails,
                     evals, CI) is runnable and testable without API keys. It is NOT a language model.
* AnthropicLLM       Claude via the Messages API (tool use).
* OpenAICompatLLM    OpenAI Chat Completions and compatible endpoints (e.g. Gemini's OpenAI-compatible URL).
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .textutil import content_tokens, count_tokens, sentences

Transport = Callable[[str, dict, dict], dict]


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


# =============================================================================================
# Mock
# =============================================================================================
_SRC_BLOCK = re.compile(r"\[(S\d+)\] \((.*?)\)\n(.*?)(?=\n\n\[S\d+\] \(|\n</sources>|\Z)", re.S)
_PRECITED = re.compile(r"^(GAP|COVERED|FACT):\s.*\[S\d+\]\.?$")
_CTRL_ID = re.compile(r"\bC-[A-Z]{2}-\d{2}\b", re.I)
_RELATIONAL = re.compile(
    r"\b(impact|impacts|affect|affects|affected|depend|depends|related|relate|linked|connected|mitigate|mitigates|"
    r"protect|protects|protected|which controls|what controls|blast radius|downstream)\b", re.I)
_COVERAGE = re.compile(r"\b(gap|gaps|coverage|covered|uncovered|not covered|missing control)\b", re.I)


class MockLLM:
    name = "mock"

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        question, sources, headers = self._parse(messages)
        has_tool_results = any(m["role"] == "tool" for m in messages)
        names = {t["name"] for t in tools}
        usage = {"input_tokens": sum(count_tokens(m.get("content", "")) for m in messages) + count_tokens(system)}
        if not has_tool_results and names:
            calls = self._plan(question, names)
            if calls:
                return LLMResponse(tool_calls=calls, usage=usage)
        tool_text = "\n\n".join(m["content"] for m in messages if m["role"] == "tool")
        text = self._synthesize(question, sources, headers, tool_text)
        usage["output_tokens"] = count_tokens(text)
        return LLMResponse(text=text, usage=usage)

    # -- planning ----------------------------------------------------------------------------
    def _plan(self, question: str, names: set[str]) -> list[ToolCall]:
        if _COVERAGE.search(question) and "assess_coverage" in names:
            return [ToolCall("call_1", "assess_coverage", {"system": question})]
        if _RELATIONAL.search(question) and "graph_neighbors" in names:
            return [ToolCall("call_1", "graph_neighbors", {"entity": question, "depth": 1})]
        m = _CTRL_ID.search(question)
        if m and "get_control" in names:
            return [ToolCall("call_1", "get_control", {"control_id": m.group(0).upper()})]
        return []

    # -- parsing -----------------------------------------------------------------------------
    @staticmethod
    def _parse(messages: list[dict]) -> tuple[str, dict[str, str], dict[str, str]]:
        ctx = next((m["content"] for m in reversed(messages) if m["role"] == "user" and "<sources>" in m["content"]), "")
        qm = re.search(r"Question:\s*(.*)\Z", ctx, re.S)
        question = qm.group(1).strip() if qm else ""
        blocks = _SRC_BLOCK.findall(ctx)
        return question, {t: body for t, _, body in blocks}, {t: h for t, h, _ in blocks}

    # -- answer writing ----------------------------------------------------------------------
    def _synthesize(self, question: str, sources: dict[str, str], headers: dict[str, str], tool_text: str) -> str:
        sources, headers = dict(sources), dict(headers)
        pre: list[str] = []
        for line in tool_text.splitlines():
            line = line.strip()
            if _PRECITED.match(line):
                pre.append(re.sub(r"^(GAP|COVERED|FACT):\s*", "", line))
        for t, h, body in _SRC_BLOCK.findall(tool_text):
            if t not in sources:
                sources[t], headers[t] = body, h

        qt = content_tokens(question)
        if re.search(r"\bwho\b", question, re.I) and re.search(r"\b(own|owns|owner|responsible|accountable)\b", question, re.I):
            qt.add("owner")
        asked_ids = {m.upper() for m in re.findall(r"\b(?:C-[A-Z]{2}-\d{2}|R-[A-Z]+(?:-[A-Z]+)*)\b", question, re.I)}
        cands: list[tuple[float, int, str, str]] = []
        seen_text: set[str] = set()
        targeted = {t for t in sources if any(i in headers.get(t, "").upper() for i in asked_ids)}
        for idx, (tag, body) in enumerate(sources.items()):
            if targeted and tag not in targeted:  # the user named a specific control/risk: answer from it
                continue
            about_asked = tag in targeted
            for sent in sentences(body):
                clean = re.sub(r"^-\s*", "", sent).strip()
                if len(clean) < 12 or clean.lower() in seen_text:
                    continue
                overlap = len(qt & content_tokens(clean))
                if overlap:
                    seen_text.add(clean.lower())
                    cands.append((overlap + (3 if about_asked else 0) + 0.2 / (idx + 1), idx, tag, clean))
        cands.sort(key=lambda c: -c[0])
        if cands:  # drop weak matches: keep only sentences scoring at least half of the best one
            cands = [c for c in cands if c[0] >= cands[0][0] * 0.5]
        picked: list[tuple[int, str, str]] = []
        per_source: dict[str, int] = {}
        for _, idx, tag, sent in cands:
            if per_source.get(tag, 0) >= 2 or len(picked) >= 3:
                continue
            per_source[tag] = per_source.get(tag, 0) + 1
            picked.append((idx, tag, sent))
        picked.sort(key=lambda p: p[0])

        if pre:  # tool findings (graph facts / coverage) are the answer; don't pad with loosely related text
            lines = list(pre[:8])
        else:
            lines = [f"{sent.rstrip('.')} [{tag}]." for _, tag, sent in picked]
        if not lines:
            return "Answer: The provided sources do not contain this information.\nEvidence: none\nGaps: the question is not answered by the sources."
        tags = sorted({t for ln in lines for t in re.findall(r"\[(S\d+)\]", ln)}, key=lambda s: int(s[1:]))
        return "Answer: " + " ".join(ln if ln.endswith(".") else ln + "." for ln in lines) + \
            f"\nEvidence: {', '.join(tags)}\nGaps: none identified in the provided sources."


# =============================================================================================
# HTTP providers
# =============================================================================================
def http_post(url: str, headers: dict, body: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise LLMError(f"HTTP {e.code}: {e.read().decode()[:300]}") from None
    except urllib.error.URLError as e:
        raise LLMError(f"network error: {e.reason}") from None


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.anthropic.com",
                 max_tokens: int = 1024, transport: Transport = http_post):
        if not model or not api_key:
            raise LLMError("anthropic provider needs CG_LLM_MODEL and CG_LLM_API_KEY")
        self.model, self.api_key, self.base_url = model, api_key, base_url.rstrip("/")
        self.max_tokens, self.transport = max_tokens, transport

    @staticmethod
    def convert_messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                        and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                           for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                "messages": self.convert_messages(messages)}
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                             for t in tools]
        data = self.transport(f"{self.base_url}/v1/messages",
                              {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                               "content-type": "application/json"}, body)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        calls = [ToolCall(b["id"], b["name"], b.get("input", {})) for b in data.get("content", [])
                 if b.get("type") == "tool_use"]
        u = data.get("usage", {})
        return LLMResponse(text, calls, {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)})


class OpenAICompatLLM:
    name = "openai"

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1",
                 transport: Transport = http_post):
        if not model or not api_key:
            raise LLMError("openai provider needs CG_LLM_MODEL and CG_LLM_API_KEY")
        self.model, self.api_key, self.base_url, self.transport = model, api_key, base_url.rstrip("/"), transport

    @staticmethod
    def convert_messages(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                out.append({"role": "assistant", "content": m.get("content") or None,
                            "tool_calls": [{"id": c["id"], "type": "function",
                                            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                           for c in m["tool_calls"]]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        body = {"model": self.model, "messages": self.convert_messages(system, messages)}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                               "parameters": t["parameters"]}} for t in tools]
        data = self.transport(f"{self.base_url}/chat/completions",
                              {"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"}, body)
        msg = data["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(c["id"], c["function"]["name"], args))
        u = data.get("usage", {})
        return LLMResponse(msg.get("content") or "", calls,
                           {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)})


def make_llm(settings, transport: Transport | None = None) -> LLM:
    p = settings.llm_provider.lower()
    kw = {"transport": transport} if transport else {}
    if p == "mock":
        return MockLLM()
    if p == "anthropic":
        return AnthropicLLM(settings.llm_model, settings.llm_api_key,
                            settings.llm_base_url or "https://api.anthropic.com", **kw)
    if p == "openai":
        return OpenAICompatLLM(settings.llm_model, settings.llm_api_key,
                               settings.llm_base_url or "https://api.openai.com/v1", **kw)
    raise LLMError(f"unknown CG_LLM_PROVIDER '{settings.llm_provider}' (use mock, anthropic or openai)")
