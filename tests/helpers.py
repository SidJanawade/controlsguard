from __future__ import annotations

from controlsguard.agent import ControlsAgent
from controlsguard.llm import LLMResponse, ToolCall

_AGENT: ControlsAgent | None = None


def shared_agent() -> ControlsAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = ControlsAgent()
    return _AGENT


class ScriptedLLM:
    """Returns pre-programmed responses so agent-harness behaviour can be tested without a model."""

    name = "scripted"

    def __init__(self, *responses: LLMResponse):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system, messages, tools):
        self.calls.append({"system": system, "messages": list(messages), "tools": [t["name"] for t in tools]})
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id=f"c_{name}", name=name, arguments=arguments)
