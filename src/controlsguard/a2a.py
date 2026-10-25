"""Minimal Agent2Agent (A2A) surface: an Agent Card plus JSON-RPC `message/send`.

Implements the subset needed for another agent to discover ControlsGuard and ask it a question:
  * GET  /.well-known/agent-card.json   (Agent Card; protocolVersion 0.3.0 shape)
  * POST /a2a   JSON-RPC 2.0 method `message/send` -> completed Task with one text artifact
Streaming, push notifications, task polling and auth schemes are intentionally not implemented.
Check field names against the current A2A spec before integrating with a third-party A2A client.
"""
from __future__ import annotations

import uuid

PROTOCOL_VERSION = "0.3.0"


def agent_card(base_url: str, version: str, requires_auth: bool) -> dict:
    card = {
        "name": "ControlsGuard",
        "description": "Answers questions about company policies, controls, risks and the systems they protect, "
                       "with cited, grounded answers, graph lookups and control-coverage checks.",
        "version": version,
        "url": f"{base_url.rstrip('/')}/a2a",
        "protocolVersion": PROTOCOL_VERSION,
        "preferredTransport": "JSONRPC",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {"id": "policy_qa", "name": "Policy and control Q&A",
             "description": "Grounded answers with citations from the policy corpus.",
             "tags": ["policy", "controls", "rag"],
             "examples": ["How quickly must a leaver's access be removed?"]},
            {"id": "coverage_check", "name": "Control coverage check",
             "description": "Finds risks affecting a system that no protecting control mitigates.",
             "tags": ["graph", "coverage", "risk"],
             "examples": ["Are there any control gaps on ledger-core?"]},
        ],
    }
    if requires_auth:
        card["securitySchemes"] = {"apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}
        card["security"] = [{"apiKey": []}]
    return card


def _error(rid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def handle_jsonrpc(agent, payload) -> dict:
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0" or "method" not in payload:
        return _error(payload.get("id") if isinstance(payload, dict) else None, -32600, "Invalid Request")
    rid = payload.get("id")
    if payload["method"] != "message/send":
        return _error(rid, -32601, f"Method not found: {payload['method']}")
    msg = (payload.get("params") or {}).get("message")
    if not isinstance(msg, dict) or not isinstance(msg.get("parts"), list):
        return _error(rid, -32602, "params.message.parts is required")
    text = " ".join(p.get("text", "") for p in msg["parts"] if isinstance(p, dict) and p.get("kind") == "text").strip()
    if not text:
        return _error(rid, -32602, "message must contain a text part")
    context_id = msg.get("contextId") or None
    resp = agent.chat(text, session_id=context_id)
    task_id = uuid.uuid4().hex
    return {
        "jsonrpc": "2.0", "id": rid,
        "result": {
            "kind": "task", "id": task_id, "contextId": resp.session_id,
            "status": {"state": "completed" if resp.status in ("ok", "abstained") else "failed"},
            "artifacts": [{
                "artifactId": uuid.uuid4().hex, "name": "agent_response",
                "parts": [{"kind": "text", "text": resp.answer}],
                "metadata": {"status": resp.status, "citations": resp.citations, "traceId": resp.trace_id},
            }],
        },
    }
