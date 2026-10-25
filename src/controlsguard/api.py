"""HTTP API (Starlette). Run:  uvicorn controlsguard.api:app --host 0.0.0.0 --port 8000

Endpoints
  POST /chat                          grounded agent answer (session memory via session_id)
  GET  /search?q=&k=&mode=            raw hybrid retrieval (bm25 | vector | hybrid)
  GET  /graph/{entity}                knowledge-graph neighbourhood
  GET  /coverage/{system}             control coverage / gap analysis
  GET  /health, GET /metrics          liveness and Prometheus metrics
  GET  /.well-known/agent-card.json   A2A Agent Card;  POST /a2a  A2A message/send
Set CG_API_KEY to require an X-API-Key header on everything except health, metrics and the agent card.
"""
from __future__ import annotations

import hmac
import re

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from . import __version__
from .a2a import agent_card, handle_jsonrpc
from .agent import ControlsAgent
from .prompts import PROMPT_VERSION

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def create_app(agent: ControlsAgent | None = None) -> Starlette:
    agent = agent or ControlsAgent()
    cfg = agent.settings

    def authorized(request: Request) -> bool:
        if not cfg.api_key:
            return True
        return hmac.compare_digest(request.headers.get("x-api-key", ""), cfg.api_key)

    def deny() -> JSONResponse:
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    def bad(msg: str, code: int = 400) -> JSONResponse:
        return JSONResponse({"error": msg}, status_code=code)

    async def health(request: Request):
        return JSONResponse({
            "status": "ok", "version": __version__, "prompt_version": PROMPT_VERSION,
            "llm": getattr(agent.llm, "name", "custom"), "chunks": len(agent.chunks),
            "graph": agent.graph.stats(), "active_sessions": len(agent.sessions),
        })

    async def metrics(request: Request):
        return PlainTextResponse(agent.metrics.render())

    async def chat(request: Request):
        if not authorized(request):
            return deny()
        try:
            body = await request.json()
        except Exception:
            return bad("body must be valid JSON")
        msg = body.get("message") if isinstance(body, dict) else None
        if not isinstance(msg, str) or not msg.strip():
            return bad("'message' (non-empty string) is required")
        if len(msg) > cfg.max_message_chars:
            return bad(f"'message' exceeds {cfg.max_message_chars} characters", 413)
        sid = body.get("session_id")
        if sid is not None and (not isinstance(sid, str) or not SESSION_RE.match(sid)):
            return bad("'session_id' must match [A-Za-z0-9_-]{1,64}")
        resp = await run_in_threadpool(agent.chat, msg, sid)
        return JSONResponse(resp.to_dict())

    async def search(request: Request):
        if not authorized(request):
            return deny()
        q = request.query_params.get("q", "").strip()
        if not q or len(q) > 500:
            return bad("'q' (1-500 chars) is required")
        mode = request.query_params.get("mode", "hybrid")
        if mode not in ("hybrid", "bm25", "vector"):
            return bad("mode must be hybrid, bm25 or vector")
        try:
            k = max(1, min(int(request.query_params.get("k", 4)), 10))
        except ValueError:
            return bad("k must be an integer")
        hits = agent.retriever.search(q, k=k, mode=mode)
        return JSONResponse({"query": q, "mode": mode, "results": [
            {"chunk_id": h.chunk.id, "entity": h.chunk.entity_id, "header": h.chunk.header,
             "score": round(h.score, 5), "ranks": h.ranks, "text": h.chunk.text} for h in hits]})

    async def graph(request: Request):
        if not authorized(request):
            return deny()
        ents = agent.graph.resolve(request.path_params["entity"])
        if not ents:
            return bad("unknown entity", 404)
        e = ents[0]
        triples = agent.graph.neighbors(e, depth=1, limit=60)
        return JSONResponse({"entity": e, "type": agent.graph.type_of(e), "label": agent.graph.label(e),
                             "relations": [{"subject": t.subject, "relation": t.relation, "object": t.object,
                                            "source_chunk": t.chunk_id} for t in triples]})

    async def coverage(request: Request):
        if not authorized(request):
            return deny()
        systems = [e for e in agent.graph.resolve(request.path_params["system"]) if agent.graph.type_of(e) == "system"]
        if not systems:
            return bad(f"unknown system; known: {', '.join(agent.graph.nodes_of_type('system'))}", 404)
        return JSONResponse(agent.graph.coverage(systems[0]))

    async def card(request: Request):
        base = str(request.base_url)
        return JSONResponse(agent_card(base, __version__, bool(cfg.api_key)))

    async def a2a(request: Request):
        if not authorized(request):
            return deny()
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
        return JSONResponse(await run_in_threadpool(handle_jsonrpc, agent, payload))

    return Starlette(routes=[
        Route("/health", health), Route("/metrics", metrics),
        Route("/chat", chat, methods=["POST"]), Route("/search", search),
        Route("/graph/{entity}", graph), Route("/coverage/{system}", coverage),
        Route("/.well-known/agent-card.json", card), Route("/.well-known/agent.json", card),
        Route("/a2a", a2a, methods=["POST"]),
    ])


def __getattr__(name: str):  # lazy `app` so importing the module never builds the index as a side effect
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
