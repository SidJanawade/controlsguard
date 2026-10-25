# ControlsGuard

A grounded, agentic GenAI assistant that answers questions about company **policies, controls, risks and the
systems they protect**, and finds **control-coverage gaps**. Built as a portfolio project for a GenAI Developer
role in a controls-technology team: it applies hosted foundation models through context engineering, RAG,
Graph RAG and tool-using agents, and treats guardrails, evaluation and observability as first-class.

It runs **offline with no API keys** (a deterministic mock model + local embeddings), and switches to a real
model with three environment variables.

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=src

python -m controlsguard.cli "How quickly must a leaver's access be removed?"
python -m controlsguard.cli "Are there any control gaps on ledger-core?"

python -m unittest discover -s tests -t .      # 70 tests
python evals/run_evals.py                      # quality + safety gates, writes evals/report.md

uvicorn controlsguard.api:create_app --factory --port 8000
python -m controlsguard.mcp_server             # MCP server (stdio); add --http for streamable HTTP
docker compose up --build
```

Use a real model: `CG_LLM_PROVIDER=anthropic|openai`, `CG_LLM_MODEL=<model id>`, `CG_LLM_API_KEY=...`
(`CG_LLM_BASE_URL` for any OpenAI-compatible endpoint, including Gemini's).

## How it works

```
question -> guard_input -> retrieve -> graph_expand -> build_context -> act (LLM <-> tools) -> verify -> respond
              | injection?   | off-topic?                                                      | grounding,
              v blocked      v abstain / clarify                                               v PII, leak check
```

| Stage | What it does |
|---|---|
| `ingestion.py` | Cleans markdown, parses structured fields, **structure-aware chunking** (one control/risk per chunk, sentence-window split with overlap for long sections, contextual header on every chunk) |
| `retrieval.py` | **Hybrid search**: BM25 + dense vectors fused with Reciprocal Rank Fusion. Per-retriever ranks are returned for explainability |
| `embeddings.py` | Offline TF-IDF + SVD embedder behind an `Embedder` protocol; swap in a hosted embedding API without touching anything else |
| `graph.py` | **Knowledge graph** (policy -> control -> risk -> system, owners, related controls), every edge carries chunk-level provenance. Traversal, paths, and `coverage()` gap analysis |
| `context.py` | **Context engineering**: assembles system instructions, tool specs, memory, graph facts and sources into a token budget with explicit priorities, stable `[S#]` citation tags and a per-request budget report |
| `memory.py` | Session/state management with TTL + LRU eviction, token-windowed memory, entity tracking for follow-ups ("what about its related controls?") |
| `tools.py` | Function-calling tools with JSON-schema contracts: `search_policies`, `get_control`, `graph_neighbors`, `assess_coverage` |
| `agent.py` | The orchestrated workflow; tool loop with a hard step cap, argument validation, unknown-tool denial, fail-soft tool errors |
| `guardrails.py` | Input/output/data guardrails (below) |
| `mcp_server.py` | **MCP server** exposing the same tools plus `ask_controlsguard` (agent-as-a-tool) |
| `a2a.py` | **A2A** Agent Card + JSON-RPC `message/send`, so other agents can discover and call it |
| `llm.py` | Provider abstraction: mock, Anthropic (tool use), OpenAI-compatible |
| `workflow.py` / `langgraph_app.py` | A tiny LangGraph-shaped state graph, and an adapter that runs the **same node functions** on real LangGraph |
| `observability.py` | JSON audit log, trace spans, Prometheus metrics at `/metrics` |

### Guardrails and responsible AI

- **Prompt injection** blocked before any retrieval or model call; **indirect injection** (instructions hidden inside retrieved documents or tool output) scrubbed line by line
- **PII redaction** (email, PAN, Aadhaar, SSN, Luhn-validated cards, phone) on input, model output, memory and logs; digit patterns are anchored so a long number can't be partially redacted
- **Grounding check**: every substantive sentence needs a valid citation and lexical support in the cited text, or the answer is replaced by a safe fallback instead of being shown
- **Abstention** on off-topic questions and **clarification** on dangling references
- System-prompt leak detection, tool allow-list, schema validation, API-key auth, input size limits

## Evaluation (`evals/`)

47 labelled cases (see `evals/report.md` after a run) with CI gates: retrieval hit@k/MRR with a BM25 vs vector vs
hybrid ablation, answer correctness + citation correctness, graph and coverage questions, refusals, injection
blocking, PII leakage (including partial leaks), and an indirect-injection test against a deliberately poisoned
document. The run fails the build if any gate regresses, mirroring the corpus's own control C-AI-02 (prompt and
model changes need evaluation evidence).

## Honest limitations

- **The default model is a mock.** It writes extractive, cited answers so the full pipeline is testable offline; it is
  not a language model. Quality numbers describe the retrieval, graph, guardrail and harness layers. To judge answer
  quality with a real LLM, run the eval against `CG_LLM_PROVIDER=anthropic|openai` and extend the dataset.
- The corpus is **synthetic** (6 policies, 19 controls, 8 risks, 6 systems). One gap, `R-VENDOR` on `ledger-core`, is planted
  so the coverage check has something real to find.
- Offline embeddings are lexical-semantic (LSA). With a hosted embedding model, semantic recall (e.g. eval case
  `ret-19`, "root rights" vs "administrator and root credentials") should improve; it is kept in the report as a known miss.
- Guardrails are transparent heuristics (regex, lexical overlap), a first layer rather than a replacement for
  model-based classifiers. The vector store is in memory (swap for pgvector/OpenSearch at scale).
- `langgraph_app.py` and the live Anthropic/OpenAI adapters were **not run against the real services** here (no network
  access to package indexes or APIs in the build environment). The adapters are covered by tests with a fake transport.
  The A2A surface follows the 0.3.0 Agent Card / `message/send` shape; verify against a real A2A client.

## JD skill map

| JD requirement | Where |
|---|---|
| Python for GenAI, preprocessing, scripting | whole project; `ingestion.py` |
| LLMs, tokenisation, embeddings, context windows | `textutil.py`, `embeddings.py`, `context.py` (token budget) |
| Prompt engineering (role, few-shot, step-by-step, format) | `prompts.py` (versioned `PROMPT_VERSION`) |
| Context engineering | `context.py`, `memory.py` |
| RAG: chunking, vector search, semantic + keyword | `ingestion.py`, `retrieval.py`, ablation in evals |
| Knowledge graphs / Graph RAG | `graph.py`, `agent.graph_expand`, `tools.graph_neighbors` |
| Agentic workflows, planning, memory, tool calling | `agent.py`, `workflow.py`, `langgraph_app.py` |
| MCP | `mcp_server.py` |
| A2A | `a2a.py` |
| Agent harness: sessions, state, memory, guardrails | `memory.py`, `guardrails.py`, tool-loop limits |
| Consuming GenAI APIs (OpenAI / Gemini / Claude) | `llm.py` |
| Docker, Git, CI | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` |
| Deployment, monitoring, maintenance | `api.py`, `observability.py`, `/health`, `/metrics` |
| Compliance, privacy, responsible AI | `guardrails.py`, policy corpus `ai-governance.md`, audit log |
| Code review, testing, documentation | 70 tests, eval gates, this README |
| Cloud (AWS/GCP/Azure), LangChain/LlamaIndex, CrewAI/ADK | **not covered**: deploy the container to any of them; frameworks are deliberately avoided to show the underlying mechanics |
