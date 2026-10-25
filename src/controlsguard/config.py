"""Runtime settings. Everything is overridable by environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("CG_DATA_DIR", ROOT / "data" / "policies")))
    # LLM: "mock" (offline, deterministic), "anthropic", or "openai" (OpenAI-compatible, incl. Gemini endpoint)
    llm_provider: str = field(default_factory=lambda: os.getenv("CG_LLM_PROVIDER", "mock"))
    llm_model: str = field(default_factory=lambda: os.getenv("CG_LLM_MODEL", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("CG_LLM_BASE_URL", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("CG_LLM_API_KEY", ""))
    # Context engineering
    context_budget_tokens: int = field(default_factory=lambda: _env_int("CG_CONTEXT_BUDGET", 1800))
    top_k: int = field(default_factory=lambda: _env_int("CG_TOP_K", 4))
    chunk_max_tokens: int = field(default_factory=lambda: _env_int("CG_CHUNK_MAX_TOKENS", 220))
    chunk_overlap_tokens: int = field(default_factory=lambda: _env_int("CG_CHUNK_OVERLAP", 40))
    # Agent harness
    max_tool_steps: int = field(default_factory=lambda: _env_int("CG_MAX_TOOL_STEPS", 3))
    session_ttl_s: int = field(default_factory=lambda: _env_int("CG_SESSION_TTL", 3600))
    # API
    api_key: str = field(default_factory=lambda: os.getenv("CG_API_KEY", ""))
    max_message_chars: int = field(default_factory=lambda: _env_int("CG_MAX_MESSAGE_CHARS", 2000))
