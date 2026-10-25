"""Tokenisation helpers shared by lexical search, the mock LLM and the guardrails."""
from __future__ import annotations

import math
import re

_WORD = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
STOPWORDS = frozenset(
    """a an the and or of to in on for with by from at as is are was were be been being it its this that these those
    what which who whom whose how why when where do does did can could should would will shall may might must
    i we you they he she them our your their about into over under per than then so if not no any all each
    me my us there here also such via""".split()
)


_CANON = {"own": "owner", "owns": "owner", "owned": "owner", "owning": "owner", "approval": "approv",
          "approvals": "approv"}


def stem(tok: str) -> str:
    """Very light suffix stripping so approves/approved/approval-ish forms and plurals line up."""
    tok = _CANON.get(tok, tok)
    if len(tok) <= 4 or not tok.isalpha():
        return tok
    if tok.endswith("ies") and len(tok) > 5:
        return tok[:-3] + "y"
    for suf in ("ing", "ed", "es", "s", "e"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            if suf == "s" and tok.endswith("ss"):
                continue
            return tok[: -len(suf)]
    return tok


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Lower-case, lightly stemmed word tokens. Hyphenated ids (c-ac-01, ledger-core) are kept whole
    and their alphabetic parts of 3+ chars are added too, so 'ledger core' still matches 'ledger-core'."""
    out: list[str] = []
    for m in _WORD.finditer(text.lower()):
        tok = m.group(0)
        if not keep_stopwords and tok in STOPWORDS:
            continue
        if "-" in tok or "_" in tok:
            out.append(tok)
            for part in re.split(r"[-_]", tok):
                if len(part) >= 3 and part.isalpha() and (keep_stopwords or part not in STOPWORDS):
                    out.append(stem(part))
        else:
            out.append(stem(tok))
    return out


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if not t.isdigit() or len(t) > 1}


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def count_tokens(text: str) -> int:
    """Approximate LLM token count (~4 chars/token). Uses tiktoken when installed."""
    try:  # pragma: no cover - optional dependency
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 4))
