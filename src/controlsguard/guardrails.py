"""Guardrails and responsible-AI checks.

Input:  PII redaction, prompt-injection detection, length limits.
Data:   indirect prompt-injection scrubbing of retrieved chunks.
Output: citation validity, grounding (lexical support) check, PII redaction, system-prompt leak check.

These are deliberately simple, transparent heuristics (regex + lexical overlap). They are a first layer,
not a substitute for model-based classifiers; the eval harness measures how well they hold up.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .textutil import content_tokens, sentences

# ---- PII -------------------------------------------------------------------------------------
# Digit-run patterns are anchored so they can never match a slice of a longer number: a 16-digit card
# must not be half-consumed by the 12-digit Aadhaar pattern (that would leave the tail in the text).
_NO_DIGIT_BEFORE = r"(?<!\d)(?<!\d[ -])"
_NO_DIGIT_AFTER = r"(?![ -]?\d)"
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("CARD", re.compile(_NO_DIGIT_BEFORE + r"(?:\d[ -]?){12}\d(?:[ -]?\d){0,6}" + _NO_DIGIT_AFTER)),
    ("SSN", re.compile(_NO_DIGIT_BEFORE + r"\d{3}-\d{2}-\d{4}" + _NO_DIGIT_AFTER)),
    ("AADHAAR", re.compile(_NO_DIGIT_BEFORE + r"\d{4}[ -]?\d{4}[ -]?\d{4}" + _NO_DIGIT_AFTER)),
    ("PHONE", re.compile(r"(?<![\w+])" + _NO_DIGIT_BEFORE + r"(?:\+\d{1,3}[ -]?)?\d[\d -]{8,14}\d" + _NO_DIGIT_AFTER)),
]


def _luhn_ok(digits: str) -> bool:
    ds = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(ds) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(ds)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass
class Redaction:
    text: str
    findings: list[str] = field(default_factory=list)  # PII types found (never the values)


def redact_pii(text: str) -> Redaction:
    findings: list[str] = []
    out = text
    for name, pat in _PII_PATTERNS:
        def repl(m: re.Match, name=name) -> str:
            raw = m.group(0)
            if name == "CARD" and not _luhn_ok(raw):
                return raw
            if name == "PHONE" and not 10 <= len(re.sub(r"\D", "", raw)) <= 13:
                return raw
            findings.append(name)
            return f"[REDACTED_{name}]"

        out = pat.sub(repl, out)
    return Redaction(out, findings)


# ---- Prompt injection ------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|rules|prompts?)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|earlier|your) (?:instructions|rules|prompts?)",
    r"(?:reveal|show|print|repeat|display|output) (?:me )?(?:your|the) (?:system|hidden|initial|original) (?:prompt|instructions?|message)",
    r"what (?:is|are) your (?:system prompt|instructions|rules)",
    r"you are now (?:in )?(?:dan|developer mode|an? unrestricted|jailbroken)",
    r"(?:act|pretend) (?:as|to be) (?:an? )?(?:unrestricted|jailbroken|evil|different) ",
    r"\bjailbreak\b",
    r"override (?:the |your )?(?:safety|guardrails?|rules|policy)",
    r"new instructions?\s*:",
    r"</?(?:system|sources|graph_facts)>",
    r"do not (?:cite|follow the rules)",
]
_INJECTION_RE = [re.compile(p, re.I) for p in _INJECTION_PATTERNS]


def detect_injection(text: str) -> list[str]:
    return [p.pattern for p in _INJECTION_RE if p.search(text)]


def scrub_retrieved(text: str) -> tuple[str, list[str]]:
    """Indirect injection: neutralise instruction-like lines inside retrieved documents."""
    hits: list[str] = []
    kept: list[str] = []
    for line in text.splitlines():
        if detect_injection(line):
            hits.append(line.strip()[:80])
            kept.append("[line removed by guardrail: instruction-like content in document]")
        else:
            kept.append(line)
    return "\n".join(kept), hits


# ---- Output checks ---------------------------------------------------------------------------
CITE_RE = re.compile(r"\[(S\d+)\]")
# Words that express graph relations ("is protected by", "mitigates"...). They are the reasoning
# connective between cited facts, not facts themselves, so they don't count against grounding.
RELATION_WORDS = frozenset({"protect", "mitigat", "affect", "relat", "cover", "covered", "gap", "risk",
                            "found", "control"})


@dataclass
class GroundingResult:
    ok: bool
    support: float
    invalid_citations: list[str]
    unsupported_sentences: list[str]
    cited: list[str]


def check_grounding(answer: str, evidence: dict[str, str], min_support: float = 0.55) -> GroundingResult:
    """`evidence` maps source tag -> text. Every substantive sentence must (a) carry a valid citation
    and (b) have most of its content words present in the cited evidence."""
    cited = sorted(set(CITE_RE.findall(answer)), key=lambda t: int(t[1:]))
    invalid = [t for t in cited if t not in evidence]
    unsupported: list[str] = []
    supports: list[float] = []
    body = re.sub(r"^(Evidence|Gaps)\s*:.*$", "", answer, flags=re.M | re.I)
    body = re.sub(r"^Answer\s*:\s*", "", body, flags=re.M | re.I)
    for s in sentences(body):
        words = content_tokens(CITE_RE.sub("", s)) - RELATION_WORDS
        if not words:
            continue
        tags = CITE_RE.findall(s)
        if not tags or any(t not in evidence for t in tags):
            unsupported.append(s[:100])
            supports.append(0.0)
            continue
        ev = set()
        for t in tags:
            ev |= content_tokens(evidence[t])
        sup = len(words & ev) / len(words)
        supports.append(sup)
        if sup < min_support:
            unsupported.append(s[:100])
    support = sum(supports) / len(supports) if supports else 0.0
    ok = bool(cited) and not invalid and not unsupported and bool(supports)
    return GroundingResult(ok, round(support, 3), invalid, unsupported, cited)


def leaks_system_prompt(answer: str, system_prompt: str) -> bool:
    """Flag if the answer reproduces a long verbatim run of the system prompt."""
    words = system_prompt.split()
    low = " ".join(answer.lower().split())
    for i in range(0, max(len(words) - 8, 0), 4):
        if " ".join(words[i : i + 8]).lower() in low:
            return True
    return False


# ---- Tool-call validation ---------------------------------------------------------------------
def validate_tool_args(schema: dict, args: dict) -> str | None:
    """Minimal JSON-schema check (object, required, string/integer types, no unknown properties)."""
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in args:
            return f"missing required argument '{req}'"
    for k, v in args.items():
        if k not in props:
            return f"unexpected argument '{k}'"
        t = props[k].get("type")
        if t == "string" and not isinstance(v, str):
            return f"argument '{k}' must be a string"
        if t == "integer" and (not isinstance(v, int) or isinstance(v, bool)):
            return f"argument '{k}' must be an integer"
        if t == "string" and len(v) > 500:
            return f"argument '{k}' too long"
    return None
