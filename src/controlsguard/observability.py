"""Structured audit logging, request tracing and Prometheus-style metrics (no external deps)."""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager

_audit = logging.getLogger("controlsguard.audit")
if not _audit.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _audit.addHandler(_h)
    _audit.setLevel(logging.INFO)
    _audit.propagate = False


def audit(event: str, **fields) -> None:
    """One JSON line per event. Callers must pass already-redacted content only."""
    _audit.info(json.dumps({"ts": round(time.time(), 3), "event": event, **fields}, default=str))


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


class Trace:
    def __init__(self, trace_id: str | None = None):
        self.trace_id = trace_id or new_trace_id()
        self.spans: list[dict] = []

    @contextmanager
    def span(self, name: str, **attrs):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.spans.append({"name": name, "ms": round((time.perf_counter() - t0) * 1000, 2), **attrs})


class Metrics:
    BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[tuple, float] = defaultdict(float)
        self.hist: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, value: float = 1, **labels) -> None:
        with self._lock:
            self.counters[(name, tuple(sorted(labels.items())))] += value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self.hist[name].append(value)

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), v in sorted(self.counters.items()):
                lab = ",".join(f'{k}="{val}"' for k, val in labels)
                lines.append(f"{name}{{{lab}}} {v}" if lab else f"{name} {v}")
            for name, vals in sorted(self.hist.items()):
                for b in self.BUCKETS:
                    lines.append(f'{name}_bucket{{le="{b}"}} {sum(1 for x in vals if x <= b)}')
                lines.append(f'{name}_bucket{{le="+Inf"}} {len(vals)}')
                lines.append(f"{name}_sum {sum(vals):.4f}")
                lines.append(f"{name}_count {len(vals)}")
        return "\n".join(lines) + "\n"
