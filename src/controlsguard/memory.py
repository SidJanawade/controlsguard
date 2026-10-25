"""Session state and conversation memory for the agent harness."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from .textutil import count_tokens


@dataclass
class Turn:
    user: str
    assistant: str
    ts: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str
    turns: list[Turn] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)  # entities discussed (working memory)
    last_seen: float = field(default_factory=time.time)

    def add_turn(self, user: str, assistant: str, entities: list[str] | None = None) -> None:
        self.turns.append(Turn(user, assistant))
        for e in entities or []:
            if e in self.entities:
                self.entities.remove(e)
            self.entities.append(e)
        self.entities = self.entities[-8:]
        self.last_seen = time.time()

    def window(self, max_tokens: int) -> list[dict]:
        """Most recent turns that fit in `max_tokens`; older turns collapse into one summary line."""
        kept: list[Turn] = []
        used = 0
        for t in reversed(self.turns):
            cost = count_tokens(t.user) + count_tokens(t.assistant)
            if used + cost > max_tokens and kept:
                break
            if used + cost > max_tokens:  # a single huge turn: keep only the question
                cost = count_tokens(t.user)
                kept.append(Turn(t.user, "(answer omitted)"))
            else:
                kept.append(t)
            used += cost
        kept.reverse()
        dropped = self.turns[: len(self.turns) - len(kept)]
        msgs: list[dict] = []
        if dropped:
            topics = "; ".join(t.user[:60] for t in dropped[-4:])
            msgs.append({"role": "user", "content": f"(Earlier in this conversation the user asked: {topics})"})
            msgs.append({"role": "assistant", "content": "Understood."})
        for t in kept:
            msgs.append({"role": "user", "content": t.user})
            msgs.append({"role": "assistant", "content": t.assistant})
        return msgs


class SessionStore:
    def __init__(self, ttl_s: int = 3600, max_sessions: int = 1000):
        self.ttl_s, self.max_sessions = ttl_s, max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str | None) -> Session:
        with self._lock:
            self._evict()
            sid = session_id or uuid.uuid4().hex[:12]
            if sid not in self._sessions:
                self._sessions[sid] = Session(sid)
            s = self._sessions[sid]
            s.last_seen = time.time()
            return s

    def _evict(self) -> None:
        now = time.time()
        for sid in [k for k, v in self._sessions.items() if now - v.last_seen > self.ttl_s]:
            del self._sessions[sid]
        while len(self._sessions) >= self.max_sessions:
            oldest = min(self._sessions, key=lambda k: self._sessions[k].last_seen)
            del self._sessions[oldest]

    def __len__(self) -> int:
        return len(self._sessions)
