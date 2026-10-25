"""Command line entry point:  python -m controlsguard.cli "which controls protect ledger-core?" """
from __future__ import annotations

import argparse
import json
import logging
import sys

from .agent import ControlsAgent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="controlsguard")
    p.add_argument("question", nargs="?", help="ask one question; omit for an interactive session")
    p.add_argument("--json", action="store_true", help="print the full structured response")
    p.add_argument("--verbose", action="store_true", help="show audit log lines on stderr")
    a = p.parse_args(argv)
    if not a.verbose:
        logging.getLogger("controlsguard.audit").setLevel(logging.CRITICAL)
    agent = ControlsAgent()

    def show(resp) -> None:
        if a.json:
            print(json.dumps(resp.to_dict(), indent=2))
            return
        print(f"\n{resp.answer}\n")
        if resp.citations:
            print("Sources:", "; ".join(f"[{c['tag']}] {c['document']} > {c['section']}" for c in resp.citations))
        if resp.tool_trace:
            print("Tools:", ", ".join(t["tool"] for t in resp.tool_trace))
        if resp.events:
            print("Guardrails:", ", ".join(e["type"] for e in resp.events))

    if a.question:
        show(agent.chat(a.question))
        return 0
    sid = None
    print("ControlsGuard interactive session. Ctrl-D to exit.")
    for line in sys.stdin if not sys.stdin.isatty() else iter(lambda: input("> "), None):
        line = line.strip()
        if not line:
            continue
        resp = agent.chat(line, sid)
        sid = resp.session_id
        show(resp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
