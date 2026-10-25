import asyncio
import io
import json
import unittest
from contextlib import redirect_stdout

from controlsguard.api import create_app
from controlsguard.agent import ControlsAgent
from controlsguard.cli import main as cli_main
from controlsguard.config import Settings
from controlsguard.mcp_server import build_server

from .helpers import shared_agent


def asgi(app, method, path, body=None, headers=None, query=b"", raw=None):
    payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else b"")
    sent = []

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(m):
        sent.append(m)

    scope = {"type": "http", "method": method, "path": path, "query_string": query, "scheme": "http", "root_path": "",
             "http_version": "1.1", "server": ("testserver", 80), "client": ("127.0.0.1", 1),
             "headers": [(b"content-type", b"application/json"), (b"host", b"testserver")]
                        + [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]}
    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    data = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body").decode()
    return status, data


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app(shared_agent())

    def test_health_and_metrics(self):
        st, body = asgi(self.app, "GET", "/health")
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(body)["graph"]["by_type"]["control"], 19)
        self.assertEqual(asgi(self.app, "GET", "/metrics")[0], 200)

    def test_chat_happy_path_and_session_continuity(self):
        st, body = asgi(self.app, "POST", "/chat", {"message": "Who owns C-AC-01?", "session_id": "api-1"})
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(body)["status"], "ok")
        st, body = asgi(self.app, "POST", "/chat", {"message": "What about its related controls?", "session_id": "api-1"})
        self.assertIn("C-AC-03", json.loads(body)["answer"])

    def test_chat_input_validation(self):
        self.assertEqual(asgi(self.app, "POST", "/chat", raw=b"not json")[0], 400)
        self.assertEqual(asgi(self.app, "POST", "/chat", {"message": ""})[0], 400)
        self.assertEqual(asgi(self.app, "POST", "/chat", {"message": 5})[0], 400)
        self.assertEqual(asgi(self.app, "POST", "/chat", {"message": "x" * 5000})[0], 413)
        self.assertEqual(asgi(self.app, "POST", "/chat", {"message": "hi", "session_id": "../etc"})[0], 400)

    def test_search_modes(self):
        for mode in ("hybrid", "bm25", "vector"):
            st, body = asgi(self.app, "GET", "/search", query=f"q=regulator+notification&mode={mode}".encode())
            self.assertEqual(st, 200)
            self.assertTrue(json.loads(body)["results"])
        self.assertEqual(asgi(self.app, "GET", "/search", query=b"q=x&mode=bad")[0], 400)
        self.assertEqual(asgi(self.app, "GET", "/search")[0], 400)

    def test_graph_and_coverage_endpoints(self):
        st, body = asgi(self.app, "GET", "/graph/C-AC-01")
        self.assertEqual(st, 200)
        self.assertTrue(any(r["relation"] == "mitigates" for r in json.loads(body)["relations"]))
        self.assertEqual(asgi(self.app, "GET", "/graph/unknown-thing")[0], 404)
        st, body = asgi(self.app, "GET", "/coverage/ledger-core")
        self.assertEqual(json.loads(body)["gaps"], ["R-VENDOR"])
        self.assertEqual(asgi(self.app, "GET", "/coverage/nowhere")[0], 404)

    def test_api_key_enforced_when_configured(self):
        app = create_app(ControlsAgent(Settings(api_key="s3cret")))
        self.assertEqual(asgi(app, "POST", "/chat", {"message": "hi"})[0], 401)
        self.assertEqual(asgi(app, "POST", "/chat", {"message": "hi"}, headers={"X-API-Key": "wrong"})[0], 401)
        self.assertEqual(asgi(app, "POST", "/chat", {"message": "Who owns C-AC-01?"}, headers={"X-API-Key": "s3cret"})[0], 200)
        self.assertEqual(asgi(app, "GET", "/health")[0], 200)  # liveness stays open
        card = json.loads(asgi(app, "GET", "/.well-known/agent-card.json")[1])
        self.assertIn("securitySchemes", card)


class A2ATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app(shared_agent())

    def rpc(self, payload):
        st, body = asgi(self.app, "POST", "/a2a", payload)
        self.assertEqual(st, 200)
        return json.loads(body)

    def test_agent_card(self):
        st, body = asgi(self.app, "GET", "/.well-known/agent-card.json")
        card = json.loads(body)
        self.assertEqual(st, 200)
        self.assertEqual(card["preferredTransport"], "JSONRPC")
        self.assertEqual(card["url"], "http://testserver/a2a")
        self.assertTrue(card["skills"])

    def test_message_send_returns_completed_task_with_artifact(self):
        out = self.rpc({"jsonrpc": "2.0", "id": "r1", "method": "message/send", "params": {"message": {
            "role": "user", "messageId": "m1", "parts": [{"kind": "text", "text": "Are there any control gaps on ledger-core?"}]}}})
        res = out["result"]
        self.assertEqual(out["id"], "r1")
        self.assertEqual(res["kind"], "task")
        self.assertEqual(res["status"]["state"], "completed")
        self.assertIn("R-VENDOR", res["artifacts"][0]["parts"][0]["text"])

    def test_context_id_continues_the_conversation(self):
        first = self.rpc({"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": {
            "role": "user", "messageId": "a", "parts": [{"kind": "text", "text": "Who owns C-AC-01?"}]}}})
        ctx = first["result"]["contextId"]
        second = self.rpc({"jsonrpc": "2.0", "id": 2, "method": "message/send", "params": {"message": {
            "role": "user", "messageId": "b", "contextId": ctx, "parts": [{"kind": "text", "text": "What about its related controls?"}]}}})
        self.assertIn("C-AC-03", second["result"]["artifacts"][0]["parts"][0]["text"])

    def test_protocol_errors(self):
        self.assertEqual(self.rpc({"jsonrpc": "2.0", "id": 1, "method": "tasks/get"})["error"]["code"], -32601)
        self.assertEqual(self.rpc({"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {}})["error"]["code"], -32602)
        self.assertEqual(self.rpc({"id": 1})["error"]["code"], -32600)
        st, body = asgi(self.app, "POST", "/a2a", raw=b"{bad")
        self.assertEqual(json.loads(body)["error"]["code"], -32700)


class McpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = build_server(shared_agent())

    def text(self, name, args):
        res = asyncio.run(self.server.call_tool(name, args))
        return res.content[0].text

    def test_tools_are_listed_with_schemas(self):
        tools = asyncio.run(self.server.list_tools())
        names = {t.name for t in tools}
        self.assertEqual(names, {"search_policies", "get_control", "graph_neighbors", "assess_coverage", "ask_controlsguard"})

    def test_tool_calls(self):
        self.assertIn("GAP: risk R-VENDOR", self.text("assess_coverage", {"system": "ledger-core"}))
        self.assertIn("Quarterly access recertification", self.text("get_control", {"control_id": "C-AC-01"}))
        self.assertIn("C-AC-02", self.text("graph_neighbors", {"entity": "R-INSIDER"}))
        self.assertIn("72 hours", self.text("ask_controlsguard", {"question": "What is the regulator notification deadline after a breach?"}))

    def test_unknown_ids_fail_soft(self):
        self.assertIn("Unknown id", self.text("get_control", {"control_id": "C-XX-99"}))


class CliTests(unittest.TestCase):
    def test_cli_answers_and_prints_sources(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli_main(["Who owns C-AC-01?"])
        self.assertEqual(code, 0)
        self.assertIn("Head of Identity and Access Management", buf.getvalue())
        self.assertIn("Sources:", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
