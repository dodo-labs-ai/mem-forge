from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from memforge.api_target import build_target
from memforge.tool_client import ToolClient


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/auth/jira-session"):
            self._send(200, {"provider": "jira", "origin": "https://jira.example.test", "status": "active"})
        elif self.path == "/api/auth/jira-origins":
            self._send(200, {"origins": [{"origin": "https://jira.example.test", "status": "active"}]})
        else:
            self._send(404, {"error": "nope"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/auth/jira-session":
            self._send(200, {"provider": "jira", "origin": body["base_url"], "status": "active"})
        elif self.path == "/api/auth/jira-session/expire":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "nope"})

    def do_DELETE(self):
        if self.path.startswith("/api/auth/jira-session"):
            self._send(200, {"ok": True, "forgotten": True})
        else:
            self._send(404, {"error": "nope"})


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_tool_client_jira_session_round_trip(server):
    client = ToolClient(
        target=build_target(edition="oss", origin=server, workspace_id=None),
        api_token=None,
    )
    assert client.get_jira_session("https://jira.example.test")["status"] == "active"
    assert client.list_jira_origins()["origins"][0]["origin"] == "https://jira.example.test"
    up = client.upload_jira_session(base_url="https://jira.example.test", cookie_header="SESSION=x", browser="Chrome")
    assert up["status"] == "active"
    assert client.mark_jira_session_expired(base_url="https://jira.example.test", error="dead")["ok"] is True
    assert client.forget_jira_session("https://jira.example.test")["forgotten"] is True
