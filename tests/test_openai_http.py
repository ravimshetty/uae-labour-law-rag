"""Stdlib OpenAI/Azure client against a local fake server (no network, no SDK)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from uae_labour_rag.generation.llm import OpenAIChat
from uae_labour_rag.index.embeddings import OpenAIEmbedder

SEEN = []


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        if self.path.endswith("/chat/completions") or "/chat/completions?" in self.path:
            out = {"choices": [{"message": {"content": "إجابة [S1]\nGIST_EN: ok"}}]}
        else:
            out = {"data": [{"index": i, "embedding": [3.0, 4.0]} for i, _ in enumerate(body["input"])]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_openai_chat_and_embeddings(server, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", server + "/v1")
    assert OpenAIChat("gpt-4o-mini").complete("sys", "user").startswith("إجابة")
    path, headers, body = SEEN[-1]
    assert path == "/v1/chat/completions" and headers["authorization"] == "Bearer sk-test"
    assert body["model"] == "gpt-4o-mini" and body["messages"][0]["role"] == "system"
    vecs = OpenAIEmbedder("text-embedding-3-large").embed(["a", "b"])
    assert vecs == [[0.6, 0.8], [0.6, 0.8]]          # L2-normalised


def test_azure_routing(server, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", server)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    OpenAIChat("my-gpt4o-deployment", azure=True).complete("s", "u")
    path, headers, body = SEEN[-1]
    assert path.startswith("/openai/deployments/my-gpt4o-deployment/chat/completions?api-version=")
    assert headers["api-key"] == "az-key" and "model" not in body


def test_missing_key_is_clear(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIChat("gpt-4o-mini")
