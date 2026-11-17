"""Zero-dependency web server (Python standard library only).

Same UI and same /api/ask + /api/health contract as the FastAPI app, for machines
where compiled packages (pydantic-core, numpy, ...) are blocked by Windows Smart
App Control / WDAC.

    python scripts/serve.py              # http://localhost:8000
    python scripts/serve.py --port 8080
"""
import argparse
import json
import sys
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uae_labour_rag.pipeline import get_pipeline  # noqa: E402

INDEX_HTML = ROOT / "src/uae_labour_rag/api/static/index.html"
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
}
WINDOW, LIMIT = 60.0, 30
HITS: dict[str, deque] = defaultdict(deque)
PIPE = None


def rate_limited(ip: str) -> bool:
    now, q = time.time(), HITS[ip]
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= LIMIT:
        return True
    q.append(now)
    return False


class Handler(BaseHTTPRequestHandler):
    server_version = "uae-labour-rag"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._json(200, {"status": "ok", "articles": len(PIPE.kb.parents),
                             "chunks": len(PIPE.kb.children), "llm": PIPE.llm.name})
        else:
            self._json(404, {"detail": "Not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/api/ask":
            return self._json(404, {"detail": "Not found"})
        if rate_limited(self.client_address[0]):
            return self._json(429, {"detail": "Too many requests"})
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 20_000)
            body = json.loads(self.rfile.read(length) or b"{}")
            query = str(body.get("query", "")).strip()
            jur = body.get("jurisdiction")
            if not 3 <= len(query) <= 2000 or jur not in (None, "federal", "difc", "both"):
                return self._json(422, {"detail": "query must be 3-2000 chars; jurisdiction federal|difc|both"})
            res = PIPE.answer(query, jur, debug=bool(body.get("debug")), verify=bool(body.get("verify")))
            self._json(200, res.to_dict())
        except json.JSONDecodeError:
            self._json(400, {"detail": "invalid JSON"})
        except Exception as e:  # noqa: BLE001 – never leak stack traces / query text
            self._json(500, {"detail": f"internal error: {type(e).__name__}"})

    def log_message(self, fmt, *args):  # no query text in access logs
        sys.stderr.write(f"{self.address_string()} {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")


def main() -> None:
    global PIPE
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    PIPE = get_pipeline()
    print(f"Serving on http://{a.host}:{a.port}  (LLM: {PIPE.llm.name}, articles: {len(PIPE.kb.parents)})")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
