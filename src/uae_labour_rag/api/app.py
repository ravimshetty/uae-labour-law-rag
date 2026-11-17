"""FastAPI service + static web UI.

Run:  uvicorn uae_labour_rag.api.app:app --app-dir src --reload --port 8000
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..pipeline import RAGPipeline, get_pipeline

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="UAE Labour & DIFC Employment Law – EN→AR RAG", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, examples=["How much gratuity after 7 years?"])
    jurisdiction: str | None = Field(None, pattern="^(federal|difc|both)$")
    debug: bool = False


@lru_cache(maxsize=1)
def pipeline() -> RAGPipeline:
    return get_pipeline()


# --- naive per-IP rate limiting (use API gateway / Redis in production) ---------
_WINDOW, _LIMIT = 60.0, 30
_hits: dict[str, deque] = defaultdict(deque)


def _rate_limit(ip: str) -> None:
    now, q = time.time(), _hits[ip]
    while q and now - q[0] > _WINDOW:
        q.popleft()
    if len(q) >= _LIMIT:
        raise HTTPException(429, "Too many requests")
    q.append(now)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.update({
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-store",   # answers may relate to personal situations
    })
    return resp


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    p = pipeline()
    return {"status": "ok", "articles": len(p.kb.parents), "chunks": len(p.kb.children), "llm": p.llm.name}


@app.post("/api/ask")
def ask(req: AskRequest, request: Request) -> dict:
    _rate_limit(request.client.host if request.client else "anon")
    return pipeline().answer(req.query, req.jurisdiction, debug=req.debug).to_dict()
