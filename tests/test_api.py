"""HTTP-level tests for the FastAPI app (skipped if fastapi/httpx are not installed)."""
import pytest

pytest.importorskip("fastapi", exc_type=ImportError)  # DLL blocks raise ImportError
pytest.importorskip("httpx", exc_type=ImportError)
from fastapi.testclient import TestClient  # noqa: E402

from uae_labour_rag.api import app as app_module  # noqa: E402


@pytest.fixture()
def client(pipe):
    app_module.pipeline.cache_clear()
    app_module.app.dependency_overrides = {}
    original = app_module.pipeline
    app_module.pipeline = lambda: pipe          # use the sample-corpus pipeline
    yield TestClient(app_module.app)
    app_module.pipeline = original


def test_health_and_index(client):
    assert client.get("/api/health").json()["status"] == "ok"
    r = client.get("/")
    assert r.status_code == 200 and "dir=" not in r.headers.get("content-type", "")
    assert r.headers["x-frame-options"] == "DENY"


def test_ask_returns_arabic_with_citations(client):
    r = client.post("/api/ask", json={"query": "how many days ijaza in DIFC?"}).json()
    assert r["jurisdiction"] == "difc" and r["citations"][0]["article"] == "27"


def test_validation(client):
    assert client.post("/api/ask", json={"query": "hi", "jurisdiction": "mars"}).status_code == 422
