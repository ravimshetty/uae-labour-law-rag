import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uae_labour_rag.config import load_settings  # noqa: E402
from uae_labour_rag.ingest.build import build_knowledge_base  # noqa: E402
from uae_labour_rag.pipeline import RAGPipeline  # noqa: E402


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rag")
    return load_settings(overrides={
        "llm.provider": "mock", "embeddings.provider": "hashing",
        "paths.audit_log": str(tmp / "audit.jsonl"), "paths.index_dir": str(tmp / "index"),
    })


@pytest.fixture(scope="session")
def kb(settings):
    return build_knowledge_base(settings, ROOT / "data/sample/sources.yaml")


@pytest.fixture(scope="session")
def pipe(settings, kb):
    return RAGPipeline(settings, kb=kb)
