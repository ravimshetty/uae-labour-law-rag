"""Build the knowledge base from data/sources.yaml."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

from ..config import ROOT, Settings
from ..index.embeddings import build_embedder
from ..index.store import KnowledgeBase
from .chunker import articles_to_chunks, split_articles
from .loaders import load_document

log = logging.getLogger(__name__)


def build_knowledge_base(settings: Settings, sources_path: str | Path | None = None) -> KnowledgeBase:
    sources_path = Path(sources_path or settings.path("paths.sources"))
    docs = yaml.safe_load(sources_path.read_text("utf-8"))["documents"]
    parents, children = [], []
    for doc in docs:
        file = Path(doc["file"])
        file = file if file.is_absolute() else ROOT / file
        if not file.exists():
            level = logging.INFO if doc.get("optional") else logging.WARNING
            log.log(level, "skip %s – file not found: %s", doc["doc_id"], file)
            continue
        text = load_document(file, doc["lang"])
        arts = split_articles(text, doc["article_pattern"], doc.get("stop_pattern"),
                              doc.get("skip_until_pattern"))
        p, c = articles_to_chunks(arts, doc=doc)
        log.info("%s: %d articles, %d chunks", doc["doc_id"], len(p), len(c))
        parents += p
        children += c
    if not children:
        raise RuntimeError("No documents ingested – put the PDFs in data/raw or use data/sample/sources.yaml")
    emb = build_embedder(settings.get("embeddings.provider"), settings.get("embeddings.model"),
                         settings.get("embeddings.dim", 1024))
    return KnowledgeBase.build(parents, children, emb)
